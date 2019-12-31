from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT, WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform, \
    TestBaseTransform
from data import *
import torch.utils.data as data
from face_ssd import build_ssd
# from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings("ignore")

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')
parser.add_argument('file', type=str,
                    help="Video file path")
parser.add_argument('out', type=str,
                    help='Output video path')
parser.add_argument('--vertical', type=int, default=0,
                    help='0 : horizontal video(default), 1 : vertical video')
parser.add_argument('--verbose', type=int, default=0,
                    help='Show current progress and remaining time')
parser.add_argument('--reduce_scale', type=float, default=2,
                    help='Reduce scale ratio. ex) 2 = half size of the input. Default : 2')

parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--threshold', default=0.1, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of WIDERFACE root directory')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

torch.set_grad_enabled(False)


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue

        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])

        if type(max_score) == torch.Tensor:
            max_score = max_score.cpu().numpy()

        norm_factor = np.sum(det_accu[:, -1:])

        if type(norm_factor) == torch.Tensor:
            norm_factor = norm_factor.cpu().numpy()

        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / norm_factor
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets


def infer(net, img, transform, thresh, cuda, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0), volatile=True)
    if cuda:
        x = x.cuda()
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1] / shrink, img.shape[0] / shrink,
                          img.shape[1] / shrink, img.shape[0] / shrink])
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            # label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [[0.1, 0.1, 0.2, 0.2, 0.01]]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


def infer_flip(net, img, transform, thresh, cuda, shrink):
    img = cv2.flip(img, 1)
    det = infer(net, img, transform, thresh, cuda, shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


def infer_multi_scale_sfd(net, img, transform, thresh, cuda, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net, img, transform, thresh, cuda, st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net, img, transform, thresh, cuda, bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, max_im_shrink)))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b

def blur_faces(im,  dets , thresh=0.5, blur_level=100):
    """Blur detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        top = max(int(bbox[1]), 0)
        left = max(int(bbox[0]), 0)
        bottom = min(int(bbox[3]), im.shape[0]-1)
        right = min(int(bbox[2]), im.shape[1]-1)

        im[top:bottom, left:right] = cv2.blur(im[top:bottom, left:right], (blur_level, blur_level))


def detect_faces(img, net, cfg=widerface_640):
    # evaluation
    cuda = args.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh = cfg['conf_thresh']

    max_im_shrink = ((2000.0 * 2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det = infer(net, img, transform, thresh, cuda, shrink)

    return det


if __name__ == '__main__':
    cap = cv2.VideoCapture(args.file)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if args.vertical == 1:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale))
    else:
        out_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // args.reduce_scale),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // args.reduce_scale))

    out = cv2.VideoWriter(args.out,
                          fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          out_size
                          )
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_count = 0
    e_time = None
    s_time = None

    cfg = widerface_640
    # load net
    num_classes = len(WIDERFace_CLASSES) + 1  # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    shrink = 1

    for i in range(4000):
        cap.read()

    while cap.isOpened():
        if args.verbose > 0 and e_time is not None:
            ittime = (e_time - s_time) * (total_frame - f_count)
            hour = int(ittime / 60.0 / 60.0)
            minute = int((ittime / 60.0) - (hour * 60))
            second = int(ittime % 60.0)

            print("Progress %d/%d(%.2f%%), Estimated time : %02d:%02d:%02d" %
                  (f_count, total_frame, (f_count / total_frame) * 100, hour, minute, second))

        s_time = time.time()

        ret, frame = cap.read()

        if frame is None:
            break

        if args.vertical == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if args.reduce_scale == 1:
            resized_img = frame
        else:
            resized_img = cv2.resize(frame, None, None, fx=args.reduce_scale, fy=args.reduce_scale)

        det = detect_faces(resized_img, net)
        blur_faces(resized_img, det, args.threshold)

        out.write(resized_img)

        e_time = time.time()
        f_count += 1

        if f_count > 100:
            break

    print("Finished!")
    cap.release()
    out.release()
    