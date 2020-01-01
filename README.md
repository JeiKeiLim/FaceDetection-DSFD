# Anonymizing videos by [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220)
Modified by [Jongkuk Lim](http://limjk.com?refer=github_DSFD)

## History
This repository was forked from [FaceDetection-DSFD](https://github.com/TencentYoutuResearch/FaceDetection-DSFD) which is implentation of [DSFD: Dual Shot Face Detector](https://arxiv.org/abs/1810.10220) by [Jian Li](https://lijiannuist.github.io/), [Yabiao Wang](https://github.com/ChaunceyWang), [Changan Wang](https://github.com/HiKapok), [Ying Tai](https://tyshiwo.github.io/), [Jianjun Qian](http://www.escience.cn/people/JianjunQian/index.html), [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN&oi=sra), Chengjie Wang, Jilin Li, Feiyue Huang.


## Introduction
Simple implementation of video anonymization.

If you are looking for a faster version, check [Anonymizing videos by lightDSFD](https://github.com/JeiKeiLim/Anonymizing_video_by_lightDSFD). 
And, if you are looking for a simpler example, [noone video](https://github.com/JeiKeiLim/noone_video) is implemented by only OpenCV examples.

Note that this repository is not designed for training models. If you are looking for training models, please visit original repository [FaceDetection-DSFD](https://github.com/TencentYoutuResearch/FaceDetection-DSFD).

## Comparisons

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_01.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_02.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_03.gif" />

<img src="https://github.com/JeiKeiLim/mygifcontainer/raw/master/deep_face_detector/compare_04.gif" />


## Requirements
CUDA supported enviornment

(Tested on NVIDIA GTX 1060(6GB) and GTX 1080 Ti(8GB))

- Torch >= 0.3.1
- Torchvision >= 0.2.1
- (Tested on torch 1.3.1 and Torchvision 0.4.2)
- Python 3.6

## Getting Started

1. Download DSFD model from original repository provided [[微云]](https://share.weiyun.com/567x0xQ) [[google drive]](https://drive.google.com/file/d/1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-/view?usp=sharing) 
2. Place `WIDERFace_DSFD_RES152.pth` to `./weights/`.
  
3. Run `./demo.py` to check if it is running.
```
python demo.py [--trained_model [TRAINED_MODEL]] [--img_root  [IMG_ROOT]] 
               [--save_folder [SAVE_FOLDER]] [--visual_threshold [VISUAL_THRESHOLD]] 
    --trained_model      Path to the saved model
    --img_root           Path of test images
    --save_folder        Path of output detection resutls
    --visual_threshold   Confidence thresh
```


## Usage
```
usage: blur_video.py [-h] -i INPUT -o OUTPUT [--vertical VERTICAL]
                     [--verbose VERBOSE] [--reduce_scale REDUCE_SCALE]
                     [--trained_model TRAINED_MODEL] [--threshold THRESHOLD]
                     [--cuda CUDA]
```

## Required arguments
```
Required file paths:
  -i INPUT, --input INPUT
                        Video file path
  -o OUTPUT, --output OUTPUT
                        Output video path
```

## Optional arguments
```
optional arguments:
  -h, --help            show this help message and exit
  --vertical VERTICAL   0 : horizontal video(default), 1 : vertical video
  --verbose VERBOSE     Show current progress and remaining time
  --reduce_scale REDUCE_SCALE
                        Reduce scale ratio. ex) 2 = half size of the input.
                        Default : 2
  --trained_model TRAINED_MODEL
                        Trained state_dict file path to open
  --threshold THRESHOLD
                        Final confidence threshold
  --cuda CUDA           Use cuda
```


### Citation
If you find DSFD useful in your research, please consider citing: 
```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
