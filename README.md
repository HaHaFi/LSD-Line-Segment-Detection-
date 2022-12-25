#  LSD (Line Segment Detector) 
## Introdution
In computer vision region, detection of the line on the images in real time is an useful function, because it has a lot of downstream task, such as vanish point method base camera calibration and road marking segmentation.






## Prospective Users
The people who want to get line information on the image and need a fast algorithm.

## Result
![](https://i.imgur.com/b2qgiXa.png)

## Preformance measuring


| Image Size | LSD speed | 
| -------- | -------- |
| 1365*2045     | 0.60058 sec |
| 1024*1024     | 0.33683 sec |
| 768*432     | 0.00768 sec |
| 500*500     | 0.01254 sec |
| 400*400     | 0.01286 sec |



## Python package install
```
pip3 install -r ./requirement.txt
```
## Engineering Infrastructure
Automatic build system: make

Testing framework: [pytest](https://pytest.org)

C++ Wrapping tool: pybind11

complier: g++

## build commamd
```
# for build _lsd package
make

# for testing _lsd
make test
```

## Usage Example

```
import _lsd
import cv2
import numpy as np

def main():
    input_img_name="test_1.png"
    
    # construct LSD class with default hyper parameters
    lsd=_lsd.lsd()
    
    # read the image you want to do LSD
    lsd.read_img(input_img_name)
    
    # process LSD and return Line segment list [ [x1,y1,x2,y2 ]...  ]
    line_segment=lsd.line_segment_detector()
    
    # draw line on image
    seg_img = np.zeros((lsd.height(), lsd.width()), np.uint8)
    for i in range(0, len(line_segment), 4):
        cv2.line(seg_img, (int(line_segment[i+0]), int(line_segment[i+1])),
                 (int(line_segment[i+2]), int(line_segment[i+3])), (255), 1)
    # output jpg
    cv2.imwrite(f"lsd_output.jpg", seg_img)

```


## References
1. https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf
2. https://ieeexplore.ieee.org/document/4731268
