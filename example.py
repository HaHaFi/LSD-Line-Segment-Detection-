import _lsd
import cv2
import numpy as np


if __name__ == "__main__":

    input_img_name = "test.png"
    output_img_name = "lsd_output.jpg"
    # construct LSD class with default hyper parameters
    lsd = _lsd.lsd()

    # read the image you want to do LSD
    lsd.read_img(input_img_name)

    # process LSD and return Line segment list [ [x1,y1,x2,y2 ]...  ]
    line_segment = lsd.line_segment_detector()

    # draw line on image
    seg_img = np.zeros((lsd.height(), lsd.width()), np.uint8)
    for i in range(0, len(line_segment), 4):
        cv2.line(seg_img, (int(line_segment[i+0]), int(line_segment[i+1])),
                 (int(line_segment[i+2]), int(line_segment[i+3])), (255), 1)

    # output jpg
    cv2.imwrite(f"lsd_output.jpg", seg_img)
