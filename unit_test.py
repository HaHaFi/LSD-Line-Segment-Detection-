import _lsd
import cv2
import numpy as np
import copy
import pytest
import math
from timeit import Timer


class TestClass:
    def test_read_img(self):
        a = _lsd.lsd()
        a.read_img("test_1.png")
        assert a.width() == 1365
        assert a.height() == 2048

        b = _lsd.lsd()
        b.read_img("test_2.png")
        assert b.width() == 1024
        assert b.height() == 1024

        c = _lsd.lsd()
        c.read_img("test_3.png")
        assert c.width() == 768
        assert c.height() == 432

    def test_getter_setter(self):
        a = _lsd.lsd()
        a.read_img("test_1.png")
        for i in range(10):
            a[i, i] = i
            assert a[i, i] == i

    def test_dot(self):
        assert _lsd.dot(1, 2, 2, 1) == 1*2+2*1
        assert _lsd.dot(0, 2, 2, 0) == 0
        assert _lsd.dot(16, 3, 22, 16) == 16*22+3*16

    def test_cross(self):
        assert _lsd.cross(1, 2, 2, 1) == 1*1-2*2
        assert _lsd.cross(0, 2, 2, 0) == 0*0-2*2
        assert _lsd.cross(16, 3, 22, 16) == 16*16-3*22

    def test_length(self):
        assert _lsd.length(1, 0) == 1
        assert _lsd.length(0, 0) == 0
        assert _lsd.length(23, 0) == 23
        assert _lsd.length(0, 2) == 2
        assert _lsd.length(3, 4) == 5

    def test_angle_diff(self):
        assert _lsd.angle_diff(math.pi, 0) == math.pi
        assert _lsd.angle_diff(math.pi, math.pi/2) == math.pi/2
        assert _lsd.angle_diff(math.pi, math.pi/4) == math.pi/4*3
        assert _lsd.angle_diff(math.pi/2, math.pi) == math.pi/2

    def test_is_similiar_direction(self):
        assert _lsd.is_similiar_direction(
            math.pi/2, math.pi, math.pi*3/4) == True
        assert _lsd.is_similiar_direction(
            math.pi/2, math.pi, math.pi/4) == False
        assert _lsd.is_similiar_direction(
            math.pi, math.pi/2, math.pi*3/4) == True
        assert _lsd.is_similiar_direction(
            math.pi, math.pi/2, math.pi/4) == False

    def test_gaussian_kernel(self):

        kernel = _lsd.gaussian_kernel(5, 1)
        assert sum(kernel) == 1
        kernel = _lsd.gaussian_kernel(12, 1)
        assert sum(kernel) == 1
        kernel = _lsd.gaussian_kernel(5, 30)
        assert sum(kernel) == 1
        kernel = _lsd.gaussian_kernel(6, 20)
        assert sum(kernel) == 1


def save_image_into_txt(inputfilename, outputfilename):
    image = cv2.imread(inputfilename, cv2.IMREAD_GRAYSCALE)
    with open(outputfilename, "w", encoding="UTF-8")as f:
        f.write(str(image.shape[1])+" "+str(image.shape[0])+" ")
        image = list(image.reshape(-1))
        f.write(" ".join(str(x) for x in image))


def save_txt_into_image(inputfilename, outputfilename):
    with open(inputfilename, "r", encoding="UTF-8")as f:
        data = f.read().strip().split(" ")
        image = np.array(data[2:]).reshape(
            int(data[1]), int(data[0])).astype(np.uint8)
        # image=cv2.fromarray
        cv2.imwrite(outputfilename, image)


setup_lsd = """
import _lsd
import cv2
import numpy as np 
a=_lsd.lsd()
a.read_img("test_%d.png")

"""
lsd_exec = """
line_segment=a.line_segment_detector()

"""


setup_glsd = """
import _lsd
import cv2
import numpy as np 
image = cv2.imread("test_%d.png",cv2.COLOR_BGR2GRAY)
lsd = cv2.createLineSegmentDetector(0)

"""
glsd_exec = """
line_segment = lsd.detect(image)
"""

setup_canny = """
import _lsd
import cv2
import numpy as np 
image = cv2.imread("test_%d.png",cv2.COLOR_BGR2GRAY)

"""
canny_exec = """
blurred = cv2.GaussianBlur(image, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)
"""

setup_hough = """
import _lsd
import cv2
import numpy as np 
image = cv2.imread("test_%d.png",cv2.COLOR_BGR2GRAY)
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
"""
hough_exec = """
blurred = cv2.GaussianBlur(image, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)
lines = cv2.HoughLinesP(canny,rho,theta,threshold,np.array([]),
                            min_line_length,max_line_gap)
"""


if __name__ == "__main__":
    retcode = pytest.main()

    # for testing performance
    # repeat = 5
    # test_image_list = (1, 6, 1)
    # # test time consume
    # for i in range(*test_image_list):
    #     image = cv2.imread("test_%d.png" % i, cv2.COLOR_BGR2GRAY)
    #     print("-"*25)
    #     print(
    #         f"test image test_{i}.png\nimage size = w:{image.shape[1]} x h:{image.shape[0]}")
    #     # test lsd
    #     lsd_time = Timer(lsd_exec, setup=setup_lsd % i)
    #     sec = lsd_time.repeat(repeat, 1)
    #     lsd_min = min(sec)
    #     lsd_max = max(sec)

    #     print(f"\tlsd times {lsd_min} sec")

    #     # Need to install opencv at version 3.x.x.x
    #     # ## test groundtrue_lsd
    #     # glsd_time = Timer(glsd_exec, setup=setup_glsd%i)
    #     # sec =glsd_time.repeat(repeat,1)
    #     # glsd_min = min(sec)
    #     # glsd_max = max(sec)

    #     # print(f"\tglsd by opencv times {lsd_min} sec")

    #     # test canny
    #     canny_time = Timer(canny_exec, setup=setup_canny % i)
    #     sec = canny_time.repeat(repeat, 1)
    #     canny_min = min(sec)
    #     canny_max = max(sec)

    #     print(f"\tcanny by opencv times {canny_min} sec")

    #     hough_time = Timer(hough_exec, setup=setup_hough % i)
    #     sec = hough_time.repeat(repeat, 1)
    #     hough_min = min(sec)
    #     hough_max = max(sec)

    #     print(f"\though by opencv times {hough_min} sec")

    # for j in range(*test_image_list):

    #     a = _lsd.lsd()
    #     a.read_img(f"test_{j}.png")
    #     line_segment = a.line_segment_detector()

    #     seg_img = np.zeros((a.height(), a.width()), np.uint8)+255

    #     for i in range(0, len(line_segment), 4):
    #         cv2.line(seg_img, (int(line_segment[i+0]), int(line_segment[i+1])), (int(
    #             line_segment[i+2]), int(line_segment[i+3])), (0), 1)

    #     cv2.imwrite(f"lsd_{j}.jpg", seg_img)

    # for i in range(*test_image_list):
    #     image = cv2.imread("test_%d.png" % i, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(image, (5, 5), 0)
    #     canny = cv2.Canny(blurred, 30, 150)
    #     cv2.imwrite(f"canny_{i}.jpg", canny)
    # for i in range(*test_image_list):

    #     image = cv2.imread("test_%d.png" % i, cv2.COLOR_BGR2GRAY)
    #     rho = 1
    #     theta = np.pi/180
    #     threshold = 1
    #     min_line_length = 10
    #     max_line_gap = 1
    #     line_image = np.copy(image)*0
    #     blurred = cv2.GaussianBlur(image, (5, 5), 0)
    #     canny = cv2.Canny(blurred, 30, 150)
    #     lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]),
    #                             min_line_length, max_line_gap)

    #     for _list in lines:
    #         x1, y1, x2, y2 = _list[0]
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #     color_edges = np.dstack((canny, canny, canny))
    #     combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    #     cv2.imwrite(f"hough_{i}.jpg", combo)
