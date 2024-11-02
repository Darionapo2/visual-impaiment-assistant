import time
import cv2
import numpy as np
import sys

cv2.namedWindow('computer webcam binary')

def main():
    im = cv2.imread('test_images/opencv_frame_0.png')
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 10)
    cv2.imshow('computer webcam binary', im)




def opencv_example():
    im = cv2.imread('test_images/opencv_frame_0.png')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    a = time.time()
    contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    b = time.time()

    print(b-a)

    print(contours)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
    cv2.imshow('computer webcam binary', im)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    opencv_example()