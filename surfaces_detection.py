import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time


def global_thresholding_opencv(threshold = 127):
    img = cv.imread('test_images/rocks.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    start = time.time()
    ret, thresh1 = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    end = time.time()

    plt.imshow(thresh1, 'gray', vmin = 0, vmax = 255)
    plt.show()

    return start, end

def global_thresholding_numpy(t = [127]):
    img = cv.imread('test_images/rocks.jpg', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    start = time.time()
    out = np.digitize(img, bins = t)
    out2 = out * int(255 / len(t))
    end = time.time()

    plt.imshow(out2, 'gray', vmin = 0, vmax = 255)
    plt.show()

    return start, end

if __name__ == '__main__':
    s, e = global_thresholding_opencv()
    print(e - s)
    s, e = global_thresholding_numpy()
    print(e - s)