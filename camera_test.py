import time
import cv2
import numpy as np
import sys

np.set_printoptions(threshold = sys.maxsize)


cam = cv2.VideoCapture(0)

cv2.namedWindow('computer webcam')
cv2.namedWindow('computer webcam theshold1')
cv2.namedWindow('computer webcam grayscale')
cv2.namedWindow('computer webcam binary')

h = 480
w = 640

def main():

    img_counter = 0

    for i in range(100):

        ret, frame = cam.read()

        if not ret:
            print('failed to grab frame')
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret2, thresh1 = cv2.threshold(grayscale_frame, 127, 255, cv2.THRESH_BINARY)

        t = [64, 127, 191]
        lvls_3_out = np.digitize(grayscale_frame, bins = t)
        lvls_3_out = lvls_3_out.astype(np.uint8)
        lvls_3_out2 = lvls_3_out * int(255 / len(t))

        binary_out = cv2.inRange(lvls_3_out2, 0, 0)

        cv2.imshow('computer webcam', frame)
        cv2.imshow('computer webcam theshold1', lvls_3_out2)
        cv2.imshow('computer webcam grayscale', grayscale_frame) # (480, 640)

        contours, _ = cv2.findContours(binary_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary_out, contours, -1, (0, 255, 0), 1)

        cv2.imshow('computer webcam binary', binary_out)

        '''
        black_pixels_coords = np.argwhere(lvls_3_out2 == 0)
        top_right_pixels = list(filter(filter_top_right, black_pixels_coords))
        top_right_pixels_count = len(top_right_pixels)

        print(top_right_pixels_count)
        '''

        k = cv2.waitKey(1)


        if k % 256 == 27:
            # ESC pressed
            print('Escape hit, closing...')
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = 'opencv_frame_{}.png'.format(img_counter)
            cv2.imwrite(img_name, binary_out)
            print('{} written!'.format(binary_out))
            img_counter += 1


    cam.release()
    cv2.destroyAllWindows()




def filter_top_right(coords):
    px, py = coords
    if px >= w/2 and py <= h/2:
        return True
    return False

if __name__ == '__main__':
    main()