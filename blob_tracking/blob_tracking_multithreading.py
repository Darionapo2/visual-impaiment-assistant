import time
import numpy as np
import cv2
from Blob import *

from multiprocessing import Process, Manager
import multiprocessing


def main():
    cap = cv2.VideoCapture('../test_images/mall.mp4')
    stop = 200

    detected_blobs = []
    max_dist = 40

    binary_res = (1280, 720)
    max_area = 0.75 * binary_res[0] * binary_res[1]

    i = 0
    n = 25

    while cap.isOpened():
        i += 1

        if stop == 1:
            break
        stop += 1

        ret, frame = cap.read()
        if not ret:
            break

        contours = get_contours(frame)
        filtered_contours = sort_and_filter(contours, max_area, n)
        new_centers = get_new_centers(filtered_contours)

        detected_blobs = match_and_create_blobs(new_centers, detected_blobs, max_dist)
        cleaned = clean_blobs(detected_blobs, new_centers, max_dist)
        draw_frames(cleaned, frame, filtered_contours, max_dist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.waitKey(25)

    cap.release()
    cv2.destroyAllWindows()


def draw_frames(cleaned, frame, filtered_contours, max_dist):
    # Drawing the frame...
    frame_contours = frame.copy()
    cv2.drawContours(frame_contours, filtered_contours, -1, (0, 255, 0), 2)

    for blob in cleaned:
        cv2.circle(frame_contours, tuple(blob.center), 3, (0, 0, 255), 1)
        cv2.circle(frame_contours, tuple(blob.center), max_dist, (255, 0, 0), 1)

    cv2.imshow('frame_contours', frame_contours)

def clean_blobs(detected_blobs, new_centers, max_dist):
    cleaned = [blob for blob in detected_blobs for center in new_centers if blob.is_near(
        center, max_dist)]

    return list(set.intersection(set(detected_blobs), set(cleaned)))

def match_and_create_blobs(new_centers, detected_blobs, max_dist):
    # Match existing centers with existing blobs
    for new_center in new_centers:

        found = False

        for blob in detected_blobs:
            if blob.is_near(new_center, max_dist):
                blob.center = np.mean((new_center, blob.center), dtype = int, axis = 0)
                found = True

        if not found:
            detected_blobs.append(Blob(new_center))

    return detected_blobs


def get_new_centers(filtered_contours):
    # Obtaining the center of each blob.
    new_centers = []
    for contour in filtered_contours:
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            detected_center = np.array((cx, cy))
            new_centers.append(detected_center)

    return new_centers

def sort_and_filter(contours, max_area, n):
    # Sort contours by area and take only the first n on them
    sorted_contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    filtered_contours = sorted_contours[:n]

    filtered_contours = [contour for contour in filtered_contours if cv2.contourArea(contour) <
                         max_area]
    return filtered_contours


def get_contours(frame):
    # convert video frame
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale_image, 150, 240, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def videoplayer():
    pass


if __name__ == '__main__':
    main()