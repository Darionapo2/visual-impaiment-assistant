import time

import numpy as np
import cv2
from Blob import *

cap = cv2.VideoCapture('../test_images/mall.mp4')
stop = 200

detected_blobs = []
max_dist = 40

binary_res = (1280, 720)
max_area = 0.75 * binary_res[0] * binary_res[1]

i = 0

while cap.isOpened():
    i += 1

    if stop == 1:
        break
    stop += 1

    ret, frame = cap.read()
    if not ret:
        break

    # convert video frame
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale_image, 150, 240, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and take only the first n on them
    n = 35
    sorted_contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    filtered_contours = sorted_contours[:n]

    filtered_contours = [contour for contour in filtered_contours if cv2.contourArea(contour) <
                         max_area]

    # Obtaining the center of each blob.
    new_centers = []
    for contour in filtered_contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            detected_center = np.array((cx, cy))
            new_centers.append(detected_center)


            # Generation of a blob object for each contour, if not existent yet.
            found = False
            for blob in detected_blobs:
                if blob.is_near(detected_center, max_dist):
                    blob.center = np.mean((detected_center, blob.center), dtype = int, axis = 0)
                    blob.area = cv2.contourArea(contour) + blob.area
                    found = True

            if not found:
                detected_blobs.append(Blob(cv2.contourArea(contour)))


    # Clean blobs
    clean_blobs = [blob for blob in detected_blobs for center in new_centers if blob.is_near(
        center, max_dist)]

    detected_blobs = list(set.intersection(set(detected_blobs), set(clean_blobs)))




    # Drawing the frame...
    frame_contours = frame.copy()
    cv2.drawContours(frame_contours, filtered_contours, -1, (0, 255, 0), 2)
    # print('----')

    for center in new_centers:
        cv2.circle(frame_contours, center, 3, (255, 0, 255), 1)

    for blob in clean_blobs:
        cv2.circle(frame_contours, tuple(blob.center), 3, (0, 0, 255), 1)
        cv2.circle(frame_contours, tuple(blob.center), max_dist, (255, 0, 0), 1)

    cv2.imshow('frame', binary)
    cv2.imshow('frame_contours', frame_contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)



cap.release()
cv2.destroyAllWindows()