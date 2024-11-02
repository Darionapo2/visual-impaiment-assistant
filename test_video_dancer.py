import time

import numpy as np
import cv2

cap = cv2.VideoCapture('test_images/sample_dancer.mp4')

stop = 200
while cap.isOpened():

    if stop == 1:
        break

    stop += 1

    ret, frame = cap.read()

    if not ret:
        break

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale_image, 225, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Ms = []

    a = time.time()
    if len(contours) >= 1:
        contours = [max(contours[1:], key = cv2.contourArea)]

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                Ms.append((cx, cy))

                print('M: ', (cx, cy))
    b = time.time()

    print(b-a)

    frame_contours = frame.copy()
    cv2.drawContours(frame_contours, contours, -1, (0, 255, 0), 2)
    for m in Ms:
        cv2.circle(frame_contours, m, 5, (0, 0, 255), 5)

    cv2.imshow('frame', binary)
    cv2.imshow('frame_contours', frame_contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()