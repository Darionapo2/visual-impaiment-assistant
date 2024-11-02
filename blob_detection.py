import numpy as np
import cv2

cap = cv2.VideoCapture('test_images/sample_dancer.mp4')

stop = 200
while cap.isOpened():

    if stop == 1:
        break

    stop += 1

    ret, frame = cap.read()

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale_image, 100, 200, cv2.THRESH_BINARY)

    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = False
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector.create(params)

    # Detect blobs.
    keypoints = detector.detect(binary_rgb)

    frame_contours = frame.copy()

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame_contours, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', binary)
    cv2.imshow('frame_contours', im_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()