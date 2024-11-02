import multiprocessing as mp
import threading
import os
import pyrealsense2 as rs2
import numpy as np
import cv2

# from tflite_support.task import core, processor, vision
import utils
from Obstacle import Obstacle

import tensorflow as tf

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate

# Load the TFLite object detection model.
model_path = "models/efficientdet_lite0.tflite"
base_options = core.BaseOptions(file_name=model_path, use_coral=False, num_threads=4)
detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.5)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

def cam_thread(frame_queue, depth_queue, running):
    print("cam_thread pid:", os.getpid())
    ctx = rs2.context()
    devs = ctx.query_devices()
    print("query_devices %d" % devs.size())

    pipe = rs2.pipeline()
    config = rs2.config()
    config.enable_stream(rs2.stream.color, 640, 480, rs2.format.bgr8, 15)
    config.enable_stream(rs2.stream.depth, 640, 480, rs2.format.z16, 15)
    pipe.start(config)

    try:
        while running.value:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            frame_number = color_frame.get_frame_number()
            print(f"frame_number: {frame_number}")

            # Put the frame into the queue for processing
            if frame_queue.full():
                frame_queue.get()  # Drop the oldest frame if the queue is full
            frame_queue.put((frame_number, color_image))

            if depth_queue.full():
                depth_queue.get()  # Drop the oldest frame if the queue is full
            depth_queue.put(depth_image)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running.value = False
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

def detection_process(frame_queue, running):
    print("detection_process pid:", os.getpid())

    while running.value:
        if not frame_queue.empty():
            frame_number, color_image = frame_queue.get()

            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(color_image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_rgb_image = cv2.resize(rgb_image, (640, 480))

            # Create a TensorImage object from the RGB image.
            input_tensor = vision.TensorImage.create_from_array(resized_rgb_image)

            # Run object detection using the model.
            detection_result = detector.detect(input_tensor)

            # Draw the detection results on the original image.
            image = utils.visualize(image, detection_result)

            # Display the processed frame
            cv2.imshow('RealSense Object Detection', image)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running.value = False
                break

def obstacle_detection_process(depth_queue, obstacle_queue, running):
    print("obstacle_detection_process pid:", os.getpid())
    previous_obstacles = []

    while running.value:
        if not depth_queue.empty():
            depth_image = depth_queue.get()

            obstacles = detect(depth_image, previous_obstacles)
            previous_obstacles = obstacles

            # Put detected obstacles into the obstacle queue for further processing
            if obstacle_queue.full():
                obstacle_queue.get()  # Drop the oldest obstacle if the queue is full
            obstacle_queue.put(obstacles)

            # Show the obstacles on the depth colormap
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            draw(depth_colormap, obstacles)

            cv2.imshow('Obstacle Detection', depth_colormap)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running.value = False
                break


def detect(frame, previous_obstacles):
    obstacles = []

    mask_obstacle_filter = cv2.inRange(frame, 200, 1000)
    closed_mask = cv2.morphologyEx(mask_obstacle_filter, cv2.MORPH_CLOSE, kernel = np.ones((7, 7)))
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = 1
    sorted_contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    filtered_contours = sorted_contours[:n]

    new_centers = []
    for contour in filtered_contours:

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            detected_center = np.array((cx, cy))
            new_centers.append(detected_center)

            found = False
            for o in previous_obstacles:
                if o.is_near(detected_center, 50):
                    o.center = detected_center
                    found = True
                    break

                elif o.is_near(detected_center, o.radius):
                    o.area = o.area + cv2.contourArea(contour)
                    o.update_radius()
                    found = True
                    break

            if not found:
                obstacles.append(Obstacle(cv2.contourArea(contour), detected_center))

    # Clean obstacles
    clean_obstacles = [o for o in obstacles for center in new_centers if o.is_near(center, 50)]
    new_obstacles = list(set.intersection(set(obstacles), set(clean_obstacles)))

    for o in new_obstacles:
        o.text_location = utils.locate_obstacles(1280, 720, np.array([0.34, 0.33]), np.array([0.5]), o)

    return new_obstacles

def draw(frame, obstacles):
    for o in obstacles:
        cv2.circle(frame, o.center, 5, (255, 0, 255), 5)

        cv2.circle(frame, o.center, int(o.radius), (255, 0, 255), 1)

        cv2.putText(frame, text = o.text_location, org = o.center,
                    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.2, color = (255, 255, 0),
                    thickness = 1, lineType = cv2.LINE_AA)

def join_non_null_values(frames):
    joined_frame = np.zeros_like(frames[0], dtype = frames[0].dtype)

    for frame in frames:
        mask = frame != 0
        joined_frame = np.where(mask, frame, joined_frame)

    return joined_frame




if __name__ == '__main__':
    mp.set_start_method('spawn')

    frame_queue = mp.Queue(maxsize=5)
    depth_queue = mp.Queue(maxsize=5)
    obstacle_queue = mp.Queue(maxsize=5)
    running = mp.Value('b', True)

    cam_thread_handle = threading.Thread(target=cam_thread, args=(frame_queue, depth_queue, running))
    detect_proc = mp.Process(target=detection_process, args=(frame_queue, running))
    obstacle_proc = mp.Process(target=obstacle_detection_process, args=(depth_queue, obstacle_queue, running))

    cam_thread_handle.start()
    detect_proc.start()
    obstacle_proc.start()

    cam_thread_handle.join()
    detect_proc.join()
    obstacle_proc.join()
