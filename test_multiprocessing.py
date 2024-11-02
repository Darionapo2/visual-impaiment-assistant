import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt
import multiprocessing as mp

np.set_printoptions(threshold=sys.maxsize)


def locate_obstacles(frame_width, frame_height, width_rel_subs, height_rel_subs, obstacle):
    x_subs = width_rel_subs * frame_width
    y_subs = height_rel_subs * frame_height

    boundaries_x = [int(sum(x_subs[:i+1])) for i in range(0, len(x_subs))]
    boundaries_x.append(frame_width)

    boundaries_y = [int(sum(y_subs[:i+1])) for i in range(0, len(y_subs))]
    boundaries_y.append(frame_height)

    px, py = obstacle.center

    rel_coord_x = np.searchsorted(boundaries_x, px, side='right')
    rel_coord_y = np.searchsorted(boundaries_y, py, side='right')

    natural_language_translation_matrix = [
        ['up-left', 'up-center', 'up-right'],
        ['down-left', 'down-center', 'down-right']
    ]

    return natural_language_translation_matrix[rel_coord_y][rel_coord_x]


def centers_distance(center1, center2):
    return np.linalg.norm(center1 - center2)


def optimal_radius(area, max_dist):
    return (sqrt(2) / 2) * sqrt(area) + max_dist


class Obstacle:
    max_dist = 50

    def __init__(self, area, center):
        self.area = area
        self.center = center
        self.radius = optimal_radius(area, self.max_dist)

    def is_near(self, center, thresh):
        return centers_distance(self.center, center) < thresh

    def update_radius(self):
        self.radius = optimal_radius(self.area, self.max_dist)


def process_frame(depth_image, color_image, prev_frame, obstacles, new_loc):
    depth_mask1 = depth_image != 0
    joined_frame = np.where(depth_mask1, depth_image, prev_frame)

    mask_obstacle_filter = cv2.inRange(joined_frame, 200, 1000)
    mask_obstacle_filter_rgb = cv2.cvtColor(mask_obstacle_filter, cv2.COLOR_GRAY2RGB)
    closed_mask = cv2.morphologyEx(mask_obstacle_filter, cv2.MORPH_CLOSE, kernel=np.ones((7, 7)))

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = 1
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
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
            for o in obstacles:
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

    clean_obstacles = [o for o in obstacles for center in new_centers if o.is_near(center, 50)]
    obstacles = list(set.intersection(set(obstacles), set(clean_obstacles)))

    color_image_contours = color_image.copy()
    for o in obstacles:
        cv2.circle(color_image_contours, o.center, 5, (255, 0, 255), 5)
        cv2.circle(color_image_contours, o.center, int(o.radius), (255, 0, 255), 1)
        cv2.putText(color_image_contours, text=new_loc, org=o.center,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.2, color=(255, 255, 0),
                    thickness=1, lineType=cv2.LINE_AA)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(joined_frame, alpha=0.03), cv2.COLORMAP_JET)

    return color_image, depth_colormap, mask_obstacle_filter_rgb, color_image_contours, obstacles


def open_cv_viewer():
    obstacles = []

    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    print(device)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    pipeline.start(config)

    stop = 200000

    old_loc = None
    new_loc = 'undefined'
    prev_frame = None

    pool = mp.Pool(processes=mp.cpu_count())

    for i in range(stop):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if i % 2 == 0:
            if prev_frame is not None:
                result = pool.apply_async(process_frame, (depth_image, color_image, prev_frame, obstacles, 'test'))
                color_image, depth_colormap, mask_obstacle_filter_rgb, color_image_contours, obstacles = result.get()

                if i % 15 == 0:
                    if len(obstacles) > 0:
                        new_loc = locate_obstacles(1280, 720, np.array([0.34, 0.33]), np.array([0.5]), obstacles[0])
                        if new_loc != old_loc:
                            old_loc = new_loc
                            print(new_loc)

                cv2.imshow('RealSense color colormap', color_image)
                cv2.imshow('RealSense depth colormap', depth_colormap)
                cv2.imshow('Obstacle filter', mask_obstacle_filter_rgb)
                cv2.imshow('Obstacle detection', color_image_contours)

                cv2.waitKey(1)

            prev_frame = depth_image

    pool.close()
    pool.join()
    pipeline.stop()


if __name__ == '__main__':
    open_cv_viewer()
