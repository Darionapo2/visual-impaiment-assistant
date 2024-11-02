import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import natual_language_compiler

from math import *

np.set_printoptions(threshold = sys.maxsize)


executor = ProcessPoolExecutor(max_workers = 2)


def locate_obstacles(frame_width, frame_height, width_rel_subs, height_rel_subs, obstacle):
    x_subs = width_rel_subs * frame_width
    y_subs = height_rel_subs * frame_height

    boundaries_x = [int(sum(x_subs[:i+1])) for i in range(0, len(x_subs))]
    boundaries_x.append(frame_width)

    boundaries_y = [int(sum(y_subs[:i+1])) for i in range(0, len(y_subs))]
    boundaries_y.append(frame_height)

    px, py = obstacle.center

    rel_coord_x = np.searchsorted(boundaries_x, px, side = 'right')
    rel_coord_y = np.searchsorted(boundaries_y, py, side = 'right')

    natural_language_translation_matrix = [
        ['upper left', 'upper center', 'upper right'],
        ['bottom left', 'bottom center', 'bottom right']
    ]

    return natural_language_translation_matrix[rel_coord_y][rel_coord_x]


def test():
    frame_width = 1280
    frame_height = 720
    width_rel_subs = np.array([0.34, 0.33])
    height_rel_subs = np.array([0.5])

    o = Obstacle(5000, (100, 650))
    locate_obstacles(frame_width, frame_height, width_rel_subs, height_rel_subs, o)


def centers_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

def optimal_radius(area, max_dist):
    return (sqrt(2)/2) * sqrt(area) + max_dist

class Obstacle:

    max_dist = 50

    def __init__(self, area, center):
        self.area = area
        self.center = center

        self.radius = optimal_radius(area, self.max_dist)

    def is_near(self, center, thresh):
        if centers_distance(self.center, center) < thresh:
            return True
        return False

    def update_radius(self):
        self.radius = optimal_radius(self.area, self.max_dist)


def open_cv_viewer():

    obstacles = []


    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    print(device)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    # Start streaming
    pipeline.start(config)

    stop = 200000

    # cv2.namedWindow('RealSense depth colormap', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('RealSense color colormap', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Obstacle filter', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Depth histogram', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('all windows', cv2.WINDOW_NORMAL)

    old_loc = None
    new_loc = 'undefined'

    for i in range(stop):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print(depth_frame.get_data()

        '''
        t = [64, 127, 191]
        lvls_3_out = np.digitize(depth_image, bins = t)
        lvls_3_out = lvls_3_out.astype(np.uint8)
        lvls_3_out2 = lvls_3_out * int(255 / len(t))

        contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('computer webcam theshold1', lvls_3_out2)
        '''

        mask_obstacle_filter = cv2.inRange(depth_image, 400, 1000)

        mask_obstacle_filter_rgb = cv2.cvtColor(mask_obstacle_filter, cv2.COLOR_GRAY2RGB)
        closed_mask = cv2.morphologyEx(mask_obstacle_filter, cv2.MORPH_CLOSE, kernel = np.ones((7, 7)))

        contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area and take only the first n on them
        n = 2
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
                for o in obstacles:
                    if o.is_near(detected_center, 50):
                        o.center = detected_center
                        found = True
                        break

                    elif o.is_near(detected_center, o.radius):
                        # o.center = np.mean((detected_center, o.center), dtype = int, axis = 0)
                        o.area = o.area + cv2.contourArea(contour)
                        o.update_radius()
                        found = True
                        break

                if not found:
                    obstacles.append(Obstacle(cv2.contourArea(contour), detected_center))


        # Clean obstacles
        clean_obstacles = [o for o in obstacles for center in new_centers if o.is_near(center, 50)]
        obstacles = list(set.intersection(set(obstacles), set(clean_obstacles)))

        if i % 30 == 0:
            try:
                new_loc = locate_obstacles(1280, 720, np.array([0.34, 0.33]), np.array([0.5]), obstacles[0])
                o_center = obstacles[0].center
                '''
                o_coordinates = (o_center[0], o_center[1])
                distance = depth_image[o_coordinates[0], o_coordinates[1]]
                print(distance)
                '''
            except IndexError:
                print('error')
            if new_loc != old_loc:
                old_loc = new_loc

                executor.submit(natual_language_compiler.read, 'ostacolo a ' + new_loc)

                print(new_loc)



        color_image_contours = color_image.copy()
        cv2.drawContours(color_image_contours, filtered_contours, -1, (0, 255, 0), 3)

        # mask_obstacle_filter_rgb = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2RGB)


        for o in obstacles:
            cv2.circle(color_image_contours, o.center, 5, (255, 0, 255), 5)
            cv2.circle(color_image_contours, o.center, int(o.radius), (255, 0, 255), 1)
            cv2.putText(color_image_contours, text = new_loc, org = o.center,
                        fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale = 1.2, color = (255, 255, 0),
                        thickness = 1, lineType = cv2.LINE_AA)


        cv2.imshow('Depth histogram', color_image)

        if i % 60 == 0:
            counts, bins = np.histogram(depth_image, range = (10, 1000), bins = 100)
            plt.stairs(counts, bins)
            plt.show()


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03),
                                           cv2.COLORMAP_JET)



        # cv2.imshow('RealSense color colormap', color_image)
        # cv2.imshow('RealSense depth colormap', depth_colormap)
        # cv2.imshow('Obstacle filter', mask_obstacle_filter_rgb)


        cv2.imshow('Obstacle detection', cv2.resize(color_image_contours, (1280, 720), interpolation = cv2.INTER_LINEAR))
        cv2.imshow('RealSense depth colormap', cv2.resize(depth_colormap, (1280, 720), interpolation = cv2.INTER_LINEAR))
        cv2.imshow('Obstacle filter', cv2.resize(closed_mask, (1280, 720), interpolation = cv2.INTER_LINEAR))
        cv2.imshow('RealSense color colormap', cv2.resize(color_image, (1280, 720), interpolation = cv2.INTER_LINEAR))

        cv2.waitKey(1)

    pipeline.stop()

def execution():
    res = []
    main_task = executor.submit(open_cv_viewer)
    return res



if __name__ == '__main__':
    # test()
    open_cv_viewer()
    # execution()