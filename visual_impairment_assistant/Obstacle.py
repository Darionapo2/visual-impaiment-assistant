import utils
import numpy as np
import cv2

class Obstacle:

    max_dist = 50
    text_location = None

    def __init__(self, area, center, depth_frame, contour, use_min_distance = False):
        self.area = area
        self.center = center
        self.label = None
        self.conf = None
        self.use_min_distance = use_min_distance
        self.distance = self.calculate_distance(depth_frame, contour)  # Calcolo iniziale della distanza
        self.radius = utils.optimal_radius(area, self.max_dist)
        self.contour = contour

    def calculate_distance(self, depth_frame, contour):
        mask = np.zeros_like(depth_frame, dtype = np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        distances = depth_frame[mask == 255]
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            if self.use_min_distance:
                return np.min(valid_distances)
            else:
                return np.mean(valid_distances)
        else:
            return None

    def is_near(self, center, thresh):
        if utils.centers_distance(self.center, center) < thresh:
            return True
        return False

    def update_radius(self):
        self.radius = utils.optimal_radius(self.area, self.max_dist)
