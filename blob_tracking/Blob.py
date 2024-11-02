import numpy as np

def centers_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

class Blob:

    previuos_locations = []

    def __init__(self, center):
        # self.area = area
        # self.contour = contour
        self.center = center

    def is_near(self, location, thresh):
        if centers_distance(self.center, location) < thresh:
            return True
        return False

    def __str__(self):
        return f'{self.center}'