import utils

class Obstacle:

    max_dist = 50
    text_location = None

    def __init__(self, area, center):
        self.area = area
        self.center = center
        self.label = None
        self.conf = None
        self.distance = None

        self.radius = utils.optimal_radius(area, self.max_dist)

    def is_near(self, center, thresh):
        if utils.centers_distance(self.center, center) < thresh:
            return True
        return False

    def update_radius(self):
        self.radius = utils.optimal_radius(self.area, self.max_dist)