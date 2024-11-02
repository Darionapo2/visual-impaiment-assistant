import math
import numpy as np

def centers_distance(center1, center2):
    return np.linalg.norm(center1 - center2)

def optimal_radius(area, max_dist):
    return (math.sqrt(2)/2) * math.sqrt(area) + max_dist

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
        ['up-left', 'up-center', 'up-right'],
        ['down-left', 'down-center', 'down-right']
    ]

    return natural_language_translation_matrix[rel_coord_y][rel_coord_x]