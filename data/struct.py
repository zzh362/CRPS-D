"""Defines data structure."""
import math
from collections import namedtuple
from enum import Enum
import config


MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction0', 'direction1', 'shape', 'type'])
Slot = namedtuple('Slot', ['x1', 'y1', 'x2', 'y2', 'angle'])


class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff


def detemine_point_shape_vertical(point, vector):
    """Determine which category the point is in."""
    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction0) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction0) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction0) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction0) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction0) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.l_up
    return PointShape.none


def detemine_point_shape_slant(point, vector):
    """Determine which category the point is in."""
    vec_direct = math.atan2(vector[1], vector[0])
    if point.shape < 0.5:
        # if direction_diff(vec_direct, point.direction1) < config.BRIDGE_ANGLE_DIFF \
        #         or abs(direction_diff(vec_direct, point.direction1) - math.pi) < config.BRIDGE_ANGLE_DIFF:
        if -math.pi < point.direction0 - vec_direct < 0 or point.direction0 - vec_direct > math.pi:
            return PointShape.t_up
        else:
            return PointShape.t_down
        return PointShape.none
    else:
        if direction_diff(vec_direct, point.direction0) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct, point.direction1) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.l_up
        return PointShape.none


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx ** 2 + disty ** 2


def calc_point_direction0_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a.direction0, point_b.direction0)

def calc_point_direction1_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a.direction1, point_b.direction1)

def match_marking_points(point_a, point_b):
    """Determine whether a detected point match ground truth."""
    dist_square = calc_point_squre_dist(point_a, point_b)
    angle0 = calc_point_direction0_angle(point_a, point_b)
    angle1 = calc_point_direction1_angle(point_a, point_b)
    if point_a.shape > 0.5 and point_b.shape < 0.5:
        return False
    if point_a.shape < 0.5 and point_b.shape > 0.5:
        return False
    if point_a.type > 0.5 and point_b.type < 0.5:
        return False
    if point_a.type < 0.5 and point_b.type > 0.5:
        return False
    return (dist_square < config.SQUARED_DISTANCE_THRESH
            and angle0 < config.DIRECTION_ANGLE_THRESH
            and angle1 < config.DIRECTION_ANGLE_THRESH)


def match_slots(slot_a, slot_b):
    """Determine whether a detected slot match ground truth."""
    dist_x1 = slot_b.x1 - slot_a.x1
    dist_y1 = slot_b.y1 - slot_a.y1
    squared_dist1 = dist_x1**2 + dist_y1**2
    dist_x2 = slot_b.x2 - slot_a.x2
    dist_y2 = slot_b.y2 - slot_a.y2
    squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
    return (abs(slot_b.angle - slot_a.angle) < config.BRIDGE_ANGLE_DIFF
            and squared_dist1 < config.SQUARED_DISTANCE_THRESH
            and squared_dist2 < config.SQUARED_DISTANCE_THRESH)
