import math

import numpy as np

from pet_learn.steps import bump_side, compose, downsample_scan, latest_before, yaw_from_quaternion


def test_downsample_keeps_the_nearest_return_per_sector():
    ranges = [5.0, 1.0, 5.0, 5.0, float('inf'), 0.0]
    assert downsample_scan(ranges, 12.0, n=3).tolist() == [1.0, 5.0, 12.0]


def test_downsample_caps_at_range_max():
    assert downsample_scan([20.0, float('nan')], 12.0, n=2).tolist() == [12.0, 12.0]


def test_latest_before():
    times = np.array([1.0, 2.0, 3.0])
    assert latest_before(times, np.array([0.5, 1.0, 2.5, 9.0])).tolist() == [-1, 0, 1, 2]


def test_yaw_from_quaternion():
    half = math.pi / 4
    assert math.isclose(yaw_from_quaternion(0, 0, math.sin(half), math.cos(half)), math.pi / 2)


def test_compose():
    x, y, yaw = compose((1.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2))
    assert math.isclose(x, 1.0) and math.isclose(y, 1.0) and math.isclose(abs(yaw), math.pi)


def test_bump_side():
    assert [bump_side(f) for f in ('bump_left', 'bump_front_left', 'bump_front_center', 'bump_front_right',
                                   'bump_right')] == [0, 0, 1, 2, 2]
