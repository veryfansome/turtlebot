import math

import numpy as np

from pet_learn.steps import bump_side, compose, latest_before, robot_sectors, yaw_from_quaternion


def test_robot_sectors_rotate_into_the_robot_frame():
    # 4 beams in the lidar frame at -90, 0, 90, 180 degrees; lidar mounted at +90 degrees, so in the
    # robot frame they're at 0 (ahead), 90 (left), 180 (behind), -90 (right).
    ranges = [1.0, 2.0, 3.0, 4.0]
    s = robot_sectors(ranges, -math.pi / 2, math.pi / 2, math.pi / 2, 9.0, n=4)
    # Sectors: [-180, -90) behind-right, [-90, 0) right, [0, 90) ahead-left, [90, 180) left-behind
    assert s.tolist() == [3.0, 4.0, 1.0, 2.0]


def test_reversed_sectors_mirror_left_and_right():
    left_only = robot_sectors([1.0], math.radians(30), 0.0, 0.0, 9.0, n=8)  # one beam 30 degrees left
    right_only = robot_sectors([1.0], math.radians(-30), 0.0, 0.0, 9.0, n=8)
    assert left_only[::-1].tolist() == right_only.tolist()


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
