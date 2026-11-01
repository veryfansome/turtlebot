import math

import numpy as np

from pet_learn.dataset import mirror_inputs
from pet_learn.goals import GOAL_SCALE, goal_features
from pet_learn.train_driver import mirror_goal


def test_goal_features_are_in_the_robot_frame():
    pose = np.array([[1.0, 1.0, math.pi / 2]])  # facing +y
    fx, fy, dist, s, c = goal_features(pose, (1.0, 2.0, math.pi / 2))[0]  # 1 m straight ahead, same heading
    assert math.isclose(fx, 1 / GOAL_SCALE, abs_tol=1e-6) and math.isclose(fy, 0, abs_tol=1e-6)
    assert math.isclose(s, 0, abs_tol=1e-6) and math.isclose(c, 1, abs_tol=1e-6)
    fx, fy, *_ = goal_features(pose, (0.0, 1.0, 0.0))[0]  # 1 m to the robot's left
    assert math.isclose(fy, 1 / GOAL_SCALE, abs_tol=1e-6)


def test_far_goals_are_clipped():
    fx, fy, dist, *_ = goal_features(np.zeros((1, 3)), (10.0, 0.0, 0.0))[0]
    assert math.isclose(fx, 1.0) and math.isclose(dist, 1.0)


def test_mirroring_a_goal_matches_mirroring_the_world():
    pose = np.zeros((1, 3))
    left = goal_features(pose, (1.0, 1.0, 0.5))
    right = goal_features(pose, (1.0, -1.0, -0.5))
    assert np.allclose(mirror_goal(left), right)


def test_mirroring_inputs_twice_is_the_identity():
    x = np.random.default_rng(0).normal(size=(5, 74)).astype(np.float32)
    assert np.array_equal(mirror_inputs(mirror_inputs(x)), x)
