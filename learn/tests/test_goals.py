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


def test_the_driver_is_symmetric_by_construction():
    import torch
    from pet_learn.model import WorldModel
    from pet_learn.train_driver import Driver, encode_pair, input_mirror
    torch.manual_seed(0)
    world, driver, m = WorldModel(82, z_dim=16, hidden=32), Driver(16, hidden=32), input_mirror(82)
    x, g, a = torch.randn(8, 82), torch.randn(8, 5), torch.randn(8, 2)
    flip_g, flip_a = torch.tensor([1., -1, 1, -1, 1]), torch.tensor([1., -1])
    with torch.no_grad():
        s, sm = encode_pair(world, x, m), encode_pair(world, x @ m, m)
        assert torch.allclose(driver.policy(sm, g * flip_g)[0], driver.policy(s, g)[0] * flip_a, atol=1e-5)
        assert torch.allclose(driver.value(sm, g * flip_g), driver.value(s, g), atol=1e-5)
        q, qm = driver.q(s, g, a), driver.q(sm, g * flip_g, a * flip_a)
        assert all(torch.allclose(u, w, atol=1e-5) for u, w in zip(q, qm))
