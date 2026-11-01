import math

import numpy as np

from pet_learn.geodesic import DistanceField, GridMap, passable
from pet_learn.reward import RewardParams, arrived, episode_rewards, heading_error, potential


def room_with_wall():
    """A 2 m x 2 m free room (5 cm cells) with a wall across the middle, open only at the right end."""
    grid = np.zeros((40, 40), dtype=np.int8)
    grid[19:21, :32] = 100
    return GridMap(grid, (0.0, 0.0), 0.05)


def test_distance_goes_around_the_wall():
    field = DistanceField(room_with_wall(), goal_xy=(0.5, 1.5), clearance=0.1)
    straight = 1.0
    d = field.at((0.5, 0.5))
    assert d > straight + 1.5  # out to the gap at x ~1.6-2.0 and back
    assert math.isclose(field.at((0.5, 1.5)), 0.0, abs_tol=0.05)


def test_unknown_cells_are_not_passable():
    gmap = GridMap(np.full((10, 10), -1, dtype=np.int8), (0.0, 0.0), 0.05)
    gmap.grid[:, :5] = 0
    assert passable(gmap, 0.0)[:, :5].all() and not passable(gmap, 0.0)[:, 5:].any()


def test_points_against_a_wall_snap_to_open_space():
    field = DistanceField(room_with_wall(), goal_xy=(0.5, 0.5), clearance=0.1)
    assert field.at((0.5, 0.92)) < 0.6  # in the wall's margin, just above the goal


def test_heading_error_wraps():
    assert math.isclose(heading_error(math.radians(170), math.radians(-170)), math.radians(20))


def test_arrival_needs_position_and_heading():
    p = RewardParams()
    assert arrived(0.1, 0.1, p)
    assert not arrived(0.1, math.radians(45), p)
    assert not arrived(0.3, 0.0, p)


def test_heading_only_counts_near_the_goal():
    p = RewardParams()
    assert potential(2.0, math.pi, p) == potential(2.0, 0.0, p)
    assert potential(0.2, math.pi, p) < potential(0.2, 0.0, p)


def test_progress_telescopes_and_time_costs():
    p = RewardParams(energy_cost=0.0)
    dists = [2.0, 1.5, 1.8, 1.0, 0.5]  # a detour in the middle
    errs = [0.0] * 5
    rewards, arrived_at = episode_rewards(dists, errs, [0.0] * 5, 1.0, p)
    assert arrived_at is None
    assert math.isclose(rewards.sum(), (potential(0.5, 0, p) - potential(2.0, 0, p)) - 4 * p.time_cost)


def test_arrival_ends_the_episode_with_a_bonus():
    p = RewardParams(energy_cost=0.0)
    rewards, arrived_at = episode_rewards([0.5, 0.15, 0.1], [0.0] * 3, [0.0] * 3, 0.1, p)
    assert arrived_at == 1 and len(rewards) == 1 and rewards[0] > p.arrival_bonus


def test_pushing_costs_more_than_driving():
    p = RewardParams()
    driving, _ = episode_rewards([1.0, 1.0], [0.0, 0.0], [0.19, 0.19], 1.0, p)
    pushing, _ = episode_rewards([1.0, 1.0], [0.0, 0.0], [0.42, 0.42], 1.0, p)
    assert pushing[0] < driving[0] < 0
