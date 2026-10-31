"""The reward for learning to drive: reach a goal pose (position and heading) efficiently.

Every step earns the progress made toward the goal (the drop in a potential), minus a cost for the
time and the energy it took, plus a bonus on arrival. Progress is potential-based (Ng et al., 1999):
it telescopes over an episode, so driving in circles can't farm it, and it doesn't change which
behaviour is best, only how quickly the learner finds out.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RewardParams:
    goal_radius: float = 0.2  # arrived within this distance (m) ...
    goal_heading: float = math.radians(20)  # ... and this heading error (rad)
    # Near the goal, heading counts as progress too: heading_weight metres per radian of heading
    # error, phased in from heading_range (m) away down to goal_radius.
    heading_weight: float = 0.3
    heading_range: float = 1.0
    arrival_bonus: float = 1.0  # about 3 m of progress
    # Costs per second. Time: 0.1/s is a third of the progress driving at full speed (0.3 m/s) earns.
    # Energy, per amp of motor current (both wheels): in the first recorded session normal driving
    # drew about 0.19 A and pushing against something about 0.42 A, so energy adds about 20% to the
    # time cost while driving and 40-70% while pushing.
    time_cost: float = 0.1
    energy_cost: float = 0.1


def heading_error(yaw: float, goal_yaw: float) -> float:
    return math.atan2(math.sin(goal_yaw - yaw), math.cos(goal_yaw - yaw))


def potential(dist: float, heading_err: float, p: RewardParams) -> float:
    near = min(max((p.heading_range - dist) / (p.heading_range - p.goal_radius), 0.0), 1.0)
    return -(dist + p.heading_weight * abs(heading_err) * near)


def arrived(dist: float, heading_err: float, p: RewardParams) -> bool:
    return dist <= p.goal_radius and abs(heading_err) <= p.goal_heading


def episode_rewards(dists, heading_errs, currents_a, dt: float, p: RewardParams = RewardParams()):
    """Rewards for an episode's steps, given each step's shortest-path distance to the goal, heading
    error, and total motor current (A). rewards[k] is for the move from step k to step k+1, so there
    is one fewer reward than steps. The episode ends at the first arrival: steps after it get none.
    Returns (rewards, arrived_at): arrived_at is the step index of arrival, or None."""
    rewards = []
    for k in range(len(dists) - 1):
        r = (potential(dists[k + 1], heading_errs[k + 1], p) - potential(dists[k], heading_errs[k], p)
             - p.time_cost * dt - p.energy_cost * currents_a[k + 1] * dt)
        if arrived(dists[k + 1], heading_errs[k + 1], p):
            rewards.append(r + p.arrival_bonus)
            return np.array(rewards), k + 1
        rewards.append(r)
    return np.array(rewards), None
