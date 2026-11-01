"""Goal-relabeled transitions for learning values and a policy (see MODEL.md).

Every recorded step can be scored against any goal. Goals are poses the robot reached later in the
session (hindsight, so the data shows how to get there) or random poses in mapped free space (so
values also learn about goals the driving never reached). For each goal, the steps up to window_s
before it become transitions, scored with the goal-reaching reward.

Only hindsight transitions teach the policy (Transitions.hindsight): the recorded commands weren't
aimed at random goals, so for those they say nothing about how to reach the goal.

A rescue (the robot got stuck and a person freed it) ends every episode that runs into it as a
failure, with RewardParams.rescue_penalty. Each rescue gets rescue_goals episodes, with goals the
robot reached later in the session or random ones: being stuck is bad wherever it was going. These
teach only the values; the policy learns to avoid getting stuck through them.
"""
from dataclasses import dataclass
import math

import numpy as np

from pet_learn.geodesic import DistanceField, passable
from pet_learn.reward import RewardParams, arrived, heading_error, motor_current, potential

GOAL_SCALE = 3.0  # goal offsets (m) are divided by this; farther goals are clipped to this distance


def goal_features(pose: np.ndarray, goal) -> np.ndarray:
    """What the policy and values see of a goal, from poses (n, 3): its offset in the robot's frame
    (clipped to GOAL_SCALE, scaled), the scaled straight-line distance, and the heading error as
    sine and cosine. (n, 5)"""
    dx, dy = goal[0] - pose[:, 0], goal[1] - pose[:, 1]
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    fx, fy = c * dx + s * dy, -s * dx + c * dy
    dist = np.hypot(fx, fy)
    shrink = np.minimum(1.0, GOAL_SCALE / np.maximum(dist, 1e-9))
    err = np.arctan2(np.sin(goal[2] - pose[:, 2]), np.cos(goal[2] - pose[:, 2]))
    return np.stack([fx * shrink / GOAL_SCALE, fy * shrink / GOAL_SCALE, np.minimum(dist, GOAL_SCALE) / GOAL_SCALE,
                     np.sin(err), np.cos(err)], axis=1).astype(np.float32)


@dataclass
class Transitions:
    """Indices into one session's steps, with the goal-dependent parts."""
    step: np.ndarray  # (m,) step index t; the next step is t + 1
    goal: np.ndarray  # (m, 5) goal features at t
    next_goal: np.ndarray  # (m, 5) goal features at t + 1
    reward: np.ndarray  # (m,)
    done: np.ndarray  # (m,) arrived at t + 1
    episode: np.ndarray  # (m,) which goal it belongs to
    ret: np.ndarray  # (m,) actual discounted return from t (the recorded driving's), for checking values
    hindsight: np.ndarray  # (m,) the goal is a pose the recorded driving went on to reach


def empty_transitions() -> Transitions:
    z = lambda *shape, dtype=np.float32: np.zeros(shape, dtype)
    return Transitions(z(0, dtype=np.int64), z(0, 5), z(0, 5), z(0), z(0), z(0, dtype=np.int64), z(0), z(0, dtype=bool))


def relabel(steps: dict, maps, ok: np.ndarray, lo: int, hi: int, n_goals: int, rng: np.random.Generator,
            window_s: float = 30.0, hindsight: float = 0.7, gamma: float = 0.99, rescue_goals: int = 100,
            p: RewardParams = RewardParams()) -> Transitions:
    """Transitions for n_goals goals, using only scored steps (ok, episodes.scored) in [lo, hi), plus rescue_goals for
    each rescue in that range."""
    rate = float(steps['rate'])
    dt = 1.0 / rate
    window = int(window_s * rate)
    pose = steps['pose'].astype(np.float64)
    currents = motor_current(steps)
    candidates = np.flatnonzero(ok[lo:hi]) + lo
    rescues = [int(r) for r in np.flatnonzero(steps.get('rescue', np.zeros(len(ok), bool))[lo:hi]) + lo
               if r > lo and ok[r - 1]]
    if len(candidates) == 0:  # e.g. a held-out tail recorded after the lidar's messages stopped
        return empty_transitions()
    jobs = [(int(rng.choice(candidates)), None) for _ in range(n_goals)]
    jobs += [(r - 1, r) for r in rescues for _ in range(rescue_goals)]
    parts = []
    for e, (goal_k, rescue) in enumerate(jobs):
        # goal_k is the episode's last step; for a rescue episode, the step before the press.
        gmap = maps.map_at(steps['t'][goal_k])
        if gmap is None:
            continue
        is_hindsight = rescue is None and rng.random() < hindsight
        later = candidates[candidates > rescue] if rescue is not None else []
        if is_hindsight:
            goal = pose[goal_k]
        elif len(later) and rng.random() < 0.5:
            goal = pose[int(rng.choice(later))]  # somewhere it went after being freed
        else:  # a random pose in mapped free space
            free = np.argwhere(passable(gmap, 0.2))
            if len(free) == 0:
                continue
            gx, gy = gmap.to_world(tuple(free[rng.integers(len(free))]))
            goal = np.array([gx, gy, rng.uniform(-math.pi, math.pi)])
        # The usable stretch of steps leading up to goal_k, at most window long.
        start = goal_k
        while start > max(lo, goal_k - window) and ok[start - 1]:
            start -= 1
        if goal_k - start < 2:
            continue
        try:
            field = DistanceField(gmap, goal[:2])
        except ValueError:
            continue
        ks = np.arange(start, goal_k + 1)
        dist = np.array([field.at(pose[k, :2]) for k in ks])
        if not np.isfinite(dist).all():
            continue  # a random goal in free space not connected to where the robot drove
        err = np.array([heading_error(pose[k, 2], goal[2]) for k in ks])
        phi = np.array([potential(d, h, p) for d, h in zip(dist, err)])
        arrive = np.array([arrived(d, h, p) for d, h in zip(dist, err)])
        reward = phi[1:] - phi[:-1] - p.time_cost * dt - p.energy_cost * currents[ks[1:]] * dt + p.arrival_bonus * arrive[1:]
        done = arrive[1:].copy()
        # Stop each episode at its first arrival.
        if done.any():
            cut = int(np.argmax(done)) + 1
            ks, reward, done = ks[:cut + 1], reward[:cut], done[:cut]
        elif rescue is not None:  # it ends stuck, waiting for a person
            reward[-1] -= p.rescue_penalty
            done[-1] = True
        if not np.isfinite(reward).all():  # NaN would spread into every weight in training
            raise ValueError(f'non-finite reward in the episode ending at step {goal_k}')
        g = goal_features(pose[ks], goal)
        ret = np.zeros(len(reward))
        acc = 0.0
        for i in range(len(reward) - 1, -1, -1):
            acc = reward[i] + gamma * acc * (1.0 - done[i])
            ret[i] = acc
        parts.append((ks[:-1], g[:-1], g[1:], reward.astype(np.float32), done, np.full(len(reward), e), ret,
                      np.full(len(reward), is_hindsight)))
    if not parts:
        return empty_transitions()
    cat = lambda i, dtype: np.concatenate([q[i] for q in parts]).astype(dtype)
    return Transitions(cat(0, np.int64), cat(1, np.float32), cat(2, np.float32), cat(3, np.float32),
                       cat(4, np.float32), cat(5, np.int64), cat(6, np.float32), cat(7, bool))
