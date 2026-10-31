"""Goal-reaching episodes: one start, one goal pose, scored with the reward.

Recordings have no goals, so they're cut into episodes after the fact (hindsight relabeling): any
later pose in a session is a goal the steps before it reached. One session gives as many episodes as
we like, of any length, each scored against its own goal.
"""
from dataclasses import dataclass
import math

import numpy as np

from pet_learn.geodesic import DistanceField
from pet_learn.reward import RewardParams, arrived, episode_rewards, heading_error

MAX_STEP = 0.1  # m per 0.1 s step; more is a SLAM correction or reset, not driving (top speed 0.03)
STUCK_FORWARD = 0.03  # told to go forward faster than this (m/s) ...
STUCK_FLOOR = 0.01  # ... while the floor sensor says slower than this: pushing without moving


@dataclass
class Episode:
    start: int  # step index
    end: int  # step index it ended at: arrival, or where it gave up
    goal: tuple  # x, y, yaw
    success: bool
    shortest: float  # shortest path from the start to the goal through the map (m)
    path: float  # distance actually driven (m, SLAM)
    duration: float  # s
    energy: float  # motor current over time, both wheels (A·s)
    bumps: int
    stuck: float  # s told to go forward while not moving
    reward: float  # total

    @property
    def spl(self) -> float:
        """Success weighted by path length: shortest / driven if it arrived, else 0. 1.0 is perfect."""
        return self.shortest / max(self.path, self.shortest) if self.success else 0.0


def usable(steps: dict) -> np.ndarray:
    """Steps an episode can include: SLAM has a pose, the robot is driving on the floor (not docked
    or picked up), and the pose didn't jump (a SLAM reset or big correction)."""
    pose = steps['pose']
    ok = np.isfinite(pose).all(axis=1) & ~steps['docked'] & ~steps['kidnapped']
    jump = np.hypot(*np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]).T)
    return ok & ~(jump > MAX_STEP)


def evaluate(steps: dict, maps, start: int, last: int, goal, p: RewardParams = RewardParams(),
             clearance: float = 0.12) -> Episode:
    """Score driving from step start toward goal (x, y, yaw), giving up after step last."""
    rate = float(steps['rate'])
    dt = 1.0 / rate
    gmap = maps.map_at(steps['t'][last])
    field = DistanceField(gmap, goal[:2], clearance)
    span = range(start, last + 1)
    pose = steps['pose']
    dists = [field.at(pose[k, :2]) for k in span]
    errs = [heading_error(pose[k, 2], goal[2]) for k in span]
    currents = np.abs(steps['wheel_current'][start:last + 1]).sum(axis=1) / 1000.0
    rewards, arrived_at = episode_rewards(dists, errs, currents, dt, p)
    end = start + (arrived_at if arrived_at is not None else len(dists) - 1)
    moved = np.hypot(*np.diff(pose[start:end + 1, :2], axis=0).T)
    cmd = steps['cmd'][start:end + 1, 0]
    floor = steps['mouse_speed'][start:end + 1]
    return Episode(
        start=start, end=end, goal=tuple(float(g) for g in goal), success=arrived_at is not None,
        shortest=dists[0], path=float(np.minimum(moved, MAX_STEP).sum()), duration=(end - start) * dt,
        energy=float(currents[:end - start + 1].sum() * dt),
        bumps=int(steps['bump'][start + 1:end + 1].any(axis=1).sum()),
        stuck=float(((cmd > STUCK_FORWARD) & (floor < STUCK_FLOOR)).sum() * dt),
        reward=float(rewards.sum()),
    )


def hindsight_episodes(steps: dict, maps, n: int, rng: np.random.Generator, min_s: float = 3.0,
                       max_s: float = 60.0, min_dist: float = 0.3, p: RewardParams = RewardParams()):
    """Up to n episodes cut from a recorded session: a random start, and the pose the robot had
    min_s to max_s seconds later as the goal. Episodes stay within one stretch of usable steps, and
    goals closer than min_dist (straight line) are skipped. Arrival is checked against the goal, so
    an episode can end early if the robot passed through the goal pose on the way."""
    rate = float(steps['rate'])
    ok = usable(steps)
    # Each step's usable stretch: the index where it ends.
    run_end = np.full(len(ok), -1)
    end = -1
    for k in range(len(ok) - 1, -1, -1):
        end = (k if end < 0 else end) if ok[k] else -1
        run_end[k] = end
    lo, hi = int(min_s * rate), int(max_s * rate)
    starts = np.flatnonzero(ok & (run_end - np.arange(len(ok)) >= lo))
    episodes = []
    for _ in range(n * 5):
        if len(episodes) >= n or len(starts) == 0:
            break
        start = int(rng.choice(starts))
        goal_k = start + int(rng.integers(lo, min(hi, run_end[start] - start) + 1))
        goal = steps['pose'][goal_k]
        if math.dist(goal[:2], steps['pose'][start, :2]) < min_dist:
            continue
        try:
            episodes.append(evaluate(steps, maps, start, goal_k, goal, p))
        except ValueError:
            continue  # no passable map yet
    return episodes
