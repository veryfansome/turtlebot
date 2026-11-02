"""Goal-reaching episodes: one start, one goal pose, scored with the reward.

Recordings have no goals, so they're cut into episodes after the fact (hindsight relabeling): any
later pose in a session is a goal the steps before it reached. One session gives as many episodes as
we like, of any length, each scored against its own goal.
"""
from dataclasses import dataclass
import math

import numpy as np

from pet_learn.geodesic import DistanceField
from pet_learn.reward import RewardParams, arrived, episode_rewards, heading_error, motor_current

MAX_STEP = 0.1  # m per 0.1 s step; more is a SLAM correction or reset, not driving (top speed 0.03)
STUCK_FORWARD = 0.03  # told to go forward faster than this (m/s) ...
STUCK_FLOOR = 0.01  # ... while the floor sensor says slower than this: pushing without moving
RESCUE_HANDLING_S = 10.0  # after a rescue press, a person is handling the robot until it's set down, or this long
# Without scans, SLAM can't correct the pose, which then follows the wheel odometry alone: it drifts,
# so steps this long after the latest scan aren't used ...
SCAN_GAP_LIMIT_S = 60.0
# ... and goals and rewards, which measure distances on the map, use only steps this close to one.
SCORED_SCAN_GAP_S = 10.0
# After a pickup SLAM starts a fresh map (/drive/slam_reset) once the robot is set down. From each
# set-down until the fresh map is under way, poses aren't trustworthy: until RESET_SETTLE_S after the
# reset (SLAM's first fix on the new map), and until a map published after the reset exists to score
# them on. A reset more than RESET_WITHIN_S after a set-down isn't counted as its.
RESET_SETTLE_S = 2.0
RESET_WITHIN_S = 10.0


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


def usable(steps: dict, max_scan_gap: float = SCAN_GAP_LIMIT_S) -> np.ndarray:
    """Steps the model can learn from: SLAM has a pose (and a scan in the last max_scan_gap s), the
    robot is driving on the floor (not docked or picked up, not being rescued), and the pose didn't
    jump (a SLAM reset or big correction)."""
    pose = steps['pose']
    ok = (np.isfinite(pose).all(axis=1) & ~steps['docked'] & ~steps['kidnapped'] & ~rescue_handling(steps)
          & ~reset_settling(steps))
    if 'scan_age' in steps:  # sessions converted before it was recorded had no gaps kept
        ok &= steps['scan_age'] <= max_scan_gap
    jump = np.hypot(*np.diff(pose[:, :2], axis=0, prepend=pose[:1, :2]).T)
    return ok & ~(jump > MAX_STEP)


def scored(steps: dict) -> np.ndarray:
    """Steps an episode can include, scored with the reward: usable, with a scan in the last
    SCORED_SCAN_GAP_S (so the pose they're measured on is still close to the map), and with the wheels
    powered. Without power (an e-stop) the robot can't act, which isn't its failure: episodes end
    there, cut short without a penalty. The world model still learns from those steps, with
    wheels_enabled as an input, that commands then do nothing."""
    return usable(steps, SCORED_SCAN_GAP_S) & wheels_enabled(steps)


def wheels_enabled(steps: dict) -> np.ndarray:
    """(n,) the motors had power (sessions converted before this was recorded: assume so)."""
    return steps.get('wheels_enabled', np.ones(len(steps['t']), bool)).astype(bool)


def map_segment(steps: dict) -> np.ndarray:
    """(n,) which SLAM map each step's pose is on: 0 until the first reset after a pickup, then 1, ..."""
    resets = steps.get('slam_reset', np.zeros(len(steps['t']), bool))
    return np.cumsum(resets)


def reset_settling(steps: dict) -> np.ndarray:
    """Steps from each set-down until the fresh map after the SLAM reset that follows it is under way
    (see RESET_SETTLE_S). Every set-down counts: a person often lifts the robot twice in quick
    succession, and between the two its pose is still on the paused old map."""
    n = len(steps['t'])
    settling = np.zeros(n, dtype=bool)
    resets = np.flatnonzero(steps.get('slam_reset', np.zeros(n, bool)))
    if len(resets) == 0:
        return settling
    rate, t = float(steps['rate']), steps['t']
    map_time = steps.get('map_time')
    for r in resets:
        end = int(r + RESET_SETTLE_S * rate) + 1
        if map_time is not None:  # and until a map from after the reset has arrived
            fresh = np.flatnonzero(map_time[r:] >= t[r])
            end = max(end, int(r + fresh[0]) if len(fresh) else n)
        settling[r:min(n, end)] = True
    lifted = steps['kidnapped']
    set_downs = np.flatnonzero(np.diff(lifted.astype(int)) == -1) + 1
    for b in set_downs:
        following = resets[resets >= b]
        if not len(following):
            continue
        r = int(following[0])
        # Lifted again before that reset (it waited for the last set-down): still on the paused old map.
        if lifted[b:r].any() or r - b <= RESET_WITHIN_S * rate:
            settling[b:r] = True
    return settling


def rescue_handling(steps: dict) -> np.ndarray:
    """Steps where a person is freeing the robot: from each rescue press until it's been picked up and
    set down again, or RESCUE_HANDLING_S if it isn't picked up."""
    n = len(steps['t'])
    handling = np.zeros(n, dtype=bool)
    rescue = steps.get('rescue', np.zeros(n, dtype=bool))
    limit = int(RESCUE_HANDLING_S * float(steps['rate']))
    for r in np.flatnonzero(rescue):
        end = min(n, r + limit)
        lifted = np.flatnonzero(steps['kidnapped'][r:end])
        if len(lifted):  # until set down
            k = r + int(lifted[0])
            while k < n and steps['kidnapped'][k]:
                k += 1
            end = k
        handling[r:end] = True
    return handling


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
    currents = motor_current(steps)[start:last + 1]
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
    ok = scored(steps)
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
