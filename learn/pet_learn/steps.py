"""Building blocks for turning recorded topics into fixed-rate training steps. Free of ROS and of the
bag format, so they can be tested anywhere."""
import math

import numpy as np


def robot_sectors(ranges, angle_min: float, angle_increment: float, mount_yaw: float, range_max: float,
                  n: int = 60, min_range: float = 0.15) -> np.ndarray:
    """Cut a lidar scan into n equal sectors of the robot's frame and keep the nearest return in each.
    Sector k spans robot angles [-pi + k*w, -pi + (k+1)*w) with w = 2*pi/n: sector 0 starts directly
    behind, straight ahead is the boundary between sectors n/2 - 1 and n/2, and reversing the sector
    order mirrors the scan left to right. Missing or out-of-range returns count as range_max, and so do
    returns closer than min_range: on the TurtleBot 4 those are the robot's own support posts (about
    0.13 m away at +-60 degrees), and nothing else can be that close through the robot's body."""
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where(np.isfinite(r) & (r >= min_range), np.minimum(r, range_max), range_max)
    angles = angle_min + np.arange(len(r)) * angle_increment + mount_yaw
    sector = np.floor((np.mod(angles + np.pi, 2 * np.pi)) / (2 * np.pi / n)).astype(int) % n
    out = np.full(n, range_max, dtype=np.float32)
    np.minimum.at(out, sector, r)
    return out


def latest_before(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """For each time in grid, the index of the last entry in sorted times at or before it (-1 if none)."""
    return np.searchsorted(times, grid, side='right') - 1


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def compose(a, b):
    """The 2D pose b, given relative to pose a, in a's parent frame. Poses are (x, y, yaw)."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, math.atan2(math.sin(ayaw + byaw), math.cos(ayaw + byaw)))


def bump_side(frame_id: str) -> int:
    """Which part of the bumper a bump hazard came from: 0 left, 1 centre, 2 right. The base names
    them bump_left, bump_front_left, bump_front_center, bump_front_right, bump_right."""
    if 'left' in frame_id:
        return 0
    if 'right' in frame_id:
        return 2
    return 1
