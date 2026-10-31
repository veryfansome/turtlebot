"""Building blocks for turning recorded topics into fixed-rate training steps. Free of ROS and of the
bag format, so they can be tested anywhere."""
import math

import numpy as np


def downsample_scan(ranges, range_max: float, n: int = 60) -> np.ndarray:
    """Cut a lidar scan into n equal sectors and keep the nearest return in each: an obstacle anywhere
    in a sector counts. Missing or out-of-range returns count as range_max (nothing seen)."""
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where(np.isfinite(r) & (r > 0), np.minimum(r, range_max), range_max)
    sectors = np.array_split(r, n)
    return np.array([s.min() if len(s) else range_max for s in sectors], dtype=np.float32)


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
