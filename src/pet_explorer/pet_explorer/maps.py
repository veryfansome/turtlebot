"""Saving occupancy grids to disk, e.g. the map from before a pickup, for matching against later.

Files are numpy .npz archives holding the grid (values as in nav_msgs/OccupancyGrid), its origin
and resolution, plus any extra arrays (such as the robot's position at the time).
"""
import numpy as np


def save_grid(path, grid: np.ndarray, origin_xy, resolution: float, **extra):
    np.savez_compressed(path, grid=grid, origin=np.asarray(origin_xy, dtype=float),
                        resolution=float(resolution), **{k: np.asarray(v) for k, v in extra.items()})


def load_grid(path):
    """Returns (grid, origin_xy, resolution, extra) where extra is a dict of the other arrays."""
    with np.load(path) as data:
        extra = {k: data[k] for k in data.files if k not in ('grid', 'origin', 'resolution')}
        return data['grid'], tuple(data['origin']), float(data['resolution']), extra
