"""Short-range goals for getting unstuck: points around the robot in known free space, most open first.

Uses the same grid conventions as frontiers.py. Kept free of ROS so it can be tested anywhere.
"""
import math

import numpy as np

from pet_explorer.frontiers import world_to_cell


def clearance(grid: np.ndarray, cell: tuple[int, int], max_cells: int, occupied_threshold: int = 50) -> float:
    """Distance in cells from cell to the nearest occupied cell, capped at max_cells."""
    r, c = cell
    r0, c0 = max(r - max_cells, 0), max(c - max_cells, 0)
    rows, cols = np.nonzero(grid[r0:r + max_cells + 1, c0:c + max_cells + 1] >= occupied_threshold)
    if len(rows) == 0:
        return float(max_cells)
    return min(float(np.hypot(rows + r0 - r, cols + c0 - c).min()), float(max_cells))


def probe_goals(grid: np.ndarray, origin_xy, resolution: float, robot_xy, radii, n_angles: int,
                min_clearance: float, occupied_threshold: int = 50) -> list[tuple[float, float]]:
    """Points on circles of the given radii around the robot, on free cells at least min_clearance
    (metres) from any obstacle, ordered from most to least clearance."""
    max_cells = math.ceil(1.0 / resolution)  # look up to 1 m for the nearest obstacle
    found = []
    for radius in radii:
        for k in range(n_angles):
            angle = 2 * math.pi * k / n_angles
            xy = (robot_xy[0] + radius * math.cos(angle), robot_xy[1] + radius * math.sin(angle))
            r, c = world_to_cell(xy, origin_xy, resolution)
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                continue
            if not 0 <= grid[r, c] < occupied_threshold:
                continue  # unknown or occupied
            clear = clearance(grid, (r, c), max_cells, occupied_threshold) * resolution
            if clear >= min_clearance:
                found.append((clear, xy))
    found.sort(key=lambda item: -item[0])
    return [xy for _, xy in found]
