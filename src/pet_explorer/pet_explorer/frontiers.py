"""Frontier detection on an occupancy grid.

A frontier cell is a free cell next to an unknown one: the edge of what the robot has mapped so
far. Grids are numpy arrays indexed [row, col] with values as in nav_msgs/OccupancyGrid
(-1 unknown, 0..100 occupancy probability). Kept free of ROS so it can be tested anywhere.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

UNKNOWN = -1


@dataclass
class Frontier:
    cells: np.ndarray  # (n, 2) array of (row, col)
    goal: tuple[int, int]  # the frontier cell nearest the centroid, so it's always a free cell

    @property
    def size(self) -> int:
        return len(self.cells)


def frontier_mask(grid: np.ndarray, occupied_threshold: int = 50) -> np.ndarray:
    """Boolean mask of free cells with an unknown cell directly above, below, left or right."""
    free = (grid >= 0) & (grid < occupied_threshold)
    unknown = np.pad(grid == UNKNOWN, 1, constant_values=False)
    near_unknown = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    return free & near_unknown


def find_frontiers(grid: np.ndarray, min_size: int = 1, occupied_threshold: int = 50) -> list[Frontier]:
    """Group frontier cells into frontiers (8-connected) and drop those under min_size cells."""
    mask = frontier_mask(grid, occupied_threshold)
    rows, cols = mask.shape
    seen = np.zeros_like(mask)
    frontiers = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr in (r - 1, r, r + 1):
                for nc in (c - 1, c, c + 1):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        if len(cells) < min_size:
            continue
        cells = np.array(cells)
        nearest = np.argmin(((cells - cells.mean(axis=0)) ** 2).sum(axis=1))
        frontiers.append(Frontier(cells, (int(cells[nearest, 0]), int(cells[nearest, 1]))))
    return frontiers


def frontier_near(mask: np.ndarray, cell: tuple[int, int], radius: int) -> bool:
    """True if any frontier cell lies within radius cells (square window) of cell."""
    r, c = cell
    window = mask[max(r - radius, 0):max(r + radius + 1, 0), max(c - radius, 0):max(c + radius + 1, 0)]
    return bool(window.any())


def cell_to_world(cell: tuple[int, int], origin_xy: tuple[float, float], resolution: float) -> tuple[float, float]:
    """Centre of a grid cell in map coordinates (assumes an unrotated map origin, as slam_toolbox publishes)."""
    r, c = cell
    return (origin_xy[0] + (c + 0.5) * resolution, origin_xy[1] + (r + 0.5) * resolution)


def world_to_cell(xy: tuple[float, float], origin_xy: tuple[float, float], resolution: float) -> tuple[int, int]:
    return (math.floor((xy[1] - origin_xy[1]) / resolution), math.floor((xy[0] - origin_xy[0]) / resolution))


def cost(goal_xy, size_m, robot_xy, distance_weight, size_weight) -> float:
    """Lower is better: prefer near, large frontiers.

    This is the scoring later stages replace with novelty (see ROADMAP.md).
    """
    return distance_weight * math.dist(goal_xy, robot_xy) - size_weight * size_m
