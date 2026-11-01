"""Shortest-path distance to a goal through a SLAM occupancy grid, the distance the reward's progress
term is measured in. The straight line would reward pushing into the couch to reach a goal behind it.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

OCCUPIED = 50  # slam_toolbox: 0 free, 100 occupied, -1 unknown


@dataclass
class GridMap:
    grid: np.ndarray  # int8 occupancy, row = y, column = x, as in nav_msgs/OccupancyGrid
    origin: tuple  # world (x, y) of cell (0, 0)'s corner
    resolution: float

    def to_cell(self, xy):
        return (int(np.floor((xy[1] - self.origin[1]) / self.resolution)),
                int(np.floor((xy[0] - self.origin[0]) / self.resolution)))

    def to_world(self, cell):
        return (self.origin[0] + (cell[1] + 0.5) * self.resolution,
                self.origin[1] + (cell[0] + 0.5) * self.resolution)

    def inside(self, cell):
        return 0 <= cell[0] < self.grid.shape[0] and 0 <= cell[1] < self.grid.shape[1]


def passable(gmap: GridMap, clearance: float) -> np.ndarray:
    """Mapped free cells at least clearance (m) from any occupied cell. Unknown cells aren't passable."""
    occupied = gmap.grid >= OCCUPIED
    free = (gmap.grid >= 0) & ~occupied
    if not occupied.any():
        return free
    return free & (ndimage.distance_transform_edt(~occupied) * gmap.resolution >= clearance)


class DistanceField:
    """Shortest-path distance (m) from anywhere in the map to one goal point.

    clearance keeps paths that far from walls; it's a little less than the robot's radius (0.17 m)
    because SLAM's walls are fuzzy and the robot fits through gaps that look tighter on the map.
    A point that isn't passable (e.g. the robot right up against a wall) is measured from the
    nearest passable cell, plus the straight line to it."""

    def __init__(self, gmap: GridMap, goal_xy, clearance: float = 0.12):
        self.gmap = gmap
        ok = passable(gmap, clearance)
        if not ok.any():
            raise ValueError('no passable cells in the map')
        # For every cell, the nearest passable cell, for snapping points that aren't passable.
        _, self.nearest = ndimage.distance_transform_edt(~ok, return_indices=True)
        goal_cell, goal_extra = self.snap(goal_xy)
        costs = np.where(ok, 1.0, np.inf)
        self.dist, _ = MCP_Geometric(costs).find_costs([goal_cell])
        self.dist = self.dist * gmap.resolution + goal_extra

    def snap(self, xy):
        """The nearest passable cell to xy, and the straight-line distance from xy to its centre
        (0 if xy's own cell is passable: distances are only as fine as the cells)."""
        cell = self.gmap.to_cell(xy)
        rows, cols = self.gmap.grid.shape
        clamped = (min(max(cell[0], 0), rows - 1), min(max(cell[1], 0), cols - 1))
        snapped = (int(self.nearest[0][clamped]), int(self.nearest[1][clamped]))
        if snapped == cell:
            return snapped, 0.0
        centre = self.gmap.to_world(snapped)
        return snapped, float(np.hypot(xy[0] - centre[0], xy[1] - centre[1]))

    def at(self, xy) -> float:
        """Shortest-path distance from xy to the goal; inf if the goal can't be reached from there."""
        cell, extra = self.snap(xy)
        return float(self.dist[cell]) + extra
