import numpy as np

from pet_explorer.frontiers import (
    cell_to_world, cost, find_frontiers, frontier_mask, frontier_near, goal_cell, world_to_cell,
)

U, F, O = -1, 0, 100  # unknown, free, occupied


def test_frontier_mask_marks_free_cells_next_to_unknown():
    grid = np.array([
        [F, F, U],
        [F, F, F],
        [O, F, F],
    ])
    expected = np.array([
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ])
    assert (frontier_mask(grid) == expected).all()


def test_occupied_cells_next_to_unknown_are_not_frontiers():
    grid = np.array([[O, U], [O, U]])
    assert not frontier_mask(grid).any()


def test_diagonal_unknown_does_not_make_a_frontier():
    grid = np.array([[F, F], [F, U]])
    assert (frontier_mask(grid) == np.array([[False, True], [True, False]])).all()


def test_find_frontiers_groups_connected_cells_and_filters_small_ones():
    grid = np.full((6, 6), F)
    grid[:, 0] = U  # frontier down column 1...
    grid[3:, 1:3] = O  # ...cut short by a wall: rows 0-2, 3 cells
    grid[5, 5] = U  # frontier of (4, 5) and (5, 4), diagonal neighbours: 2 cells
    frontiers = find_frontiers(grid)
    assert sorted(f.size for f in frontiers) == [2, 3]
    assert [f.size for f in find_frontiers(grid, min_size=3)] == [3]


def test_frontier_goal_is_a_frontier_cell():
    grid = np.full((5, 5), F)
    grid[0, :] = U
    (frontier,) = find_frontiers(grid)
    assert frontier.goal == (1, 2)
    assert frontier_mask(grid)[frontier.goal]


def test_goal_cell_moves_away_from_avoided_spots():
    grid = np.full((5, 11), F)
    grid[0, :] = U  # frontier along row 1, cols 0-10, centred on (1, 5)
    (frontier,) = find_frontiers(grid)
    origin, resolution = (0.0, 0.0), 1.0
    assert goal_cell(frontier, [], 1.5, origin, resolution) == (1, 5)
    # Avoiding the centre (world (5.5, 1.5)) shifts the goal just outside the radius, not off the frontier.
    cell = goal_cell(frontier, [(5.5, 1.5)], 1.5, origin, resolution)
    assert cell in [(1, 3), (1, 7)]
    # Avoiding every cell leaves no goal.
    assert goal_cell(frontier, [(5.5, 1.5)], 20.0, origin, resolution) is None


def test_frontier_near():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert frontier_near(mask, (6, 6), 1)
    assert not frontier_near(mask, (8, 8), 2)
    assert not frontier_near(mask, (-20, -20), 3)


def test_world_cell_round_trip():
    origin, resolution = (-2.0, 1.0), 0.05
    cell = (12, 34)
    assert world_to_cell(cell_to_world(cell, origin, resolution), origin, resolution) == cell


def test_cost_prefers_near_and_large():
    robot = (0.0, 0.0)
    near = cost((1.0, 0.0), 1.0, robot, 3.0, 1.0)
    far = cost((5.0, 0.0), 1.0, robot, 3.0, 1.0)
    big = cost((1.0, 0.0), 4.0, robot, 3.0, 1.0)
    assert big < near < far
