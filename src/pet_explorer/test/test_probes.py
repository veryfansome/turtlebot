import numpy as np

from pet_explorer.probes import clearance, probe_goals

U, F, O = -1, 0, 100  # unknown, free, occupied


def test_clearance():
    grid = np.full((11, 11), F)
    assert clearance(grid, (5, 5), 4) == 4.0  # nothing within range: capped
    grid[5, 8] = O
    assert clearance(grid, (5, 5), 4) == 3.0
    grid[2, 2] = O
    assert clearance(grid, (3, 3), 4) == np.hypot(1, 1)


def test_probe_goals_skip_unknown_and_occupied_and_prefer_open_space():
    # 10 cm cells, robot in the middle of a 2.1 m room: unknown along the left edge (col 0), wall on
    # the right (col 20).
    grid = np.full((21, 21), F)
    grid[:, 0] = U
    grid[:, 20] = O
    origin, robot = (0.0, 0.0), (1.05, 1.05)
    goals = probe_goals(grid, origin, 0.1, robot, radii=[0.6], n_angles=4, min_clearance=0.3)
    # East (1.65, 1.05) is 0.4 m from the wall; the others are at least 1 m (the cap) from anything.
    assert len(goals) == 4
    assert np.allclose(goals[-1], (1.65, 1.05))
    goals = probe_goals(grid, origin, 0.1, robot, radii=[0.6], n_angles=4, min_clearance=0.5)
    assert len(goals) == 3 and not any(np.allclose(g, (1.65, 1.05)) for g in goals)
    # At 1 m, west lands on unknown and east on the wall; only north and south remain.
    goals = probe_goals(grid, origin, 0.1, robot, radii=[1.0], n_angles=4, min_clearance=0.3)
    assert sorted(round(g[1], 2) for g in goals) == [0.05, 2.05]
