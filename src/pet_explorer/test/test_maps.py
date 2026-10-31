import numpy as np

from pet_explorer.maps import load_grid, save_grid


def test_save_and_load_round_trip(tmp_path):
    grid = np.array([[-1, 0], [100, 0]], dtype=np.int8)
    path = tmp_path / 'map.npz'
    save_grid(path, grid, (-1.5, 2.0), 0.05, robot_xy=(0.1, 0.2))
    loaded, origin, resolution, extra = load_grid(path)
    assert loaded.dtype == np.int8 and (loaded == grid).all()
    assert origin == (-1.5, 2.0)
    assert resolution == 0.05
    assert tuple(extra['robot_xy']) == (0.1, 0.2)
