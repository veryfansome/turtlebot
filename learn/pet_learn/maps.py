"""The SLAM maps saved with a session's steps (pet_learn.convert writes <session>.maps.npz)."""
import numpy as np

from pet_learn.geodesic import GridMap


class SessionMaps:
    """A session's SLAM maps in time order; map_at(t) is the latest one at time t (s since the first scan)."""

    def __init__(self, path: str):
        f = np.load(path)
        self.t = f['t']
        offsets = np.concatenate([[0], np.cumsum(f['shape'].prod(axis=1))])
        self.maps = [
            GridMap(f['data'][offsets[k]:offsets[k + 1]].reshape(f['shape'][k]),
                    tuple(f['origin'][k]), float(f['resolution'][k]))
            for k in range(len(self.t))
        ]

    def map_at(self, t: float):
        k = np.searchsorted(self.t, t, side='right') - 1
        return self.maps[k] if k >= 0 else None
