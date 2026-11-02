"""Shared by the world-model search's eval scripts: a small real slice of the pinned data."""
import os

import numpy as np

from eval.manifest import steps_files


def small_sessions(n: int = 2):
    """The n smallest pinned sessions as dataset.SessionData, with their training-window pairs and a
    Normalizer: enough real data to exercise a genome in seconds."""
    from pet_learn.dataset import Normalizer, SessionData
    paths = sorted(steps_files(os.environ['PET_STEPS']), key=os.path.getsize)[:n]
    sessions = [SessionData(dict(np.load(p))) for p in paths]
    train = [(i, s) for i, data in enumerate(sessions) for s in data.train]
    return sessions, train, Normalizer(sessions)
