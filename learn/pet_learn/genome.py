"""Genomes for the world model's evolutionary search (learn/evolve, the evolve plugin's registry mode).

A genome picks one implementation per design axis of the world model's training, each a file in
evolve/chunks/<axis>/<impl>.py, with optional params:

    {"chunks": {"arch": {"impl": "baseline_mlp", "params": {"z": 64}}, "objective": {...}, ...}}

The axes and what an impl must expose (each baseline file's docstring is the full contract):

- arch: build(input_dim, **params) -> the world model (pet_learn.model.WorldModel's interface).
- objective: loss(model, batch, **params) -> (total, parts): the training loss.
- optim: make(parameters, **params) -> (optimizer, scheduler or None, batch size).
- batcher: make(sessions, train, norm, rng, batch_size, **params) -> next_batch(): standardized
  training windows, augmented.

BASELINE is the default genome: what a plain train_world run trains and the evolve search's starting
point (evolve.json's baseline_genome is the same). Its params were tuned on RunPod (2026-11-02:
waves bs-lr-1, shape-1, reg-1, sched-1, confirm-1, part-drop-1 in cloud/waves/): dropout 0.1 in the
encoder and the dynamics (none in the heads), sensors hidden in 10% of windows, batches of 2,048.
Over 3 seeds it scored (on the score of then, horizons up to 1 s) 1.225 against 1.119 for ORIGINAL (val fitness, higher is better) and cut the
test error relative to the naive baselines from 0.401 to 0.364. Dropout in the heads too scored the
same (1.209 +/- 0.018 against 1.126 +/- 0.020 over 8 seeds) but made the 1 s lidar prediction worse
(0.727 against 0.710, where without head dropout it's 0.707).

The loss weighs the lidar's term 3x, the IR's 2x and the bump's 0.5x (waves head-w-1, head-w-2,
head-w-conf-1, 2026-11-02): over 5 fresh seeds this scored (horizons up to 1 s) 1.240 against 1.212 for equal weights,
with the lidar's error 1 s ahead 12% lower (0.635 against 0.722) and IR's lower too (0.483 against
0.502), at about 3% on motor current and 2% on floor speed, and slip AUC 0.875 against 0.908.

ORIGINAL is the same impls with no params: their defaults, which reproduce the training this project
had before the search existed, bit for bit. A genome that leaves out an axis gets ORIGINAL's, so its
meaning doesn't change when BASELINE does. Impls are append-only: an archived genome must keep
meaning what it meant when it was scored.
"""
import importlib.util
import json
import os

CHUNKS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evolve', 'chunks'))
AXES = ('arch', 'objective', 'optim', 'batcher')
ORIGINAL = {'chunks': {
    'arch': {'impl': 'baseline_mlp', 'params': {}},
    'objective': {'impl': 'baseline_world_loss', 'params': {}},
    'optim': {'impl': 'baseline_adamw', 'params': {}},
    'batcher': {'impl': 'baseline_uniform', 'params': {}},
}}
BASELINE = {'chunks': {
    'arch': {'impl': 'baseline_mlp', 'params': {'enc_dropout': 0.1, 'dyn_dropout': 0.1}},
    'objective': {'impl': 'baseline_world_loss', 'params': {'scan_weight': 3.0, 'ir_weight': 2.0, 'bump_weight': 0.5}},
    'optim': {'impl': 'baseline_adamw', 'params': {'batch': 2048, 'lr': 0.0006, 'weight_decay': 0.001}},
    'batcher': {'impl': 'baseline_uniform', 'params': {'hide': 0.1}},
}}


def load(path=None, overrides=None) -> dict:
    """The genome at path (the baseline if None), with any axis params overridden by overrides
    ({axis: {param: value}}; how the command-line flags of a plain run set the baseline's params)."""
    if path is None:
        genome = json.loads(json.dumps(BASELINE))
    else:
        with open(path) as f:
            genome = json.load(f)
    return load_dict(genome, overrides, path)


def load_dict(genome: dict, overrides=None, path=None) -> dict:
    """A genome given as a dict, checked and completed like load() (a copy; the argument is untouched)."""
    genome = json.loads(json.dumps(genome))
    chunks = genome.get('chunks')
    if not isinstance(chunks, dict):
        raise ValueError(f'{path}: a genome needs a "chunks" dict')
    for axis in AXES:
        spec = chunks.setdefault(axis, json.loads(json.dumps(ORIGINAL['chunks'][axis])))
        spec.setdefault('params', {})
        if not isinstance(spec.get('impl'), str):
            raise ValueError(f'{path}: axis {axis!r} needs an impl name')
    unknown = set(chunks) - set(AXES)
    if unknown:
        raise ValueError(f'{path}: unknown axes {sorted(unknown)} (axes: {list(AXES)})')
    for axis, params in (overrides or {}).items():
        chunks[axis]['params'].update(params)
    return genome


def impl(genome: dict, axis: str):
    """(module, params) for the genome's impl on axis, loaded from evolve/chunks/<axis>/<impl>.py."""
    spec = genome['chunks'][axis]
    path = os.path.join(CHUNKS, axis, spec['impl'] + '.py')
    if not os.path.isfile(path):
        have = sorted(f[:-3] for f in os.listdir(os.path.join(CHUNKS, axis)) if f.endswith('.py'))
        raise ValueError(f'axis {axis!r} has no impl {spec["impl"]!r} (have {have})')
    module_spec = importlib.util.spec_from_file_location(f'evolve_chunks.{axis}.{spec["impl"]}', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, dict(spec['params'])


def recipe(genome: dict) -> str:
    return ', '.join(f'{axis}={genome["chunks"][axis]["impl"]}' for axis in AXES)
