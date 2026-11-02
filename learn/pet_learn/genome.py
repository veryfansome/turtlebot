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

BASELINE reproduces the training this project had before the search existed. Impls are
append-only: an archived genome must keep meaning what it meant when it was scored.
"""
import importlib.util
import json
import os

CHUNKS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evolve', 'chunks'))
AXES = ('arch', 'objective', 'optim', 'batcher')
BASELINE = {'chunks': {
    'arch': {'impl': 'baseline_mlp', 'params': {}},
    'objective': {'impl': 'baseline_world_loss', 'params': {}},
    'optim': {'impl': 'baseline_adamw', 'params': {}},
    'batcher': {'impl': 'baseline_uniform', 'params': {}},
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
        spec = chunks.setdefault(axis, json.loads(json.dumps(BASELINE['chunks'][axis])))
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
