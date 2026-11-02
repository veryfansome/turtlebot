import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pet_learn import genome as genomes

LEARN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = os.path.normpath(os.path.join(LEARN, '..', 'data', 'steps'))
has_data = pytest.mark.skipif(not os.path.isdir(STEPS), reason='needs the converted sessions in data/steps')


def test_the_baseline_genome_names_every_axis_and_loads_its_impls():
    g = genomes.load()
    assert set(g['chunks']) == set(genomes.AXES)
    for axis in genomes.AXES:
        module, params = genomes.impl(g, axis)
        assert params == {}
    assert genomes.recipe(g).startswith('arch=baseline_mlp')


def test_flags_override_the_baseline_params_and_genomes_reject_unknown_axes(tmp_path):
    g = genomes.load(None, {'optim': {'batch': 256}})
    assert g['chunks']['optim']['params'] == {'batch': 256}
    bad = tmp_path / 'g.json'
    bad.write_text(json.dumps({'chunks': {'vision': {'impl': 'x'}}}))
    with pytest.raises(ValueError):
        genomes.load(str(bad))
    missing = tmp_path / 'm.json'
    missing.write_text(json.dumps({'chunks': {'arch': {'impl': 'no_such_impl'}}}))
    with pytest.raises(ValueError):
        genomes.impl(genomes.load(str(missing)), 'arch')


def test_the_baseline_optimizer_and_architecture_are_todays():
    import torch
    arch, _ = genomes.impl(genomes.load(), 'arch')
    model = arch.build(88)
    assert model.z_dim == 64
    optim, _ = genomes.impl(genomes.load(), 'optim')
    opt, scheduler, batch = optim.make(model.parameters())
    assert isinstance(opt, torch.optim.AdamW) and scheduler is None and batch == 1024
    assert opt.defaults['lr'] == 6e-4 and opt.defaults['weight_decay'] == 1e-3


def run_eval(module, genome_path):
    env = dict(os.environ, PET_STEPS=STEPS)
    return subprocess.run([sys.executable, '-m', module, genome_path], cwd=LEARN, env=env, capture_output=True, text=True)


@has_data
def test_the_baseline_passes_the_smoke_and_leakage_gates(tmp_path):
    g = tmp_path / 'baseline.json'
    g.write_text(json.dumps(genomes.BASELINE))
    smoke = run_eval('eval.smoke', str(g))
    assert smoke.returncode == 0, smoke.stderr
    guard = run_eval('eval.guard_leakage', str(g))
    assert guard.returncode == 0, guard.stderr


@has_data
def test_a_batcher_reading_held_out_driving_is_caught(tmp_path):
    leaky = os.path.join(genomes.CHUNKS, 'batcher', 'zz_test_leaky.py')
    with open(leaky, 'w') as f:
        f.write('''import numpy as np
from pet_learn.dataset import window_batch
def make(sessions, train, norm, rng, batch_size, hide=0.05):
    def next_batch():
        s = sessions[0]
        return window_batch(s, rng.choice(s.val, batch_size), norm)  # val windows: not allowed
    return next_batch
''')
    try:
        g = tmp_path / 'leaky.json'
        g.write_text(json.dumps({'chunks': {'batcher': {'impl': 'zz_test_leaky', 'params': {}}}}))
        guard = run_eval('eval.guard_leakage', str(g))
        assert guard.returncode == 1 and 'held-out' in guard.stderr
    finally:
        os.remove(leaky)


def test_the_adapter_turns_a_run_record_into_the_engines_metrics(tmp_path, monkeypatch):
    import eval.adapter as adapter
    record = {'val_score': 0.52, 'test_score': 0.55, 'iterations': 14800, 'best_iteration': 12800, 'seconds': 400,
              'results': {'test_1s': {'scan': 0.68, 'floor': 0.4, 'odom': 0.09, 'wheels': 0.06, 'current': 0.52,
                                      'bump': 0.89, 'slip': 0.85, 'ir': 0.52, 'pose': 0.14}}}

    def fake_run(cmd, cwd, env, capture_output, text):
        out = cmd[cmd.index('--out-dir') + 1]
        os.makedirs(out)
        with open(os.path.join(out, 'world-x.json'), 'w') as f:
            json.dump(record, f)
        assert json.loads(env['PET_CODE_VERSION'])['commit'] == 'evolve-export'
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(adapter.subprocess, 'run', fake_run)
    monkeypatch.setenv('PET_STEPS', '/data')
    g = tmp_path / 'g.json'
    g.write_text(json.dumps(genomes.BASELINE))
    monkeypatch.setattr(sys, 'argv', ['adapter', str(tmp_path / 'res'), str(g), '1', 'inner', 'full'])
    adapter.main()
    metrics = json.load(open(tmp_path / 'res' / 'metrics.json'))
    assert abs(metrics['combined_score'] - 0.6539) < 1e-3 and metrics['correct']  # -log(0.52)
    assert 'scan' in metrics['text_feedback'] and 'wheels' in metrics['text_feedback']
    assert set(metrics['private']) == {'seed1'}


def test_the_score_is_a_geometric_mean_no_head_can_dominate():
    from pet_learn.train_world import HORIZONS, score
    from pet_learn.dataset import TARGETS
    heads = [n for n, (_, b) in TARGETS.items() if not b] + ['pose']
    rows = {(n, h): (0.5, 1.0) for n in heads for h in HORIZONS}
    assert abs(score(rows) - 0.5) < 1e-9
    rows[('pose', 1)] = (60.0, 1.0)  # one wild ratio
    assert score(rows) < 0.7  # an arithmetic mean would be about 3
    rows[('ir', 10)] = (float('nan'), float('nan'))  # never recorded: left out
    assert np.isfinite(score(rows))
