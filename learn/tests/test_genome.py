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


def test_the_new_arch_objective_and_optim_params_default_to_the_old_model():
    """Depth, dropout, warmup and decay: at their defaults the baseline builds the same layers and draws
    the same initial weights as before they existed (the fingerprint is the pre-change WorldModel's, at
    commit 6150a07, seed 0, torch as locked), and no scheduler."""
    import torch
    arch, _ = genomes.impl(genomes.load(), 'arch')
    torch.manual_seed(0)
    new = arch.build(88, enc_layers=2, dyn_layers=2, head_layers=1, dropout=0.0)
    sd = new.state_dict()
    assert len(sd) == 78
    assert sum(float(v.double().sum()) for v in sd.values()) == pytest.approx(1819.1722541377292, rel=1e-12)
    assert sum(float((v.double() ** 2).sum()) for v in sd.values()) == pytest.approx(2424.1014644946567, rel=1e-12)
    assert not any(isinstance(m, torch.nn.Dropout) for m in new.modules())
    optim, _ = genomes.impl(genomes.load(), 'optim')
    assert optim.make(new.parameters(), warmup=0, decay_halflife=0)[1] is None


def test_depth_and_dropout_change_the_model_and_a_schedule_warms_up_then_halves():
    import torch
    arch, _ = genomes.impl(genomes.load(), 'arch')
    deep = arch.build(88, enc_layers=3, dyn_layers=4, head_layers=2, dropout=0.1)
    linears = lambda seq: sum(isinstance(m, torch.nn.Linear) for m in seq)
    assert linears(deep.encoder) == 4 and linears(deep.dynamics) == 5 and linears(deep.pose_head) == 3
    assert all(linears(h) == 3 for h in deep.heads.values())
    assert sum(isinstance(m, torch.nn.Dropout) for m in deep.encoder) == 3
    for bad in ({'dyn_layers': 0}, {'dropout': -0.1}, {'dropout': 1.0}):
        with pytest.raises(ValueError):
            arch.build(88, **bad)
    optim, _ = genomes.impl(genomes.load(), 'optim')
    opt, scheduler, _ = optim.make(deep.parameters(), lr=1e-3, warmup=4, decay_halflife=10)
    lrs = []
    for _ in range(24):
        lrs.append(opt.param_groups[0]['lr'])
        opt.step(); scheduler.step()
    assert lrs[:4] == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3])
    assert lrs[4] == pytest.approx(1e-3) and lrs[14] == pytest.approx(5e-4)
    opt, scheduler, _ = optim.make(deep.parameters(), lr=1e-3, warmup=2.5)  # a fractional warmup never overshoots
    seen = []
    for _ in range(5):
        seen.append(opt.param_groups[0]['lr'])
        opt.step(); scheduler.step()
    assert max(seen) == pytest.approx(1e-3)


@has_data
def test_the_objective_weighs_consistency_as_asked(monkeypatch):
    import torch
    from eval.common import small_sessions
    from pet_learn.train_world import to_torch
    monkeypatch.setenv('PET_STEPS', STEPS)
    sessions, train, norm = small_sessions()
    batcher, params = genomes.impl(genomes.load(), 'batcher')
    batch = to_torch(batcher.make(sessions, train, norm, np.random.default_rng(0), 64, **params)(), 'cpu')
    objective, _ = genomes.impl(genomes.load(), 'objective')
    arch, _ = genomes.impl(genomes.load(), 'arch')
    torch.manual_seed(0)
    model = arch.build(batch['inputs'].shape[-1])
    _, one = objective.loss(model, batch)
    _, three = objective.loss(model, batch, consistency=3.0)
    assert three['consistency'].item() == pytest.approx(3 * one['consistency'].item(), rel=1e-5)
    assert one['scan'].item() == pytest.approx(three['scan'].item())
