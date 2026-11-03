"""Offline tests for the RunPod wave tooling (no pods, no network). Run: cd learn && uv run pytest ../cloud/tests -q"""
import json
import os
import pathlib
import sys

import pytest

CLOUD = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, str(CLOUD.parent / 'learn'))
import wave  # noqa: E402
import wave_pod  # noqa: E402
from pet_learn import genome as genomes  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(wave, 'WAVES', str(tmp_path / 'waves'))
    monkeypatch.setattr(wave, 'RESULTS', str(tmp_path / 'results'))
    return tmp_path


def test_a_grid_is_every_combination_or_the_lists_paired():
    base = genomes.load()
    before = json.loads(json.dumps(base))
    full = wave.grid_genomes(base, {'optim.batch': [256, 1024], 'optim.lr': [3e-4, 6e-4]}, pairs=False)
    assert len(full) == 4 and 'grid-batch256-lr0.0003' in full
    assert full['grid-batch256-lr0.0003']['chunks']['optim']['params'] == {**base['chunks']['optim']['params'], 'batch': 256, 'lr': 3e-4}
    paired = wave.grid_genomes(base, {'optim.batch': [256, 1024], 'optim.lr': [3e-4, 6e-4]}, pairs=True)
    assert sorted(paired) == ['grid-batch1024-lr0.0006', 'grid-batch256-lr0.0003']
    assert base == before  # the base is untouched


def test_a_wave_records_its_genomes_and_refuses_to_overwrite(dirs):
    out = wave.write_wave('w1', {'baseline': genomes.load()}, [0, 1], 'inner', 'full', 3, 3600)
    w = json.loads((pathlib.Path(out) / 'wave.json').read_text())
    assert w['genomes'] == {'baseline': 'genomes/baseline.json'} and w['chunks'] == []  # baseline impls are committed
    assert json.loads((pathlib.Path(out) / 'genomes' / 'baseline.json').read_text())['chunks']['arch']['impl'] == 'baseline_mlp'
    with pytest.raises(SystemExit):
        wave.write_wave('w1', {'baseline': genomes.load()}, [0], 'inner', 'full', 3, 3600)


def test_an_uncommitted_impl_travels_with_the_wave(dirs):
    new = pathlib.Path(genomes.CHUNKS) / 'optim' / 'zz_test_new_impl.py'
    new.write_text('def make(parameters, **p):\n    raise NotImplementedError\n')
    try:
        g = genomes.load_dict({'chunks': {'optim': {'impl': 'zz_test_new_impl'}}})
        out = wave.write_wave('w2', {'cand': g}, [0], 'inner', 'full', 3, 3600)
        assert (pathlib.Path(out) / 'chunks' / 'optim' / 'zz_test_new_impl.py').exists()
        assert json.loads((pathlib.Path(out) / 'wave.json').read_text())['chunks'] == ['optim/zz_test_new_impl.py']
    finally:
        new.unlink()


def test_record_states_and_verify_exit_codes(dirs):
    wave.write_wave('w3', {'a': genomes.load(), 'b': genomes.load()}, [0], 'inner', 'full', 2, 3600)
    assert wave.record_state({'correct': True, 'fitness': 0.6}) == 'ok'
    assert wave.record_state({'correct': True, 'fitness': None}) == 'bad'
    assert wave.record_state({'correct': True, 'fitness': True}) == 'bad'
    assert wave.record_state({'correct': False, 'fitness': None}) == 'failed'
    results = dirs / 'results' / 'w3'
    results.mkdir(parents=True)
    assert wave.verify('w3') == 2  # nothing home yet
    (results / 'a.json').write_text(json.dumps({'correct': False, 'fitness': None}))
    (results / 'b.json').write_text(json.dumps({'correct': False, 'fitness': None}))
    assert wave.verify('w3') == 3  # every genome failed: the environment, not the candidates
    (results / 'b.json').write_text(json.dumps({'correct': True, 'fitness': 0.61}))
    assert wave.verify('w3') == 0


def test_the_estimate_counts_jobs_over_concurrency_plus_overhead(dirs):
    wave.write_wave('w4', {'a': genomes.load(), 'b': genomes.load()}, [0, 1, 2], 'inner', 'full', 3, 3600)
    est = wave.estimate(wave.load_wave('w4'), minutes_per_job=12, price=0.6)
    assert est['jobs'] == 6
    assert est['hours'] == pytest.approx((wave.OVERHEAD_MINUTES + 6 * 12 / 3) / 60)
    assert est['dollars'] == pytest.approx(est['hours'] * 0.6)


def test_seeds_fold_into_one_ingestable_record(tmp_path):
    w = {'name': 'w5', 'seeds': [0, 1]}
    for seed, score in ((0, 0.6), (1, 0.7)):
        d = tmp_path / 'g' / f's{seed}'
        d.mkdir(parents=True)
        (d / 'metrics.json').write_text(json.dumps({'combined_score': score, 'public': {'val_error_vs_baseline': 0.5},
                                                    'text_feedback': f'seed {seed}'}))
    r = wave_pod.fold('g', w, str(tmp_path), 'runpod-geforce-rtx-4090-c3')
    assert r['correct'] is True and r['fitness'] == pytest.approx(0.65) and r['env'] == 'runpod-geforce-rtx-4090-c3'
    assert [s['seed'] for s in r['per_seed']] == [0, 1] and wave.record_state(r) == 'ok'
    (tmp_path / 'g' / 's1' / 'metrics.json').unlink()  # one seed's training died
    r = wave_pod.fold('g', w, str(tmp_path), 'runpod-geforce-rtx-4090-c3')
    assert r['correct'] is False and r['fitness'] is None and 'seed 1' in r['guardrail']


def test_a_wave_never_overwrites_a_committed_impl(tmp_path, monkeypatch):
    learn = tmp_path / 'learn'
    (learn / 'evolve' / 'chunks' / 'optim').mkdir(parents=True)
    (learn / 'evolve' / 'chunks' / 'optim' / 'x.py').write_text('original\n')
    (tmp_path / 'wave' / 'chunks' / 'optim').mkdir(parents=True)
    (tmp_path / 'wave' / 'chunks' / 'optim' / 'x.py').write_text('different\n')
    monkeypatch.setattr(wave_pod, 'LEARN', str(learn))
    with pytest.raises(SystemExit):
        wave_pod.install_chunks(str(tmp_path / 'wave'), ['optim/x.py'])
    (tmp_path / 'wave' / 'chunks' / 'optim' / 'x.py').write_text('original\n')
    wave_pod.install_chunks(str(tmp_path / 'wave'), ['optim/x.py'])  # the same file is fine


def test_the_estimate_uses_the_probes_throughput_at_the_waves_concurrency(dirs):
    wave.write_wave('w6', {'a': genomes.load(), 'b': genomes.load()}, [0, 1, 2], 'inner', 'full', 3, 3600)
    probe = {'gpu': wave.DEFAULT_GPU, 'concurrency_iters_per_second': {'1': 50, '2': 90, '4': 150, '6': 160}}
    assert wave.throughput(probe, 3) == 90 and wave.throughput(probe, 6) == 160 and wave.throughput(probe, 0) is None
    est = wave.estimate(wave.load_wave('w6'), price=1.0, probe=probe)
    # 6 jobs, 3 at once (measured at 2): 2 full rounds of 3 at 90 it/s
    assert est['hours'] == pytest.approx((wave.OVERHEAD_MINUTES + 6 * wave.FULL_RUN_ITERS / 90 / 60) / 60)
    wave.write_wave('w6b', {'a': genomes.load()}, [0, 1, 2, 3], 'inner', 'full', 3, 3600)
    est = wave.estimate(wave.load_wave('w6b'), price=1.0, probe=probe)
    # 4 jobs, 3 at once: one round of 3 at 90 it/s, then 1 alone at 50 it/s
    assert est['hours'] == pytest.approx((wave.OVERHEAD_MINUTES + 3 * wave.FULL_RUN_ITERS / 90 / 60
                                          + wave.FULL_RUN_ITERS / 50 / 60) / 60)
    assert 'probe' in est['basis']


def test_a_committed_impl_edited_since_refuses_to_ship(dirs):
    path = pathlib.Path(genomes.CHUNKS) / 'optim' / 'baseline_adamw.py'
    original = path.read_text()
    try:
        path.write_text(original + '# edited\n')
        with pytest.raises(SystemExit):
            wave.write_wave('w7', {'a': genomes.load()}, [0], 'inner', 'full', 3, 3600)
    finally:
        path.write_text(original)


def test_grid_ids_carry_the_waves_name():
    out = wave.grid_genomes(genomes.load(), {'optim.batch': [256]}, pairs=False, prefix='bs-1')
    assert list(out) == ['bs-1-batch256']


def test_ingest_refuses_incomplete_results_and_appends_each_genome_once(dirs, monkeypatch):
    wave.write_wave('w8', {'a': genomes.load(), 'b': genomes.load()}, [0], 'inner', 'full', 2, 3600,
                    meta={'a': {'parent': 'gen0-baseline', 'inventor': 'tester'}})
    results = dirs / 'results' / 'w8'
    results.mkdir(parents=True)
    (results / 'a.json').write_text(json.dumps({'correct': True, 'fitness': 0.6, 'env': 'runpod-x-c2'}))
    with pytest.raises(SystemExit):
        wave.ingest('w8')  # b missing
    (results / 'b.json').write_text(json.dumps({'correct': True, 'fitness': 0.5, 'env': 'runpod-x-c2'}))
    calls = []

    class Done:
        returncode, stderr = 0, ''
    monkeypatch.setattr(wave.subprocess, 'run', lambda cmd, **kw: calls.append(cmd) or Done())
    wave.ingest('w8')
    wave.ingest('w8')  # again: nothing appended twice
    assert len(calls) == 2
    a = next(c for c in calls if 'a' == c[c.index('--id') + 1])
    assert a[a.index('--parent') + 1] == 'gen0-baseline' and a[a.index('--env') + 1] == 'runpod-x-c2'


def test_lineage_must_name_the_waves_genomes(dirs):
    with pytest.raises(SystemExit):
        wave.write_wave('w9', {'a': genomes.load()}, [0], 'inner', 'full', 2, 3600, meta={'zz': {'parent': 'x'}})


def test_seeds_pulled_home_fold_after_an_early_stop(dirs):
    wave.write_wave('w10', {'a': genomes.load(), 'b': genomes.load()}, [0], 'inner', 'full', 2, 3600)
    results = dirs / 'results' / 'w10'
    (results / 'a' / 's0').mkdir(parents=True)
    (results / 'a' / 's0' / 'metrics.json').write_text(json.dumps({'combined_score': 0.7, 'public': {}}))
    (results / '.env').write_text(json.dumps({'env': 'runpod-x-c2', 'commit': 'abc'}))
    wave.fold_home('w10')
    record = json.loads((results / 'a.json').read_text())
    assert record['fitness'] == 0.7 and record['commit'] == 'abc' and not (results / 'b.json').exists()


def test_env_tags_keep_the_whole_gpu_name(monkeypatch):
    import types
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
                                                            get_device_name=lambda i: 'NVIDIA A100 80GB PCIe'))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    assert wave_pod.env_tag(3) == 'runpod-a100-80gb-pcie-c3'


def test_a_refused_wave_leaves_nothing_behind(dirs):
    with pytest.raises(SystemExit):
        wave.write_wave('w11', {'a': genomes.load()}, [0], 'inner', 'full', 2, 3600, meta={'zz': {}})
    assert not (dirs / 'waves' / 'w11').exists()


def test_resuming_under_different_code_is_refused(tmp_path, monkeypatch):
    w = tmp_path / 'wave'
    (w / 'genomes').mkdir(parents=True)
    (w / 'wave.json').write_text(json.dumps({'name': 'w12', 'kind': 'score', 'genomes': {}, 'chunks': [], 'seeds': [0],
                                             'split': 'inner', 'mode': 'full', 'concurrency': 1, 'job_timeout_s': 60}))
    monkeypatch.setattr(wave_pod, 'CLOUD', str(tmp_path))
    results = tmp_path / 'results' / 'w12'
    results.mkdir(parents=True)
    (results / '.env').write_text(json.dumps({'env': 'runpod-x-c1', 'commit': 'old'}))
    monkeypatch.setattr(wave_pod, 'gate', lambda cmd, timeout=600: (True, ''))
    monkeypatch.setattr(wave_pod, 'env_tag', lambda c: 'runpod-x-c1')
    monkeypatch.setattr(wave_pod, 'shipped_commit', lambda: 'new')
    monkeypatch.setattr(sys, 'argv', ['wave_pod', str(w)])
    with pytest.raises(SystemExit):
        wave_pod.main()
    assert 'failed' in (results / '.done').read_text()
