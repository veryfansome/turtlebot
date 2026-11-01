import argparse
import json

import torch

from pet_learn import runs


def test_a_run_record_is_saved_beside_and_inside_the_model(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(['--batch', '1024'])
    record = runs.new_record('world', parser, args)
    steps_file = tmp_path / '20261101-140635.npz'
    steps_file.write_bytes(b'steps')
    runs.add_sessions(record, [str(steps_file)], [argparse.Namespace(inputs=[0] * 2448)])
    record.update(iterations=100, best_iteration=80, results={'test_1s': {'scan': 0.8, 'pose': 0.2, 'bump': 0.9}})
    assert set(record['code']) == {'commit', 'dirty', 'changes'} and record['code']['commit']
    assert record['sessions'][0]['file'] and record['defaults']['batch'] == 256
    out = str(tmp_path / 'world-test.pt')
    runs.save(out, {'model': {'w': torch.zeros(2)}}, record)
    with open(out.replace('.pt', '.json')) as f:
        saved = json.load(f)
    assert saved['model'] == 'world-test.pt' and saved['args']['batch'] == 1024 and 'finished' in saved
    assert torch.load(out, weights_only=False)['record']['best_iteration'] == 80
    line = runs.summary(saved)
    assert 'kept 80 of 100' in line and 'scan 0.80' in line and '1 sessions/2448 steps' in line
    assert 'batch=1024' in line and 'lr=' not in line  # only what differs from the defaults
