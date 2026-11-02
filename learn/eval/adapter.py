"""The world-model search's scored command (the evolve plugin's eval contract).

    python -m eval.adapter RESULTS_DIR GENOME SEED SPLIT MODE

Trains the world model the genome describes on the pinned sessions (PET_STEPS, checked by
eval.preflight) with the given seed, exactly as pet_learn.train_world always does (until the val error
stops improving), and writes RESULTS_DIR/metrics.json:

- combined_score: minus the log of the model's error relative to the naive "nothing changes" baselines,
  a geometric mean over heads and horizons (train_world.score), so 0 is level with the baselines and
  higher is better; on the inner split's val windows (the slice of each session that
  also picks the training iteration to keep); on the final split, on every window of the sessions in
  PET_FINAL_STEPS (recorded after the search began, never optimized against). Higher is better.
- public: what inventors may see: the val and test scores, the test error per head at 1 s, and how
  long it trained.
- private: per seed, the full per-head, per-horizon test breakdown.
- text_feedback: one sentence on where the error is.

MODE (proxy or full) doesn't change the run: shortening training was found to reverse rankings in the
sibling projects' searches, so every tier trains to the same stopping rule.
"""
import glob
import hashlib
import math
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def code_version(genome_path: str) -> dict:
    """Which code made the model, for its run record: there's no git in a scoring copy, so a
    fingerprint of the harness and the genome's impls stands in for the commit."""
    from pet_learn import genome as genomes
    genome = genomes.load(genome_path)
    h = hashlib.sha1(json.dumps(genome, sort_keys=True).encode())
    files = sorted(glob.glob(os.path.join(ROOT, 'pet_learn', '*.py')))
    files += [os.path.join(genomes.CHUNKS, axis, spec['impl'] + '.py') for axis, spec in sorted(genome['chunks'].items())]
    for path in files:
        with open(path, 'rb') as f:
            h.update(os.path.relpath(path, ROOT).encode() + f.read())
    return {'commit': 'evolve-export', 'dirty': True, 'changes': h.hexdigest()[:10]}


def main():
    results_dir, genome_path, seed, split, mode = sys.argv[1:6]
    os.makedirs(results_dir, exist_ok=True)
    out_dir = os.path.join(results_dir, 'model')
    cmd = [sys.executable, '-m', 'pet_learn.train_world', '--genome', genome_path, '--seed', str(seed),
           '--steps', os.environ['PET_STEPS'], '--out-dir', out_dir]
    if split == 'final':
        final = os.environ.get('PET_FINAL_STEPS')
        if not final:
            sys.exit('the final split needs PET_FINAL_STEPS: sessions recorded after the search began')
        cmd += ['--final-steps', final]
    elif split != 'inner':
        sys.exit(f'unknown split {split!r}')
    env = dict(os.environ, PET_CODE_VERSION=json.dumps(code_version(genome_path)))
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if run.returncode != 0:
        sys.stderr.write(run.stdout[-2000:] + run.stderr[-4000:])
        sys.exit(f'training failed (exit {run.returncode})')
    [record_path] = glob.glob(os.path.join(out_dir, 'world-*.json'))
    with open(record_path) as f:
        record = json.load(f)
    score = record['final_score'] if split == 'final' else record['val_score']  # a ratio: lower is better
    at_1s = record['results']['test_1s']
    continuous = {k: v for k, v in at_1s.items() if k not in ('bump', 'slip')}
    worst, best = max(continuous, key=continuous.get), min(continuous, key=continuous.get)
    metrics = {
        'combined_score': -math.log(score),
        'correct': True,
        'public': {
            'val_error_vs_baseline': record['val_score'],
            'test_error_vs_baseline': record['test_score'],
            'test_1s': at_1s,
            'iterations': record['iterations'],
            'best_iteration': record['best_iteration'],
            'seconds': record['seconds'],
        },
        'private': {f'seed{seed}': {'results': record['results'], 'final_score': record.get('final_score')}},
        'text_feedback': (f"Error {score:.3f} of the naive baselines' (lower is better); at 1 s ahead the hardest head "
                          f"is {worst} ({continuous[worst]:.2f}) and the easiest {best} ({continuous[best]:.2f}); "
                          f"kept iteration {record['best_iteration']} of {record['iterations']}."),
    }
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({'combined_score': metrics['combined_score']}))


if __name__ == '__main__':
    main()
