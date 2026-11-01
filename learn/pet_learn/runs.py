"""Run records: how each saved model was made and how it did, kept next to it.

    uv run python -m pet_learn.runs [--kind world|driver] [--last N]

Every model in data/models/ gets <name>.json beside it (and the same record inside the .pt): the
arguments and their defaults, the code (git commit, and a fingerprint of uncommitted changes to the
training code: learn/pet_learn and its dependencies), the device, the sessions it saw (each with a
fingerprint of its steps file, so reconverted data shows), when it stopped and why, and its test
results. Comparing two runs means comparing their records: a difference in data, code or settings
shows up there rather than having to be guessed from the results. This command lists them, oldest
first, with every setting that differs from its default.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
MODELS = os.path.join(REPO, 'data', 'models')
CODE = ['learn/pet_learn', 'learn/pyproject.toml', 'learn/uv.lock']  # what a run's results depend on


def code_version() -> dict:
    """The git commit, and whether the training code (CODE) had uncommitted changes, with a fingerprint
    of them, so two runs on the same uncommitted code can be told apart from runs on different code."""
    def git(*cmd):
        return subprocess.run(['git', '-C', REPO, *cmd], capture_output=True, text=True, check=False).stdout
    diff = git('diff', 'HEAD', '--', *CODE)
    untracked = git('ls-files', '--others', '--exclude-standard', '--', *CODE)
    for path in untracked.split():
        with open(os.path.join(REPO, path), 'rb') as f:
            diff += path + f.read().decode(errors='replace')
    return {'commit': git('rev-parse', '--short', 'HEAD').strip(), 'dirty': bool(diff),
            'changes': hashlib.sha1(diff.encode()).hexdigest()[:10] if diff else None}


def new_record(kind: str, parser, args) -> dict:
    """A record to fill in as the run goes; made when the run starts, so it has the code that runs."""
    return {
        'kind': kind,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'code': code_version(),
        'args': dict(vars(args)),
        'defaults': {a.dest: a.default for a in parser._actions if a.dest != 'help'},
    }


def add_sessions(record: dict, paths: list, sessions: list):
    """The sessions a run saw: name, steps, and a fingerprint of the steps file (changes on reconversion)."""
    def fingerprint(path):
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    record['sessions'] = [{'name': os.path.basename(p)[:-4], 'steps': int(len(s.inputs)), 'file': fingerprint(p)}
                          for p, s in zip(paths, sessions)]


def save(out: str, checkpoint: dict, record: dict):
    """The checkpoint (with its record inside) at out, and the record as JSON beside it."""
    import torch
    record = dict(record, finished=time.strftime('%Y-%m-%d %H:%M:%S'), model=os.path.basename(out))
    torch.save(dict(checkpoint, record=record), out)
    with open(out.replace('.pt', '.json'), 'w') as f:
        json.dump(record, f, indent=1)


def load_records(kind=None) -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(MODELS, '*.json'))):
        with open(path) as f:
            r = json.load(f)
        if kind is None or r.get('kind') == kind:
            records.append(r)
    return sorted(records, key=lambda r: r['started'])


def summary(r: dict) -> str:
    """One line per run: what it was trained on and with, and its headline results."""
    a, code, res = r['args'], r['code'], r.get('results', {})
    sessions = r.get('sessions', [])
    data = f"{len(sessions)} sessions/{sum(s['steps'] for s in sessions)} steps"
    files = ''.join(s.get('file', '') for s in sessions)
    data += ' data ' + (hashlib.sha1(files.encode()).hexdigest()[:6] if files else '?')  # ?: recorded before fingerprints
    version = code['commit'] + (f"+{code['changes']}" if code['dirty'] else '')
    defaults = r.get('defaults', {})
    changed = ' '.join(f'{k}={v}' for k, v in a.items() if k not in ('steps', 'world') and k in defaults and v != defaults[k])
    head = f"{r['model']:32s} {version:18s} {data:34s} {a.get('device', '?'):4s} {changed or 'defaults'}"
    if r['kind'] == 'world':
        at1 = res.get('test_1s', {})
        scores = ' '.join(f"{k} {at1[k]:.2f}" for k in ('scan', 'odom', 'ir', 'pose', 'bump') if k in at1)
        stop = f" ({r['stopped']})" if r.get('stopped') else ''
        return f"{head}  kept {r.get('best_iteration')} of {r.get('iterations')}{stop}  test@1s: {scores}"
    probe = res.get('policy', {})
    turns = ' '.join(f"{k} {probe[k][1]:+.2f}" for k in ('left', 'back-left') if k in probe)
    return (f"{head}  on {os.path.basename(a['world'])}  "
            f"values {res.get('value_rank_correlation', float('nan')):.2f}  turn: {turns}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', choices=['world', 'driver'])
    parser.add_argument('--last', type=int, help='show only the most recent this many (default: all)')
    args = parser.parse_args()
    records = load_records(args.kind)
    if not records:
        print(f'no run records in {MODELS} (models saved before records existed have none)')
    shown = records[-args.last:] if args.last else records
    if len(shown) < len(records):
        print(f'({len(records) - len(shown)} older records not shown)')
    for r in shown:
        print(summary(r))


if __name__ == '__main__':
    main()
