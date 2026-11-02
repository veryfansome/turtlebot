"""The evolve campaign, kept outside this repo: a private GitHub repo holds what the search makes.

    uv run python evolve/campaign.py status
    uv run python evolve/campaign.py push      # after a round: this campaign's state to the campaign repo
    uv run python evolve/campaign.py pull      # restore it on another machine (or after a fresh clone)

This repo keeps the contract: evolve.json, EVOLVE.md, the baseline impls (chunks/<axis>/baseline_*.py)
and the harness that scores them. The campaign repo keeps the campaign: every candidate impl
(chunks/<axis>/<impl>.py, append-only), the archive (archive/genomes.jsonl, which names those impls and
holds their content hashes), the round notes (rounds/) and insights.md. They're git-ignored here.

The engine needs a candidate's impl on disk, not committed (it copies the impls a genome names into the
scoring export, and a RunPod wave ships its own copy), so nothing about scoring changes. What this
guards is keeping them: an archived genome is reproducible only while the exact impl it names exists.
So push never deletes, pull never overwrites a differing impl, and the archive may only grow (one copy
must be a prefix of the other).

A local clone of the campaign repo lives in evolve/.campaign/ (git-ignored).
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.join(HERE, '.campaign')
REMOTE = 'git@github.com:veryfansome/turtlebot-evolve.git'
ARCHIVE = os.path.join('archive', 'genomes.jsonl')


def git(*args, check=True):
    r = subprocess.run(['git', '-C', CLONE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f'git {" ".join(args)}: {r.stderr.strip()}')
    return r.stdout


def clone():
    if not os.path.isdir(os.path.join(CLONE, '.git')):
        r = subprocess.run(['git', 'clone', '-q', REMOTE, CLONE], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f'cloning {REMOTE}: {r.stderr.strip()}')
    else:
        git('fetch', '-q', 'origin')
        if git('rev-parse', '--verify', '-q', 'origin/main', check=False).strip():
            git('merge', '-q', '--ff-only', 'origin/main')


def campaign_files(root: str) -> list:
    """Paths (relative to root, an evolve/ folder or the clone) that belong to the campaign."""
    out = []
    chunks = os.path.join(root, 'chunks')
    for axis in sorted(os.listdir(chunks)) if os.path.isdir(chunks) else []:
        for name in sorted(os.listdir(os.path.join(chunks, axis))):
            if name.endswith('.py') and not name.startswith('baseline_') and name != '__init__.py':
                out.append(os.path.join('chunks', axis, name))
    for top in ('archive', 'rounds'):
        base = os.path.join(root, top)
        for dirpath, dirnames, files in os.walk(base):
            if 'private' in dirnames and top == 'archive':
                dirnames.remove('private')  # the engine's private store stays local
            out += [os.path.relpath(os.path.join(dirpath, f), root) for f in sorted(files)]
    if os.path.exists(os.path.join(root, 'insights.md')):
        out.append('insights.md')
    return sorted(out)


def check_archive(src: str, dst: str):
    """The archive may only grow: whichever copy is shorter must be the other's beginning."""
    a, b = os.path.join(src, ARCHIVE), os.path.join(dst, ARCHIVE)
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    with open(a, 'rb') as f:
        ta = f.read()
    with open(b, 'rb') as f:
        tb = f.read()
    if not (ta.startswith(tb) or tb.startswith(ta)):
        raise SystemExit('the archive has diverged between here and the campaign repo (neither is the other plus '
                         'appended records): merge it by hand before syncing')


def copy_new(src: str, dst: str, label: str) -> list:
    """Copy src's campaign files to dst. An impl that exists in both and differs is refused (impls are
    append-only); the archive is taken from whichever copy is longer; other files (round notes, insights)
    are copied over."""
    check_archive(src, dst)
    changed = []
    for rel in campaign_files(src):
        a, b = os.path.join(src, rel), os.path.join(dst, rel)
        if os.path.exists(b) and filecmp.cmp(a, b, shallow=False):
            continue
        if rel.startswith('chunks' + os.sep) and os.path.exists(b):
            raise SystemExit(f'{rel} differs {label}: impls are append-only, so one of them was edited; resolve it by hand')
        if rel == ARCHIVE and os.path.exists(b) and os.path.getsize(b) >= os.path.getsize(a):
            continue  # the other copy already has these records and more
        os.makedirs(os.path.dirname(b) or '.', exist_ok=True)
        shutil.copy2(a, b)
        changed.append(rel)
    return changed


def push():
    clone()
    changed = copy_new(HERE, CLONE, 'between here and the campaign repo')
    if not changed:
        print('campaign repo already has everything here')
        return
    git('add', '-A')
    git('commit', '-q', '-m', f'Sync {time.strftime("%Y-%m-%d %H:%M")}: {len(changed)} files')
    git('push', '-q', 'origin', 'HEAD:main')
    print(f'pushed {len(changed)} files to {REMOTE}')


def pull():
    clone()
    changed = copy_new(CLONE, HERE, 'between the campaign repo and here')
    print(f'pulled {len(changed)} files from {REMOTE}' if changed else 'nothing new in the campaign repo')


def status():
    have = campaign_files(HERE)
    print(f'here: {len(have)} campaign files ({sum(r.startswith("chunks") for r in have)} candidate impls)')
    if not os.path.isdir(os.path.join(CLONE, '.git')):
        print('campaign repo not cloned yet (push or pull clones it)')
        return
    theirs = set(campaign_files(CLONE))
    missing = [r for r in have if r not in theirs or not filecmp.cmp(os.path.join(HERE, r), os.path.join(CLONE, r), shallow=False)]
    print(f'not yet pushed: {len(missing)}' + (f' (e.g. {missing[:5]})' if missing else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cmd', choices=['push', 'pull', 'status'])
    args = parser.parse_args()
    {'push': push, 'pull': pull, 'status': status}[args.cmd]()


if __name__ == '__main__':
    sys.exit(main())
