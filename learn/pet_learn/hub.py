"""The converted sessions on Hugging Face: a private dataset repo RunPod pods pull from, instead of
waiting for an upload from home.

    uv run python -m pet_learn.hub push            # upload new or changed data/steps files; pin the revision
    uv run python -m pet_learn.hub pull --to DIR   # fetch the pinned revision (on a pod)
    uv run python -m pet_learn.hub status          # what's pinned, and whether data/steps matches it

The repo holds data/steps as is: <session>.npz (the steps) and <session>.maps.npz (its SLAM maps).
data_source.json (committed, next to this package) pins the repo and the revision the code was last
pushed with, so a pod pulls exactly that data; eval.preflight still checks every file's fingerprint
against eval/data_manifest.json. The raw bags (data/drives) stay home.

The token: HF_TOKEN in the environment, or in $RUNPOD_ENV_FILE (default ~/.runpod.env), where the
RunPod tooling keeps it. It's never printed. A pod needs only a read token.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.dirname(HERE)
SOURCE = os.path.join(LEARN, 'data_source.json')
STEPS = os.path.normpath(os.path.join(LEARN, '..', 'data', 'steps'))
REPO = 'veryfansome/turtlebot-steps'

CARD = """---
license: other
pretty_name: "TurtleBot 4 robot pet, converted driving sessions"
---

# TurtleBot 4 robot pet: converted driving sessions

Private training data for a robot pet's learned driver (a TurtleBot 4 driven around a home with
its game controller). Each recording is two files, made by `pet_learn.convert`:

- `<session>.npz`: one row per 0.1 s step: lidar sectors, odometry, floor sensor, wheel speeds and
  currents, IR, hazards, the drive command, SLAM pose and more (`STEP_FIELDS` in convert.py).
- `<session>.maps.npz`: every SLAM map of the session.

Pinned by revision from the project (`learn/data_source.json`); pull with `python -m pet_learn.hub pull`.
"""


def token() -> str:
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    env_file = os.path.expanduser(os.environ.get('RUNPOD_ENV_FILE', '~/.runpod.env'))
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    line = line[len('export '):]
                if line.startswith('HF_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"\'')
    raise SystemExit('no Hugging Face token: set HF_TOKEN, or put HF_TOKEN=... in ~/.runpod.env')


def pinned() -> dict:
    if not os.path.exists(SOURCE):
        return {}
    with open(SOURCE) as f:
        return json.load(f)


def push(folder: str, repo: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type='dataset').private:
        raise SystemExit(f'{repo} exists and is public: refusing to upload recordings to it')
    files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not files:
        raise SystemExit(f'no .npz files in {folder}')
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo='README.md', repo_id=repo, repo_type='dataset',
                    commit_message='Dataset card')
    commit = api.upload_folder(folder_path=folder, repo_id=repo, repo_type='dataset', allow_patterns=['*.npz'],
                               commit_message=f'Sessions: {len(files) // 2} recordings ({len(files)} files)')
    revision = api.repo_info(repo, repo_type='dataset').sha
    with open(SOURCE, 'w') as f:
        json.dump({'repo': repo, 'revision': revision, 'files': [os.path.basename(p) for p in files]}, f, indent=1)
        f.write('\n')
    print(f'pushed {len(files)} files to {repo} (private) at revision {revision}; pinned in {SOURCE}')
    print(f'commit: {getattr(commit, "commit_url", commit)}')


def pull(to: str, revision: str = None):
    from huggingface_hub import snapshot_download
    source = pinned()
    if not source:
        raise SystemExit(f'nothing pinned in {SOURCE}: push first')
    revision = revision or source['revision']
    snapshot_download(source['repo'], repo_type='dataset', revision=revision, local_dir=to,
                      allow_patterns=['*.npz'], token=token())
    got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(to, '*.npz')))
    missing = sorted(set(source['files']) - set(got))
    if missing:
        raise SystemExit(f'pulled {source["repo"]}@{revision[:10]} but missing {missing}')
    print(f'pulled {len(got)} files from {source["repo"]}@{revision[:10]} into {to}')


def status(folder: str):
    source = pinned()
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, '*.npz')))
    if not source:
        print(f'nothing pinned; {folder} has {len(have)} files')
        return
    print(f'pinned: {source["repo"]}@{source["revision"][:10]}, {len(source["files"])} files')
    new = sorted(set(have) - set(source['files']))
    print(f'{folder}: {len(have)} files' + (f'; not pushed yet: {new}' if new else '; all pushed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('push')
    p.add_argument('--steps', default=STEPS)
    p.add_argument('--repo', default=pinned().get('repo', REPO))
    p = sub.add_parser('pull')
    p.add_argument('--to', required=True)
    p.add_argument('--revision', help='default: the pinned one')
    p = sub.add_parser('status')
    p.add_argument('--steps', default=STEPS)
    args = parser.parse_args()
    if args.cmd == 'push':
        push(args.steps, args.repo)
    elif args.cmd == 'pull':
        pull(args.to, args.revision)
    else:
        status(args.steps)


if __name__ == '__main__':
    main()
