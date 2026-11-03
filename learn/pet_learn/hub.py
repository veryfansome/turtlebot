"""The converted sessions on Hugging Face: a private dataset repo RunPod pods pull from, instead of
waiting for an upload from home.

    uv run python -m pet_learn.hub push            # upload new or changed data/steps files; pin the revision
    uv run python -m pet_learn.hub pull --to DIR [--manifest eval/data_manifest.json]   # fetch the pinned revision (on a pod)
    uv run python -m pet_learn.hub status          # what's pinned, and whether data/steps matches it
    uv run python -m pet_learn.hub push-drives     # back up the raw bags (data/drives) to a second private dataset
    uv run python -m pet_learn.hub pull-drives --to DIR [--session S ...]   # restore them

The repo holds data/steps as is: <session>.npz (the steps) and <session>.maps.npz (its SLAM maps).
data_source.json (committed, next to this package) pins the repo and the revision the code was last
pushed with, so a pod pulls exactly that data; eval.preflight still checks every file's fingerprint
against eval/data_manifest.json.

The raw bags (data/drives/<session>/: the .mcap and its metadata.yaml) are only a backup, in their own
private dataset (DRIVES_REPO), so the steps repo pods download stays small and its pin isn't touched.
They are what a reconversion (convert --force) or a signal that isn't converted yet needs; nothing pins them.

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
DRIVES = os.path.normpath(os.path.join(LEARN, '..', 'data', 'drives'))
DRIVES_REPO = 'veryfansome/turtlebot-drives'
BAG_FILES = ['*/*.mcap', '*/metadata.yaml']  # one folder per session, as ros2 bag record writes it

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


DRIVES_CARD = """---
license: other
pretty_name: "TurtleBot 4 robot pet, raw driving recordings"
---

# TurtleBot 4 robot pet: raw driving recordings

The ROS 2 bags (MCAP) the converted sessions in `veryfansome/turtlebot-steps` are made from, one folder
per session as `ros2 bag record` wrote it. A backup: kept so the sessions can be reconverted, or new
signals pulled out of them. Restore with `python -m pet_learn.hub pull-drives --to DIR`.
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


def private_repo(repo: str):
    """The API, with repo created private if it's new; refuses a public one."""
    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type='dataset').private:
        raise SystemExit(f'{repo} exists and is public: refusing to upload recordings to it')
    return api


def push(folder: str, repo: str):
    api = private_repo(repo)
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


def bag_sessions(folder: str) -> list:
    """The session folders under data/drives that hold a bag (one or more .mcap files). A bag whose recording
    wasn't stopped cleanly has no metadata.yaml (ros2 bag reindex rebuilds it); it's still a recording."""
    return sorted({os.path.basename(os.path.dirname(m)) for m in glob.glob(os.path.join(folder, '*', '*.mcap'))})


def bag_patterns(sessions: list) -> list:
    return [f'{s}/{p.split("/")[1]}' for s in sessions for p in BAG_FILES]


def push_drives(folder: str, repo: str):
    """Uploads every session's bag; files already there unchanged aren't sent again."""
    sessions = bag_sessions(folder)
    if not sessions:
        raise SystemExit(f'no bags (a folder with an .mcap) in {folder}')
    unfinished = [s for s in sessions if not os.path.exists(os.path.join(folder, s, 'metadata.yaml'))]
    if unfinished:
        print(f'note: no metadata.yaml (the recording wasn\'t stopped cleanly; ros2 bag reindex rebuilds it), '
              f'backing up the .mcap anyway: {", ".join(unfinished)}')
    api = private_repo(repo)
    api.upload_file(path_or_fileobj=DRIVES_CARD.encode(), path_in_repo='README.md', repo_id=repo,
                    repo_type='dataset', commit_message='Dataset card')
    commit = api.upload_folder(folder_path=folder, repo_id=repo, repo_type='dataset',
                               allow_patterns=bag_patterns(sessions), commit_message=f'Bags: {len(sessions)} sessions')
    print(f'pushed the bags of {len(sessions)} sessions to {repo} (private): {", ".join(sessions)}')
    print(f'commit: {getattr(commit, "commit_url", commit)}')


def bag_files(repo_files: list, sessions: list = None) -> list:
    """The repo's bag files to fetch: all, or only these sessions'."""
    import fnmatch
    patterns = bag_patterns(sessions) if sessions else BAG_FILES
    return sorted(f for f in repo_files if any(fnmatch.fnmatchcase(f, p) for p in patterns))


def pull_drives(to: str, repo: str, sessions: list = None):
    """Fetches the bags and checks every file the repo holds for them arrived."""
    from huggingface_hub import HfApi, snapshot_download
    files = bag_files(HfApi(token=token()).list_repo_files(repo, repo_type='dataset'), sessions)
    absent = sorted(set(sessions or []) - {f.split('/')[0] for f in files})
    if absent or not files:
        raise SystemExit(f'{repo} has no bag for {absent or "any session"}')
    snapshot_download(repo, repo_type='dataset', local_dir=to, allow_patterns=files, token=token())
    missing = [f for f in files if not os.path.exists(os.path.join(to, f))]
    if missing:
        raise SystemExit(f'pulled {repo} but missing {missing}')
    print(f'pulled {len(files)} files, the bags of {len({f.split("/")[0] for f in files})} sessions, from {repo} into {to}')


def wanted(source: dict, manifest: str = None) -> list:
    """The files to fetch: everything pinned, or with a manifest (eval/data_manifest.json) only its sessions and
    their maps. The evolve search measures exactly the manifest's sessions, while the pin grows with every
    recording, so a pod scoring the search must fetch only those, or its preflight refuses the folder."""
    if manifest is None:
        return list(source['files'])
    with open(manifest) as f:
        sessions = json.load(f)['sessions']
    files = sorted(set(sessions) | {name.replace('.npz', '.maps.npz') for name in sessions})
    absent = sorted(set(files) - set(source['files']))
    if absent:
        raise SystemExit(f'the pinned revision lacks {absent}: the manifest names sessions that were never pushed')
    return files


def pull(to: str, revision: str = None, manifest: str = None):
    from huggingface_hub import snapshot_download
    source = pinned()
    if not source:
        raise SystemExit(f'nothing pinned in {SOURCE}: push first')
    revision = revision or source['revision']
    files = wanted(source, manifest)
    snapshot_download(source['repo'], repo_type='dataset', revision=revision, local_dir=to,
                      allow_patterns=files, token=token())
    got = sorted(os.path.basename(p) for p in glob.glob(os.path.join(to, '*.npz')))
    missing = sorted(set(files) - set(got))
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
    p.add_argument('--manifest', help="only this data manifest's sessions and their maps (eval/data_manifest.json)")
    p = sub.add_parser('status')
    p.add_argument('--steps', default=STEPS)
    p = sub.add_parser('push-drives')
    p.add_argument('--drives', default=DRIVES)
    p.add_argument('--repo', default=DRIVES_REPO)
    p = sub.add_parser('pull-drives')
    p.add_argument('--to', required=True)
    p.add_argument('--repo', default=DRIVES_REPO)
    p.add_argument('--session', action='append', help='only this session (repeatable); default: all')
    args = parser.parse_args()
    if args.cmd == 'push-drives':
        push_drives(args.drives, args.repo)
    elif args.cmd == 'pull-drives':
        pull_drives(args.to, args.repo, args.session)
    elif args.cmd == 'push':
        push(args.steps, args.repo)
    elif args.cmd == 'pull':
        pull(args.to, args.revision, args.manifest)
    else:
        status(args.steps)


if __name__ == '__main__':
    main()
