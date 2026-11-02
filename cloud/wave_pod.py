"""Runs a wave on the RunPod pod (cloud/runpod.sh wave starts it under tmux, from learn/, with learn's
interpreter as PET_PYTHON and the pinned sessions in PET_STEPS).

    python ../cloud/wave_pod.py ../cloud/waves/NAME

Scores every genome the way the evolve search does, since `evolve ingest` will take the numbers without
running the engine's own gates:
1. the wave's new impls go into learn/evolve/chunks (never over a committed one);
2. eval.preflight once: the data must be the pinned data, and the GPU must train. If not, nothing is
   scored: that's the environment, not the candidates;
3. per genome, eval.smoke and eval.guard_leakage: a genome failing either gets a failed record;
4. per passing genome and seed, eval.adapter, through the job pool (concurrency trainings share the GPU,
   threads sized to the pod's CPU quota);
5. each genome's seeds fold into cloud/results/NAME/<id>.json as soon as its last seed ends (so a run
   stopped early still brings home every finished genome): fitness (the mean combined_score), correct,
   per_seed, public, text_feedback, env (runpod-<gpu>-c<concurrency>: scores compare only within one
   environment), and commit (the code that ran: cloud/.commit, written by bootstrap).
Last, cloud/results/NAME/.done gets TB_RUN_ID, so the lane polling for it can tell this run's marker from
an earlier one's. A probe wave (kind "probe") runs cloud/probe.py instead.
"""
import filecmp
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

CLOUD = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(os.path.dirname(CLOUD), 'learn')
sys.path.insert(0, CLOUD)


def env_tag(concurrency: int) -> str:
    """runpod-<the GPU's full name, slugged>-c<concurrency>, e.g. runpod-geforce-rtx-4090-c3: the whole
    name, since 'A100 80GB PCIe' and 'H100 PCIe' would share a last word."""
    import torch
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower().replace('nvidia', '')).strip('-')
    return f'runpod-{slug}-c{concurrency}'


def shipped_commit() -> str:
    path = os.path.join(CLOUD, '.commit')
    return open(path).read().strip() if os.path.exists(path) else None


def gate(cmd: list, timeout: int = 600):
    """(ok, reason) for a gate command run from learn/."""
    try:
        r = subprocess.run(cmd, cwd=LEARN, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f'{" ".join(cmd[2:])} timed out after {timeout}s'
    return r.returncode == 0, (r.stderr or r.stdout).strip()[-600:]


def install_chunks(wave_dir: str, chunks: list):
    for rel in chunks:
        src = os.path.join(wave_dir, 'chunks', rel)
        dst = os.path.join(LEARN, 'evolve', 'chunks', rel)
        if os.path.exists(dst) and not filecmp.cmp(src, dst, shallow=False):
            raise SystemExit(f'{rel} differs from the committed impl of that name: impls are append-only')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)


def fold(gid: str, wave: dict, results: str, env: str, commit: str = None) -> dict:
    per_seed, failures = [], []
    for seed in wave['seeds']:
        mfile = os.path.join(results, gid, f's{seed}', 'metrics.json')
        if not os.path.exists(mfile):
            log = os.path.join(results, '_pool', f'{gid}__s{seed}.log')
            tail = open(log).read().strip()[-400:] if os.path.exists(log) else 'no log'
            failures.append(f'seed {seed}: {tail}')
            continue
        with open(mfile) as f:
            m = json.load(f)
        per_seed.append({'seed': seed, 'combined_score': m['combined_score'], 'public': m.get('public', {}),
                         'text_feedback': m.get('text_feedback')})
    record = {'id': gid, 'wave': wave['name'], 'env': env, 'commit': commit, 'per_seed': per_seed}
    if failures:
        record.update(fitness=None, correct=False, guardrail='; '.join(failures))
    else:
        scores = [s['combined_score'] for s in per_seed]
        record.update(fitness=statistics.fmean(scores), combined_score=statistics.fmean(scores), correct=True,
                      public={'seed_scores': scores, **per_seed[0]['public']},
                      text_feedback=per_seed[0]['text_feedback'])
    return record


def write_record(results: str, record: dict):
    with open(os.path.join(results, f'{record["id"]}.json'), 'w') as f:
        json.dump(record, f, indent=1)
    print(f'{record["id"]}: fitness {record["fitness"]}', flush=True)


def done(results: str, status: str):
    with open(os.path.join(results, '.done'), 'w') as f:
        f.write(f"{os.environ.get('TB_RUN_ID', '')} {status}\n")


def main():
    wave_dir = os.path.abspath(sys.argv[1])
    with open(os.path.join(wave_dir, 'wave.json')) as f:
        wave = json.load(f)
    results = os.path.join(CLOUD, 'results', wave['name'])
    os.makedirs(results, exist_ok=True)
    python = os.environ.get('PET_PYTHON', sys.executable)
    try:
        if wave['kind'] == 'probe':
            import probe
            probe.run(results, python)
            done(results, 'ok')
            return
        install_chunks(wave_dir, wave['chunks'])
        ok, why = gate([python, '-m', 'eval.preflight'])
        if not ok:
            raise SystemExit(f'preflight failed: {why}')
        env, commit = env_tag(wave['concurrency']), shipped_commit()
        previous = os.path.join(results, '.env')
        if os.path.exists(previous):  # resuming: finished seeds would be labelled with code that didn't make them
            with open(previous) as f:
                before = json.load(f)
            if before.get('commit') != commit or before.get('env') != env:
                raise SystemExit(f'results/{wave["name"]} were made by {before} but this pod now runs {commit} as {env}: '
                                 'make a new wave rather than mix them')
        with open(os.path.join(results, '.env'), 'w') as f:  # for folding at home after an early stop (wave.py fold)
            json.dump({'env': env, 'commit': commit}, f)
        jobs, pending = [], {}
        for gid, rel in wave['genomes'].items():
            genome = os.path.join(wave_dir, rel)
            failed = None
            for name in ('eval.smoke', 'eval.guard_leakage'):
                ok, why = gate([python, '-m', name, genome])
                if not ok:
                    failed = f'{name}: {why}'
                    break
            if failed:
                write_record(results, {'id': gid, 'wave': wave['name'], 'env': env, 'commit': commit, 'fitness': None,
                                       'correct': False, 'guardrail': failed, 'per_seed': []})
                continue
            pending[gid] = set(wave['seeds'])
            for seed in wave['seeds']:
                out = os.path.join(results, gid, f's{seed}')
                jobs.append({'id': f'{gid}__s{seed}', 'resource': 'gpu', 'timeout': wave['job_timeout_s'],
                             'cmd': f"'{python}' -m eval.adapter '{out}' '{genome}' {seed} {wave['split']} {wave['mode']}"})

        finished = re.compile(r'^\[jobpool\] (?:resume: )?(\S+?)__s(\d+)(?::| already) (?:done|failed|skipped)')

        def log(message: str):
            # A genome's record is written as soon as its last seed ends.
            print(message, flush=True)
            m = finished.match(message)
            if m and m.group(1) in pending:
                pending[m.group(1)].discard(int(m.group(2)))
                if not pending[m.group(1)]:
                    write_record(results, fold(m.group(1), wave, results, env, commit))
                    del pending[m.group(1)]
        if jobs:
            import jobpool
            jobpool.run_pool(jobs, os.path.join(results, '_pool'), gpu_slots=wave['concurrency'], cpu_slots=1,
                             job_timeout=wave['job_timeout_s'], log=log)
        for gid in list(pending):  # anything the log didn't account for
            write_record(results, fold(gid, wave, results, env, commit))
        done(results, 'ok')
    except BaseException as e:
        done(results, f'failed: {e}')
        raise


if __name__ == '__main__':
    main()
