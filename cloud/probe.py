"""The speed probe: how fast the world model trains on a RunPod pod, before spending on a real wave.

Run on the pod by cloud/wave_pod.py for a wave made with `cloud/wave.py probe NAME`. Measures:
- seconds per training iteration (eval passes included, as in a real run) on the GPU and on the CPU, at
  batch 256, 1024 and 4096: from the difference between a short and a longer run of the same setup, so
  loading the data doesn't count;
- total throughput with 1, 2, 4 and 6 trainings sharing the GPU (batch 1024), which sets a wave's
  concurrency: these models are small, and in the sibling projects the gain from a cloud GPU was
  running several at once, not running one faster.
The Mac's GPU took about 19 ms per step at batch 1024 (2026-11-01), for comparison. A full training run
is about 15,000 iterations (it stops when val stops improving), so minutes_per_full_run is that times
the per-iteration time at the best concurrency, per training. Writes probe.json.
"""
import json
import os
import subprocess
import tempfile
import time

SHORT, LONG = 200, 600
FULL_RUN_ITERS = 15000


def threads() -> str:
    """The pod's real CPU quota (pods see the host's 128 cores but are capped at ~13): torch's default thread
    count would oversubscribe it, and the CPU timings would measure throttling."""
    import jobpool
    return str(max(1, int(jobpool.detect_cpu_cores())))


def timed_run(python: str, iters: int, device: str, batch: int, out: str) -> float:
    t0 = time.time()
    n = threads()
    subprocess.run([python, '-m', 'pet_learn.train_world', '--iters', str(iters), '--patience', str(10 ** 6),
                    '--device', device, '--batch', str(batch), '--out-dir', out, '--steps', os.environ['PET_STEPS']],
                   check=True, capture_output=True, text=True, timeout=1800,
                   env=dict(os.environ, OMP_NUM_THREADS=n, MKL_NUM_THREADS=n,
                            PET_CODE_VERSION=json.dumps({'commit': 'probe', 'dirty': False, 'changes': None})))
    return time.time() - t0


def seconds_per_iter(python: str, device: str, batch: int, out: str) -> float:
    short = timed_run(python, SHORT, device, batch, out)
    long = timed_run(python, LONG, device, batch, out)
    return (long - short) / (LONG - SHORT)


def concurrent_throughput(python: str, k: int, out: str) -> float:
    """Total training iterations per second with k batch-1024 trainings on the GPU at once."""
    env = dict(os.environ, PET_CODE_VERSION=json.dumps({'commit': 'probe', 'dirty': False, 'changes': None}),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    t0 = time.time()
    procs = [subprocess.Popen([python, '-m', 'pet_learn.train_world', '--iters', str(LONG), '--patience', str(10 ** 6),
                               '--device', 'cuda', '--batch', '1024', '--seed', str(i), '--out-dir', out,
                               '--steps', os.environ['PET_STEPS']],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env) for i in range(k)]
    if any(p.wait(timeout=1800) != 0 for p in procs):
        raise RuntimeError(f'a training failed with {k} at once')
    return k * LONG / (time.time() - t0)


def run(results: str, python: str):
    import torch
    out = tempfile.mkdtemp(prefix='probe-models-')
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'per_iter_seconds': {},
              'concurrency_iters_per_second': {}}
    for device in ('cuda', 'cpu'):
        for batch in (256, 1024, 4096):
            s = seconds_per_iter(python, device, batch, out)
            report['per_iter_seconds'][f'{device}-{batch}'] = round(s, 4)
            print(f'{device} batch {batch}: {s * 1000:.1f} ms per iteration', flush=True)
    for k in (1, 2, 4, 6):
        rate = concurrent_throughput(python, k, out)
        report['concurrency_iters_per_second'][str(k)] = round(rate, 1)
        print(f'{k} at once: {rate:.0f} iterations/s in total', flush=True)
    best_k, best_rate = max(report['concurrency_iters_per_second'].items(), key=lambda kv: kv[1])
    report['best_concurrency'] = int(best_k)
    # A wave runs best_k trainings at once; each takes FULL_RUN_ITERS at best_rate / best_k iterations/s.
    report['minutes_per_full_run'] = round(FULL_RUN_ITERS / best_rate / 60 * int(best_k), 1)
    with open(os.path.join(results, 'probe.json'), 'w') as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
