"""jobpool.py — a generic DAG job runner with resource pools + mutex groups, for fanning out independent
GPU/CPU jobs on ONE pod so variants scale with pod SIZE (cores/GPUs) instead of wall-clock.

Motivation: in the encoder study the encoder TRAINING is GPU-bound (a full 109-149M encoder) but the SCORING
is light — a tiny world-model trained on FROZEN precomputed embeddings, bottlenecked on CPU (the retrieval eval
+ baseline arms), so the GPU sits mostly idle. Running the scores sequentially wastes that idle capacity. This
runner fans them out: many CPU-bound scores share one idle GPU + many cores, while GPU-bound builds are capped
separately.

A job is a shell command plus metadata:
  - `resource`: 'gpu' | 'cpu'  — which pool it draws from (capped independently).
  - `deps`:     job ids that must SUCCEED before it starts (a failed/skipped dep skips this job).
  - `group`:    optional mutex key — at most ONE running job per group. Used for the shared base_cache: two
                `cli score`s on the SAME (root, split) share the objective-independent baseline-cache entry and
                would race its (non-atomic) write, so they share a group and serialize; DIFFERENT groups run
                concurrently.
  - `timeout`:  optional per-job seconds; a job running longer is killed + marked failed (no infinite hang).

Guarantees: deps honored; gpu/cpu caps never exceeded; no two same-group jobs run at once; per-job failure is
ISOLATED (dependents skipped, the rest proceed); RESUME is gated on a `.done` sentinel written only on exit
code 0 — so it exactly matches the live success criterion (a job that printed JSON then failed is NOT skipped;
a build job that emits no JSON but exits 0 IS skipped). On any abnormal exit (exception / SIGTERM / SIGINT) all
running children are terminated and handles closed, so a resumed run never races orphaned children. Malformed
specs (cycle, dup id, bad dep, a resource with jobs but ≤0 slots) are rejected up front.

  python cloud/jobpool.py --spec jobs.json --out OUT --gpu-slots 3 --cpu-slots 1 [--job-timeout S]

Copied from jepa/terminal-jepa/evolve/jobpool.py (its tests came with it: cloud/tests/test_jobpool.py), with one
change: each job runs in its own process group, and a timeout or cleanup kills the whole group. A turtlebot job is
`sh -c` running eval.adapter, which runs the training as its own child, so killing only the direct child left the
training running on the GPU after its job was marked failed.
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import time


def detect_cpu_cores():
    """The pod's ACTUAL allocated cores — the cgroup CFS quota, NOT os.cpu_count()/nproc. A RunPod GPU pod SEES
    the host's 128 physical cores but is quota-capped (e.g. cfs_quota_us=1360000 / period 100000 = 13.6 cores);
    sizing threads to nproc oversubscribes ~9× → 93%+ CFS throttling. Falls back to os.cpu_count()."""
    try:                                                   # cgroup v2: "<max|N> <period>"
        f = pathlib.Path("/sys/fs/cgroup/cpu.max")
        if f.exists():
            q, p = f.read_text().split()
            if q != "max":
                return max(1.0, int(q) / int(p))
    except Exception:
        pass
    for d in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):   # cgroup v1
        try:
            quota = int(pathlib.Path(f"{d}/cpu.cfs_quota_us").read_text())
            if quota > 0:
                period = int(pathlib.Path(f"{d}/cpu.cfs_period_us").read_text())
                return max(1.0, quota / period)
        except Exception:
            pass
    return float(os.cpu_count() or 1)


def detect_gpu_mem_mb():
    """Total GPU memory (MB) via nvidia-smi, or None if unavailable (→ no gpu-mem constraint). Used as the
    concurrent-GPU-memory budget so a 20GB build + many GPU-light scores can't oversubscribe VRAM → CUDA OOM."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _kill_group(proc):
    """Kill a job and everything it started (its process group: jobs start in a new session)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass


def _validate(jobs, caps):
    """Structural checks BEFORE anything runs: unique ids, known resource, deps exist, no dependency cycle, and
    every resource that HAS a job must have ≥1 slot (else those jobs can never launch → the loop would hang)."""
    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate job ids: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    by_id = {j["id"]: j for j in jobs}
    for j in jobs:
        if j.get("resource") not in ("gpu", "cpu"):
            raise ValueError(f"job {j['id']}: resource must be 'gpu' or 'cpu', got {j.get('resource')!r}")
        for d in j.get("deps", []):
            if d not in by_id:
                raise ValueError(f"job {j['id']}: dep {d!r} is not a known job id")
    for r in {j["resource"] for j in jobs}:                # M1: a used resource with ≤0 slots would deadlock
        if caps.get(r, 0) < 1:
            n = sum(1 for j in jobs if j["resource"] == r)
            raise ValueError(f"{caps.get(r, 0)} slots for resource {r!r} but {n} job(s) need it — would hang")
    WHITE, GREY, BLACK = 0, 1, 2                            # cycle detection (DFS)
    color = {i: WHITE for i in ids}

    def visit(i, stack):
        color[i] = GREY
        for d in by_id[i].get("deps", []):
            if color[d] == GREY:
                raise ValueError(f"dependency cycle: {' -> '.join(stack + [i, d])}")
            if color[d] == WHITE:
                visit(d, stack + [i])
        color[i] = BLACK

    for i in ids:
        if color[i] == WHITE:
            visit(i, [])


def run_pool(jobs, out_dir, gpu_slots=1, cpu_slots=4, poll=2.0, resume=True, job_timeout=None,
             threads_per_job=None, cpu_cores=None, gpu_mem_mb=None, log=print):
    """Run the job DAG. Returns {job_id: status} where status ∈ {'done','failed','skipped'}. Never raises on a
    job failure (isolated); raises only on a malformed spec. `job_timeout` (seconds) is the default per-job cap;
    a job may override with its own `timeout`. Success is defined by exit code 0 and recorded as a `<id>.done`
    sentinel (the RESUME key). Each job's BLAS/OMP thread count is CAPPED (see below; a job may override with
    `threads`) so peak concurrency fits the pod's real (cgroup) CPU quota — else fanned-out scores oversubscribe
    the cap and the CFS scheduler throttles them (measured 93% throttled with default (nproc-many) threading)."""
    caps = {"gpu": gpu_slots, "cpu": cpu_slots}
    _validate(jobs, caps)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cores = cpu_cores if cpu_cores else detect_cpu_cores()
    # budget for PEAK concurrency = gpu_slots + cpu_slots (a gpu build can overlap cpu scores; they share cores)
    n_threads = int(threads_per_job) if threads_per_job else max(1, int(cores // (gpu_slots + cpu_slots)))
    _THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS")
    log(f"[jobpool] cpu budget {cores:.1f} cores (cgroup quota) / {gpu_slots}+{cpu_slots} slots "
        f"→ {n_threads} thread(s)/job")

    # GPU-MEMORY budget: a continuous resource capping concurrent GPU memory (each job's `gpu_mem` MB), so a big
    # build (~20GB) + many GPU-light scores (~1GB each on the same card) can't oversubscribe VRAM → CUDA OOM.
    # It is a RESOURCE constraint, not a dependency, so it never couples failures across independent jobs.
    gpu_budget = gpu_mem_mb
    if gpu_budget is None and any(j.get("gpu_mem") for j in jobs):
        det = detect_gpu_mem_mb()
        gpu_budget = int(det * 0.92) if det else None       # 8% headroom (fragmentation / non-PyTorch use)
    if gpu_budget:
        toobig = [j["id"] for j in jobs if j.get("gpu_mem", 0) > gpu_budget]
        if toobig:                                          # a job bigger than the whole budget could never launch
            raise ValueError(f"jobs {toobig} need gpu_mem > budget {gpu_budget}MB — would hang")
        log(f"[jobpool] gpu-memory budget {gpu_budget} MB (concurrent gpu_mem ≤ this)")

    def done_marker(jid):
        return out / f"{jid}.done"

    status = {}                 # id -> 'done'|'failed'|'skipped' (terminal states only)
    running = {}                # id -> {proc, resource, group, start, timeout, gpu_mem, logf, outf}
    used = {"gpu": 0, "cpu": 0, "gpu_mem": 0}
    busy_groups = set()

    if resume:                  # pre-mark jobs whose success-sentinel exists (exit-code-0 in a prior run)
        for j in jobs:
            if done_marker(j["id"]).exists():
                status[j["id"]] = "done"
                log(f"[jobpool] resume: {j['id']} already done — skipping")
    else:                       # fresh run: clear any stale sentinels so this run's outcome is authoritative
        for j in jobs:
            done_marker(j["id"]).unlink(missing_ok=True)

    def _finish(jid, rc):
        r = running.pop(jid)
        r["logf"].close()
        r["outf"].close()
        used[r["resource"]] -= 1
        used["gpu_mem"] -= r.get("gpu_mem", 0)
        if r["group"]:
            busy_groups.discard(r["group"])
        if rc == 0:
            done_marker(jid).write_text(str(int(time.time())))   # sentinel gates RESUME (matches live rc==0)
            status[jid] = "done"
        else:
            done_marker(jid).unlink(missing_ok=True)             # a failed re-run clears a stale sentinel
            status[jid] = "failed"
        log(f"[jobpool] {jid}: {status[jid]} (rc={rc}, {time.time() - r['start']:.0f}s)")

    def _launch(j):
        logf = outf = proc = None
        try:
            logf = open(out / f"{j['id']}.log", "w")
            outf = open(out / f"{j['id']}.out", "w")
            jt = max(1, int(j.get("threads", n_threads)))  # per-job thread override, else the pool budget (min 1)
            jenv = {**os.environ, **{v: str(jt) for v in _THREAD_VARS}}
            proc = subprocess.Popen(j["cmd"], shell=True, stdout=outf, stderr=logf, env=jenv, start_new_session=True)
        except Exception:
            for h in (logf, outf):
                try:
                    h and h.close()
                except Exception:
                    pass
            raise
        running[j["id"]] = {"proc": proc, "resource": j["resource"], "group": j.get("group"),
                            "start": time.time(), "timeout": j.get("timeout", job_timeout),
                            "gpu_mem": j.get("gpu_mem", 0), "logf": logf, "outf": outf}
        used[j["resource"]] += 1
        used["gpu_mem"] += j.get("gpu_mem", 0)
        if j.get("group"):
            busy_groups.add(j["group"])
        log(f"[jobpool] launch {j['id']} ({j['resource']}"
            + (f", group={j['group']}" if j.get("group") else "") + f") — {len(running)} running")

    def _reap():                # move finished (or timed-out) children to terminal state; True if any finished
        did = False
        for jid, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is None and r["timeout"] and (time.time() - r["start"]) > r["timeout"]:
                log(f"[jobpool] {jid}: TIMEOUT after {r['timeout']}s — killing")
                _kill_group(r["proc"])
                rc = r["proc"].wait()
                rc = rc if rc != 0 else 137                  # ensure it counts as failure
            if rc is not None:
                _finish(jid, rc)
                did = True
        return did

    def _kill_all():            # abnormal-exit cleanup: terminate orphans + close handles
        for jid, r in list(running.items()):
            _kill_group(r["proc"])
            for h in (r["logf"], r["outf"]):
                try:
                    h.close()
                except Exception:
                    pass

    def _on_term(*_):
        raise KeyboardInterrupt("SIGTERM")

    # SIGTERM and SIGHUP (a closed tmux pane or ssh session) run the finally cleanup. Jobs run in their own
    # sessions, so a hangup no longer reaches them directly: without this they'd outlive the pool.
    installed = {}                                          # only restore what WE installed
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            installed[sig] = signal.getsignal(sig)
            signal.signal(sig, _on_term)
        except (ValueError, OSError):
            installed.pop(sig, None)                        # not on the main thread (e.g. under a test runner)

    try:
        while len(status) < len(jobs):
            launched = False
            for j in jobs:                                  # spec order → deterministic scheduling
                if j["id"] in status or j["id"] in running:
                    continue
                deps = j.get("deps", [])
                if any(status.get(d) in ("failed", "skipped") for d in deps):
                    status[j["id"]] = "skipped"
                    bad = [d for d in deps if status.get(d) in ("failed", "skipped")]
                    log(f"[jobpool] {j['id']}: skipped (dep {bad} not ok)")
                    launched = True
                    continue
                if not all(status.get(d) == "done" for d in deps):
                    continue                                # deps still pending
                if used[j["resource"]] >= caps[j["resource"]]:
                    continue                                # resource pool full
                g = j.get("group")
                if g and g in busy_groups:
                    continue                                # a same-group job is running
                if gpu_budget and used["gpu_mem"] + j.get("gpu_mem", 0) > gpu_budget:
                    continue                                # would exceed the concurrent GPU-memory budget
                _launch(j)
                launched = True
            reaped = _reap()                                # reaping a job UNBLOCKS dependents next pass → progress
            if len(status) >= len(jobs):
                break
            if not launched and not reaped:
                if not running:                             # nothing launched/reaped AND nothing running → stuck
                    for j in jobs:                          # (unreachable for a valid DAG; fail-safe, never hang)
                        status.setdefault(j["id"], "skipped")
                    break
                time.sleep(poll)                            # jobs running — wait for one to finish
    finally:
        for sig, orig in installed.items():                 # restore BEFORE killing → a signal mid-cleanup uses orig
            try:
                signal.signal(sig, orig)
            except (ValueError, OSError, TypeError):
                pass
        _kill_all()

    summary = {"status": status,
               "counts": {s: sum(1 for v in status.values() if v == s) for s in ("done", "failed", "skipped")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"[jobpool] DONE — {summary['counts']}")
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON file: {\"jobs\": [{id,cmd,resource,deps,group,timeout}]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--cpu-slots", type=int, default=4)
    ap.add_argument("--job-timeout", type=float, default=None, help="default per-job timeout in seconds")
    ap.add_argument("--threads-per-job", type=int, default=None,
                    help="BLAS/OMP threads per job (default: auto = cgroup_cores // (gpu+cpu slots))")
    ap.add_argument("--cpu-cores", type=float, default=None, help="override the detected cgroup core budget")
    ap.add_argument("--gpu-mem-mb", type=int, default=None,
                    help="concurrent GPU-memory budget in MB (default: auto = 0.92 × nvidia-smi total)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    spec = json.loads(pathlib.Path(a.spec).read_text())
    status = run_pool(spec["jobs"], a.out, gpu_slots=a.gpu_slots, cpu_slots=a.cpu_slots,
                      job_timeout=a.job_timeout, threads_per_job=a.threads_per_job, cpu_cores=a.cpu_cores,
                      gpu_mem_mb=a.gpu_mem_mb, resume=not a.no_resume)
    raise SystemExit(1 if any(v == "failed" for v in status.values()) else 0)


if __name__ == "__main__":
    main()
