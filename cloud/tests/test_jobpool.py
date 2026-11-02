"""Unit tests for cloud/jobpool.py — the fan-out scheduler's guarantees, verified with instrumented mock jobs
that record their REAL start/end intervals (so overlaps are checked against wall-clock, not just the schedule).

Ported from jepa/terminal-jepa/tests/test_jobpool.py. Run: cd learn && uv run pytest ../cloud/tests -q
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # cloud/
import jobpool


def _job(jid, resource="cpu", deps=None, group=None, dur=0.3, rc=0, emit_json=True, timeout=None, out_dir="."):
    """A mock job that records [start,end] to <id>.iv, optionally prints a JSON line (build jobs like reencode
    do NOT — they print plain text), then exits rc. `emit_json` lets tests exercise the non-JSON resume path."""
    js = "print(json.dumps({'id':'%s'}));" % jid if emit_json else ""
    py = (f"import time,json,sys; s=time.time(); time.sleep({dur}); e=time.time(); "
          f"open(r'{out_dir}/{jid}.iv','w').write(f'{{s}} {{e}}'); {js} sys.exit({rc})")
    j = {"id": jid, "cmd": f"{sys.executable} -c \"{py}\"", "resource": resource, "deps": deps or [], "group": group}
    if timeout is not None:
        j["timeout"] = timeout
    return j


def _intervals(out, ids):
    iv = {}
    for i in ids:
        p = pathlib.Path(out) / f"{i}.iv"
        if p.exists():
            s, e = p.read_text().split()
            iv[i] = (float(s), float(e))
    return iv


def _max_overlap(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1)); events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, d in events:
        cur += d; mx = max(mx, cur)
    return mx


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def test_resource_caps(tmp_path):
    out = str(tmp_path)
    jobs = [_job(f"c{i}", "cpu", dur=0.4, out_dir=out) for i in range(5)] + \
           [_job(f"g{i}", "gpu", dur=0.4, out_dir=out) for i in range(3)]
    st = jobpool.run_pool(jobs, out, gpu_slots=1, cpu_slots=2, poll=0.05, resume=False, log=lambda *_: None)
    assert all(v == "done" for v in st.values()), st
    iv = _intervals(out, [j["id"] for j in jobs])
    assert _max_overlap([iv[f"c{i}"] for i in range(5)]) <= 2, "cpu cap exceeded"
    assert _max_overlap([iv[f"g{i}"] for i in range(3)]) <= 1, "gpu cap exceeded"


def test_deps_ordered(tmp_path):
    out = str(tmp_path)
    jobs = [_job("a", dur=0.2, out_dir=out), _job("b", dur=0.2, out_dir=out),
            _job("c", deps=["a", "b"], dur=0.2, out_dir=out)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, poll=0.05, resume=False, log=lambda *_: None)
    assert all(v == "done" for v in st.values())
    iv = _intervals(out, ["a", "b", "c"])
    assert iv["c"][0] >= iv["a"][1] and iv["c"][0] >= iv["b"][1], "c started before its deps finished"


def test_diamond(tmp_path):
    out = str(tmp_path)
    jobs = [_job("a", dur=0.15, out_dir=out), _job("b", deps=["a"], dur=0.15, out_dir=out),
            _job("c", deps=["a"], dur=0.15, out_dir=out), _job("d", deps=["b", "c"], dur=0.15, out_dir=out)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, poll=0.05, resume=False, log=lambda *_: None)
    assert all(v == "done" for v in st.values())
    iv = _intervals(out, ["a", "b", "c", "d"])
    assert iv["b"][0] >= iv["a"][1] and iv["c"][0] >= iv["a"][1], "b/c started before a"
    assert iv["d"][0] >= iv["b"][1] and iv["d"][0] >= iv["c"][1], "d started before b/c"


def test_group_mutex(tmp_path):
    out = str(tmp_path)
    jobs = [_job(f"m{i}", "cpu", group="G", dur=0.3, out_dir=out) for i in range(3)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, poll=0.05, resume=False, log=lambda *_: None)
    assert all(v == "done" for v in st.values())
    ivs = list(_intervals(out, [f"m{i}" for i in range(3)]).values())
    for i in range(len(ivs)):
        for j in range(i + 1, len(ivs)):
            assert not _overlap(ivs[i], ivs[j]), "two same-group jobs overlapped"


def test_failure_isolation(tmp_path):
    out = str(tmp_path)
    jobs = [_job("x", rc=1, dur=0.1, out_dir=out), _job("y", deps=["x"], dur=0.1, out_dir=out),
            _job("z", dur=0.1, out_dir=out)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, poll=0.05, resume=False, log=lambda *_: None)
    assert st == {"x": "failed", "y": "skipped", "z": "done"}, st


def test_transitive_skip(tmp_path):
    out = str(tmp_path)
    jobs = [_job("x", rc=1, dur=0.1, out_dir=out), _job("y", deps=["x"], dur=0.1, out_dir=out),
            _job("z", deps=["y"], dur=0.1, out_dir=out)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, poll=0.05, resume=False, log=lambda *_: None)
    assert st == {"x": "failed", "y": "skipped", "z": "skipped"}, st


def test_cycle_rejected(tmp_path):
    jobs = [{"id": "a", "cmd": "true", "resource": "cpu", "deps": ["b"]},
            {"id": "b", "cmd": "true", "resource": "cpu", "deps": ["a"]}]
    try:
        jobpool.run_pool(jobs, str(tmp_path), resume=False, log=lambda *_: None); assert False
    except ValueError as e:
        assert "cycle" in str(e)


def test_duplicate_ids_rejected(tmp_path):
    jobs = [{"id": "a", "cmd": "true", "resource": "cpu"}, {"id": "a", "cmd": "true", "resource": "cpu"}]
    try:
        jobpool.run_pool(jobs, str(tmp_path), resume=False, log=lambda *_: None); assert False
    except ValueError as e:
        assert "duplicate" in str(e)


def test_zero_slots_rejected(tmp_path):
    # M1: a resource with jobs but ≤0 slots must be REJECTED up front (else infinite hang) — gpu AND cpu
    for res, gs, cs in [("gpu", 0, 4), ("gpu", -1, 4), ("cpu", 1, 0)]:
        try:
            jobpool.run_pool([_job("j", res, out_dir=str(tmp_path))], str(tmp_path),
                             gpu_slots=gs, cpu_slots=cs, resume=False, log=lambda *_: None)
            assert False, f"{res} with 0 slots not rejected"
        except ValueError as e:
            assert "slots" in str(e)


def test_zero_slots_no_false_reject(tmp_path):
    # M1 must NOT reject a resource with ZERO jobs even if its slots are 0 (only-cpu jobs + gpu_slots=0 must run)
    out = str(tmp_path)
    st = jobpool.run_pool([_job("c", "cpu", dur=0.1, out_dir=out)], out,
                          gpu_slots=0, cpu_slots=2, resume=False, log=lambda *_: None)
    assert st == {"c": "done"}, st


def test_resume_success_sentinel(tmp_path):
    # a plain success re-run is skipped via the .done sentinel (not by parsing stdout)
    out = str(tmp_path)
    jobs = [_job("r1", dur=0.1, out_dir=out), _job("r2", deps=["r1"], dur=0.1, out_dir=out)]
    jobpool.run_pool(jobs, out, cpu_slots=2, poll=0.05, resume=False, log=lambda *_: None)
    assert (tmp_path / "r1.done").exists() and (tmp_path / "r2.done").exists()
    iv1 = _intervals(out, ["r1", "r2"])
    msgs = []
    st = jobpool.run_pool(jobs, out, cpu_slots=2, poll=0.05, resume=True, log=lambda m: msgs.append(m))
    assert all(v == "done" for v in st.values())
    assert sum("resume:" in m for m in msgs) == 2, "resume did not skip already-done jobs"
    assert _intervals(out, ["r1", "r2"]) == iv1, "resume re-ran a job"


def test_resume_non_json_build_job(tmp_path):
    # M2 (false negative fixed): a job that exits 0 but prints NO json (like reencode/precompute) must resume-skip
    out = str(tmp_path)
    jobs = [_job("b", emit_json=False, dur=0.1, out_dir=out)]
    jobpool.run_pool(jobs, out, resume=False, poll=0.05, log=lambda *_: None)
    assert (tmp_path / "b.done").exists(), "success sentinel not written for a non-JSON job"
    iv1 = _intervals(out, ["b"])
    msgs = []
    jobpool.run_pool(jobs, out, resume=True, poll=0.05, log=lambda m: msgs.append(m))
    assert sum("resume:" in m for m in msgs) == 1, "resume did not skip the non-JSON build job"
    assert _intervals(out, ["b"]) == iv1, "resume re-ran the non-JSON build job"


def test_resume_json_then_fail(tmp_path):
    # M2 (false positive fixed): a job that prints full JSON then EXITS NONZERO must NOT be resume-skipped
    out = str(tmp_path)
    jobs = [_job("f", emit_json=True, rc=1, dur=0.1, out_dir=out)]
    jobpool.run_pool(jobs, out, resume=False, poll=0.05, log=lambda *_: None)
    assert not (tmp_path / "f.done").exists(), "sentinel written for a job that exited nonzero"
    iv1 = _intervals(out, ["f"])
    jobpool.run_pool(jobs, out, resume=True, poll=0.05, log=lambda *_: None)
    assert _intervals(out, ["f"]) != iv1, "resume wrongly skipped a job that printed JSON then failed"


def test_timeout_kills(tmp_path):
    # M4: a job exceeding its timeout is killed and marked failed (no infinite hang on a stuck child)
    out = str(tmp_path)
    jobs = [_job("slow", dur=5.0, timeout=0.5, out_dir=out)]
    t0 = __import__("time").time()
    st = jobpool.run_pool(jobs, out, poll=0.1, resume=False, log=lambda *_: None)
    assert st["slow"] == "failed", "timed-out job not marked failed"
    assert __import__("time").time() - t0 < 3.0, "timeout did not kill promptly"


def _omp_job(jid, threads=None):
    cmd = f"{sys.executable} -c \"import os; print(os.environ.get('OMP_NUM_THREADS','unset'))\""
    j = {"id": jid, "cmd": cmd, "resource": "cpu", "deps": [], "group": None}
    if threads is not None:
        j["threads"] = threads
    return j


def test_thread_env_injected(tmp_path):
    # cpu_cores=12, peak concurrency = gpu_slots(1)+cpu_slots(3)=4 → 12//4 = 3 threads/job
    out = str(tmp_path)
    jobpool.run_pool([_omp_job("t")], out, gpu_slots=1, cpu_slots=3, cpu_cores=12,
                     resume=False, poll=0.05, log=lambda *_: None)
    assert (tmp_path / "t.out").read_text().strip() == "3", (tmp_path / "t.out").read_text()


def test_thread_per_job_override(tmp_path):
    out = str(tmp_path)
    jobpool.run_pool([_omp_job("t", threads=1)], out, cpu_cores=12, cpu_slots=3,
                     resume=False, poll=0.05, log=lambda *_: None)
    assert (tmp_path / "t.out").read_text().strip() == "1", "per-job threads override ignored"


def test_detect_cpu_cores():
    c = jobpool.detect_cpu_cores()
    assert isinstance(c, float) and c >= 1.0, c


def test_gpu_mem_budget(tmp_path):
    # budget 10000: two 8000MB jobs can't overlap (16000>10000) → serialize even though cpu_slots allows both
    out = str(tmp_path)
    jobs = [dict(_job("big1", dur=0.3, out_dir=out), gpu_mem=8000),
            dict(_job("big2", dur=0.3, out_dir=out), gpu_mem=8000)]
    st = jobpool.run_pool(jobs, out, cpu_slots=4, gpu_mem_mb=10000, poll=0.05, resume=False, log=lambda *_: None)
    assert all(v == "done" for v in st.values()), st
    iv = _intervals(out, ["big1", "big2"])
    assert not _overlap(iv["big1"], iv["big2"]), "two jobs exceeding the gpu-mem budget overlapped"


def test_gpu_mem_small_overlap(tmp_path):
    # budget 10000: three 3000MB jobs (9000<10000) SHOULD overlap → peak gpu use ≤ budget, all concurrent
    out = str(tmp_path)
    jobs = [dict(_job(f"s{i}", dur=0.3, out_dir=out), gpu_mem=3000) for i in range(3)]
    jobpool.run_pool(jobs, out, cpu_slots=4, gpu_mem_mb=10000, poll=0.05, resume=False, log=lambda *_: None)
    ivs = [_intervals(out, [f"s{i}"])[f"s{i}"] for i in range(3)]
    assert _max_overlap(ivs) == 3, "3 small jobs within the gpu-mem budget did not run concurrently"


def test_gpu_mem_too_big_rejected(tmp_path):
    jobs = [dict(_job("huge", out_dir=str(tmp_path)), gpu_mem=20000)]
    try:
        jobpool.run_pool(jobs, str(tmp_path), gpu_mem_mb=10000, resume=False, log=lambda *_: None)
        assert False, "a job needing more gpu_mem than the budget was not rejected"
    except ValueError as e:
        assert "gpu_mem" in str(e)


def test_gpu_mem_no_failure_coupling(tmp_path):
    # KEY property: gpu_mem is a RESOURCE constraint, not a dep — a big job FAILING must not skip an independent
    # big job it merely serialized with (contrast with dep-based skipping)
    out = str(tmp_path)
    jobs = [dict(_job("f", rc=1, dur=0.1, out_dir=out), gpu_mem=8000),
            dict(_job("ok", dur=0.1, out_dir=out), gpu_mem=8000)]
    st = jobpool.run_pool(jobs, out, gpu_mem_mb=10000, poll=0.05, resume=False, log=lambda *_: None)
    assert st == {"f": "failed", "ok": "done"}, st


if __name__ == "__main__":
    import inspect
    import tempfile
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        if inspect.signature(fn).parameters:            # test(tmp_path) vs test()
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
        print(f"PASS {name}")
    print(f"ALL {len(fns)} JOBPOOL TESTS PASSED")


def test_a_timeout_kills_the_jobs_children_too(tmp_path):
    """A job that starts a long-running child (as eval.adapter starts the training): the timeout must take the child
    down with it, not leave it running after the job is marked failed."""
    import os
    import time as _time
    pidfile = tmp_path / 'child.pid'
    child = f"{sys.executable} -c \"import os,time; open(r'{pidfile}','w').write(str(os.getpid())); time.sleep(60)\""
    job = {"id": "parent", "cmd": f"{child}; echo after", "resource": "cpu", "timeout": 2}
    status = jobpool.run_pool([job], tmp_path / "out", gpu_slots=0, cpu_slots=1, poll=0.2, log=lambda m: None)
    assert status["parent"] == "failed"
    pid = int(pidfile.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        _time.sleep(0.1)
    raise AssertionError(f"the job's child {pid} is still running after its timeout")
