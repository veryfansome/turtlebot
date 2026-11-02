"""Waves: batches of world-model trainings for a RunPod pod, made, checked and ingested here.

    cd learn
    uv run python ../cloud/wave.py make NAME --genome ID=PATH ... [--seeds 0,1,2] [--meta LINEAGE.json]
    uv run python ../cloud/wave.py grid NAME --param optim.batch=256,1024,4096 --param optim.lr=3e-4,6e-4 [--seeds 0,1,2]
    uv run python ../cloud/wave.py probe NAME                      # the speed probe (cloud/probe.py)
    uv run python ../cloud/wave.py estimate NAME [--minutes-per-job M]
    uv run python ../cloud/wave.py verify NAME                     # exit 0 ok, 2 missing results, 3 all failed
    uv run python ../cloud/wave.py fold NAME                       # records from pulled seeds, after an early stop
    uv run python ../cloud/wave.py summary NAME                    # results side by side
    uv run python ../cloud/wave.py ingest NAME                     # into the evolve archive (learn/evolve), once

A wave is a set of genomes (pet_learn.genome) x seeds, each scored on the pod exactly as the evolve
search scores a candidate: eval.preflight once, eval.smoke and eval.guard_leakage per genome (ingesting
skips the engine's own gates, so the pod runs them), then eval.adapter per seed. A grid search is a
wave whose genomes differ only in params; an evolve wave is a round's candidates. Each genome's seeds
fold into one record, cloud/results/NAME/<id>.json, in the form `evolve ingest` takes: fitness (the
mean combined_score), correct, per_seed, public, text_feedback, env.

cloud/waves/NAME/ holds wave.json, the genome files, and any impl file the committed code doesn't have
(a new candidate's), which the pod copies into learn/evolve/chunks.
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

CLOUD = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CLOUD)
LEARN = os.path.join(REPO, 'learn')
WAVES = os.path.join(CLOUD, 'waves')
RESULTS = os.path.join(CLOUD, 'results')
sys.path.insert(0, LEARN)

DEFAULT_MINUTES_PER_JOB = 10.0  # until a probe has measured it; a full run is ~7 min on the Mac's GPU
OVERHEAD_MINUTES = 20.0  # launch, bootstrap (uv sync of a CUDA torch), data pull, teardown
FULL_RUN_ITERS = 15000  # a full training run, roughly: it stops when val stops improving
DEFAULT_GPU = 'NVIDIA GeForce RTX 4090'  # cloud/runpod.sh's default
META_KEYS = ('id', 'parent', 'inventor', 'operator', 'axis_changed', 'rationale', 'generation')


def in_head(path: str) -> bool:
    """The file is in the committed code (what a pod gets: git archive HEAD), unchanged since."""
    rel = os.path.relpath(path, REPO)
    if subprocess.run(['git', '-C', REPO, 'cat-file', '-e', f'HEAD:{rel}'], capture_output=True).returncode != 0:
        return False
    if subprocess.run(['git', '-C', REPO, 'diff', '--quiet', 'HEAD', '--', rel], capture_output=True).returncode != 0:
        raise SystemExit(f'{rel} is committed but edited since: impls are append-only (add a new impl instead), '
                         "and the pod would get the committed version")
    return True


def write_wave(name: str, genomes: dict, seeds: list, split: str, mode: str, concurrency: int,
               job_timeout_s: int, meta: dict = None, kind: str = 'score') -> str:
    """cloud/waves/NAME/: wave.json, genomes/<id>.json, and chunks/<axis>/<impl>.py for impls the committed
    code doesn't have. genomes: {id: genome dict}."""
    from pet_learn import genome as genomes_mod
    out = os.path.join(WAVES, name)
    if os.path.exists(out):
        raise SystemExit(f'{out} exists: pick a new wave name (waves are a record of what ran)')
    # Everything is checked before anything is written: a refused wave leaves nothing behind.
    checked, chunks = {}, {}
    for gid, g in genomes.items():
        if not gid.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise SystemExit(f'genome id {gid!r}: letters, digits, - _ . only')
        checked[gid] = g = genomes_mod.load_dict(g)
        for axis, spec in g['chunks'].items():
            path = os.path.join(genomes_mod.CHUNKS, axis, spec['impl'] + '.py')
            if not os.path.isfile(path):
                raise SystemExit(f'{gid}: no impl {axis}/{spec["impl"]}')
            if not in_head(path):
                chunks[f'{axis}/{spec["impl"]}.py'] = path
    for gid, m in (meta or {}).items():
        if gid not in genomes or set(m) - set(META_KEYS):
            raise SystemExit(f'--meta: {gid!r} must be one of the wave\'s genomes, with keys among {META_KEYS}')
    os.makedirs(os.path.join(out, 'genomes'))
    for rel, path in chunks.items():
        os.makedirs(os.path.dirname(os.path.join(out, 'chunks', rel)), exist_ok=True)
        shutil.copy(path, os.path.join(out, 'chunks', rel))
    for gid, g in checked.items():
        with open(os.path.join(out, 'genomes', f'{gid}.json'), 'w') as f:
            json.dump(g, f, indent=1)
    chunks = sorted(chunks)
    head = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    wave = {'name': name, 'kind': kind, 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'code': head,
            'genomes': {gid: f'genomes/{gid}.json' for gid in genomes}, 'chunks': chunks,
            'seeds': seeds, 'split': split, 'mode': mode, 'concurrency': concurrency,
            'job_timeout_s': job_timeout_s, 'meta': meta or {}}
    with open(os.path.join(out, 'wave.json'), 'w') as f:
        json.dump(wave, f, indent=1)
    return out


def load_wave(name: str) -> dict:
    with open(os.path.join(WAVES, name, 'wave.json')) as f:
        return json.load(f)


def grid_genomes(base: dict, params: dict, pairs: bool, prefix: str = 'grid') -> dict:
    """{id: genome} for every combination of params ({'optim.batch': [256, 1024], ...}), or with pairs,
    the i-th value of each list together (e.g. a batch with its sqrt-scaled lr). Ids start with prefix
    (the wave's name), so they stay unique across waves once ingested."""
    keys = list(params)
    if pairs:
        lengths = {len(v) for v in params.values()}
        if len(lengths) != 1:
            raise SystemExit('--pairs needs every --param list the same length')
        combos = list(zip(*params.values()))
    else:
        combos = list(itertools.product(*params.values()))
    out = {}
    for combo in combos:
        g = json.loads(json.dumps(base))
        parts = []
        for key, value in zip(keys, combo):
            axis, param = key.split('.', 1)
            g['chunks'][axis]['params'][param] = value
            parts.append(f'{param}{value:g}' if isinstance(value, float) else f'{param}{value}')
        out[f'{prefix}-' + '-'.join(parts)] = g
    return out


def parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def pod_price() -> float:
    r = subprocess.run(['bash', os.path.join(CLOUD, 'runpod.sh'), 'price'], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'could not get the pod price: {r.stderr.strip()[-300:]}')
    return float(r.stdout.strip().splitlines()[-1])


def latest_probe(gpu: str = None):
    """The newest probe.json measured on this GPU type (default: RUNPOD_GPU_TYPE, else runpod.sh's), or None."""
    gpu = gpu or os.environ.get('RUNPOD_GPU_TYPE', DEFAULT_GPU)
    found = []
    for name in os.listdir(RESULTS) if os.path.isdir(RESULTS) else []:
        path = os.path.join(RESULTS, name, 'probe.json')
        if os.path.exists(path):
            with open(path) as f:
                probe = json.load(f)
            if probe.get('gpu') == gpu:
                found.append((os.path.getmtime(path), probe))
    return max(found, key=lambda p: p[0])[1] if found else None


def throughput(probe: dict, concurrency: int):
    """Total training iterations per second at this concurrency: the probe's measurement for the largest
    number of trainings at once it tried that doesn't exceed it."""
    rates = {int(k): v for k, v in probe.get('concurrency_iters_per_second', {}).items()}
    usable = [k for k in rates if k <= concurrency]
    return rates[max(usable)] if usable else None


def estimate(wave: dict, minutes_per_job=None, price=None, probe='latest') -> dict:
    """Pod hours and dollars. With a probe from this GPU type: total iterations over the measured total
    throughput at the wave's concurrency. Otherwise minutes_per_job (each training's own time) per job,
    concurrency at once."""
    jobs = len(wave['genomes']) * len(wave['seeds']) if wave['kind'] == 'score' else 1
    concurrency = max(1, wave['concurrency'])
    probe = latest_probe() if probe == 'latest' else probe
    rate = throughput(probe, concurrency) if probe and not minutes_per_job else None
    if wave['kind'] != 'score':
        work, basis = 15.0, 'the probe itself'
    elif rate:
        # Full rounds of `concurrency` trainings at that rate, then the last, partly filled round at its own.
        k = min(concurrency, jobs)
        rounds, rest = divmod(jobs, k)
        work = rounds * k * FULL_RUN_ITERS / throughput(probe, k) / 60
        if rest:
            work += rest * FULL_RUN_ITERS / (throughput(probe, rest) or throughput(probe, k)) / 60
        basis = f'probe: {rate:.0f} iterations/s at {concurrency} at once'
    else:
        work = jobs * (minutes_per_job or DEFAULT_MINUTES_PER_JOB) / concurrency
        basis = f'{minutes_per_job or DEFAULT_MINUTES_PER_JOB} min per training (no probe for this GPU)'
    minutes = OVERHEAD_MINUTES + work
    price = price if price is not None else pod_price()
    return {'jobs': jobs, 'concurrency': concurrency, 'basis': basis, 'hours': minutes / 60,
            'price_per_hour': price, 'dollars': minutes / 60 * price}


def record_state(record: dict) -> str:
    """ok: a real measurement; failed: a measured candidate failure (ingestable as such); bad: unusable.
    Testing for the presence of 'fitness' would pass a {"fitness": null} failure, so it's typed."""
    fitness = record.get('fitness')
    if record.get('correct') is True and isinstance(fitness, (int, float)) and not isinstance(fitness, bool):
        return 'ok'
    if record.get('correct') is False:
        return 'failed'
    return 'bad'


def verify(name: str) -> int:
    wave = load_wave(name)
    results = os.path.join(RESULTS, name)
    if wave['kind'] == 'probe':
        ok = os.path.exists(os.path.join(results, 'probe.json'))
        print(f'{name}: probe results {"present" if ok else "MISSING"}')
        return 0 if ok else 2
    states = {}
    for gid in wave['genomes']:
        path = os.path.join(results, f'{gid}.json')
        if not os.path.exists(path):
            states[gid] = 'missing'
            continue
        with open(path) as f:
            states[gid] = record_state(json.load(f))
    for gid, st in states.items():
        print(f'  {st:8s} {gid}')
    if any(st in ('missing', 'bad') for st in states.values()):
        return 2
    if all(st == 'failed' for st in states.values()):
        return 3  # every candidate failing is the environment, not the candidates
    return 0


def fold_home(name: str):
    """Records for genomes whose seeds came home but whose record didn't (a run stopped early)."""
    import wave_pod
    wave = load_wave(name)
    results = os.path.join(RESULTS, name)
    with open(os.path.join(results, '.env')) as f:
        env = json.load(f)
    for gid in wave['genomes']:
        out = os.path.join(results, f'{gid}.json')
        seeds = [os.path.exists(os.path.join(results, gid, f's{s}', 'metrics.json')) for s in wave['seeds']]
        if os.path.exists(out) or not all(seeds):
            print(f'{gid}: {"has a record" if os.path.exists(out) else f"{sum(seeds)}/{len(seeds)} seeds home: not folded"}')
            continue
        with open(out, 'w') as f:
            json.dump(wave_pod.fold(gid, wave, results, env['env'], env.get('commit')), f, indent=1)
        print(f'{gid}: folded')


def summary(name: str):
    wave = load_wave(name)
    results = os.path.join(RESULTS, name)
    if wave['kind'] == 'probe':
        with open(os.path.join(results, 'probe.json')) as f:
            print(json.dumps(json.load(f), indent=1))
        return
    rows = []
    for gid in wave['genomes']:
        path = os.path.join(results, f'{gid}.json')
        if not os.path.exists(path):
            rows.append((gid, None, 'missing', ''))
            continue
        with open(path) as f:
            r = json.load(f)
        seeds = [round(s['combined_score'], 4) for s in r.get('per_seed', []) if s.get('combined_score') is not None]
        rows.append((gid, r.get('fitness'), record_state(r), f'seeds {seeds}' if seeds else r.get('guardrail', '')))
    rows.sort(key=lambda row: -(row[1] if isinstance(row[1], (int, float)) else -1e9))
    for gid, fitness, st, note in rows:
        print(f'{gid:40s} {fitness if fitness is None else round(fitness, 4)!s:>8} {st:7s} {note}')


def ingest(name: str, env: str = None, allow_all_failed: bool = False):
    """Each genome's record into the evolve archive, once: a wave's ingested ids are kept in
    .ingested, so running it again appends nothing twice."""
    wave = load_wave(name)
    results = os.path.join(RESULTS, name)
    state = verify(name)
    if state == 2 or (state == 3 and not allow_all_failed):
        raise SystemExit('not ingesting: ' + ('results are missing or unusable' if state == 2 else
                                              'every genome failed (the environment, not the candidates?)'))
    marker = os.path.join(results, '.ingested')
    done = set(open(marker).read().split()) if os.path.exists(marker) else set()
    for gid, rel in wave['genomes'].items():
        if gid in done:
            print(f'{gid}: already ingested')
            continue
        path = os.path.join(results, f'{gid}.json')
        with open(path) as f:
            record = json.load(f)
        meta = wave['meta'].get(gid, {})
        cmd = ['evolve', '--root', LEARN, 'ingest', '--result', path, '--genome', os.path.join(WAVES, name, rel),
               '--id', meta.get('id', gid), '--mode', wave['mode'], '--split', wave['split'],
               '--env', env or record.get('env') or 'runpod']
        for key in ('parent', 'inventor', 'operator', 'axis_changed', 'rationale', 'generation'):
            if meta.get(key) is not None:
                cmd += [f'--{key.replace("_", "-")}', str(meta[key])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f'{gid}: ingest failed, stopping (earlier ones are recorded in {marker}): {r.stderr.strip()[-400:]}')
        with open(marker, 'a') as f:
            f.write(gid + '\n')
        print(f'{gid}: ingested')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('name')
        p.add_argument('--seeds', default='0,1,2')
        p.add_argument('--split', default='inner', choices=['inner', 'final'])
        p.add_argument('--mode', default='full', choices=['proxy', 'full'])
        p.add_argument('--concurrency', type=int, default=3, help='trainings at once on the pod (they share its GPU)')
        p.add_argument('--job-timeout', type=int, default=3600)

    p = sub.add_parser('make')
    common(p)
    p.add_argument('--genome', action='append', required=True, help='ID=PATH (a genome JSON) or ID=baseline')
    p.add_argument('--meta', help='JSON {id: {parent, inventor, operator, axis_changed, rationale, generation}}: '
                                  "each candidate's lineage, passed to evolve ingest")
    p = sub.add_parser('grid')
    common(p)
    p.add_argument('--base', default='baseline', help='a genome JSON, or baseline')
    p.add_argument('--param', action='append', required=True, help='axis.param=v1,v2,... e.g. optim.batch=256,1024')
    p.add_argument('--pairs', action='store_true', help='take the lists together, not every combination')
    p = sub.add_parser('probe')
    p.add_argument('name')
    for verb in ('estimate', 'verify', 'fold', 'summary', 'ingest'):
        p = sub.add_parser(verb)
        p.add_argument('name')
        if verb == 'estimate':
            p.add_argument('--minutes-per-job', type=float)
            p.add_argument('--price', type=float, help="$/hr; default: RunPod's price for the configured GPU")
        if verb == 'ingest':
            p.add_argument('--env', help="default: the env each record carries (the pod's GPU and concurrency)")
            p.add_argument('--allow-all-failed', action='store_true')
    args = parser.parse_args()

    from pet_learn import genome as genomes_mod
    if args.cmd in ('make', 'grid'):
        seeds = [int(s) for s in args.seeds.split(',')]
        if args.cmd == 'make':
            genomes = {}
            for spec in args.genome:
                gid, _, path = spec.partition('=')
                genomes[gid] = genomes_mod.load(None if path == 'baseline' else path)
        else:
            base = genomes_mod.load(None if args.base == 'baseline' else args.base)
            params = {}
            for spec in args.param:
                key, _, values = spec.partition('=')
                if '.' not in key or key.split('.')[0] not in genomes_mod.AXES:
                    raise SystemExit(f'--param {key!r}: axis.param, axis one of {genomes_mod.AXES}')
                params[key] = [parse_value(v) for v in values.split(',')]
            base_id = f'{args.name}-baseline' if args.base == 'baseline' else f'{args.name}-base'
            genomes = {base_id: base, **grid_genomes(base, params, args.pairs, prefix=args.name)}
        meta = None
        if args.cmd == 'make' and args.meta:
            with open(args.meta) as f:
                meta = json.load(f)
        out = write_wave(args.name, genomes, seeds, args.split, args.mode, args.concurrency, args.job_timeout, meta)
        print(f'wrote {out}: {len(genomes)} genomes x {len(seeds)} seeds')
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in estimate(load_wave(args.name)).items()}))
    elif args.cmd == 'probe':
        out = write_wave(args.name, {'baseline': genomes_mod.load()}, [0], 'inner', 'full', 1, 3600, kind='probe')
        print(f'wrote {out}')
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in estimate(load_wave(args.name)).items()}))
    elif args.cmd == 'estimate':
        est = estimate(load_wave(args.name), args.minutes_per_job, args.price)
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in est.items()}))
    elif args.cmd == 'verify':
        sys.exit(verify(args.name))
    elif args.cmd == 'fold':
        fold_home(args.name)
    elif args.cmd == 'summary':
        summary(args.name)
    else:
        ingest(args.name, args.env, args.allow_all_failed)


if __name__ == '__main__':
    main()
