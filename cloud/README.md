# cloud/: training waves on RunPod

Runs batches of world-model trainings ("waves") on a rented RunPod GPU pod: hyperparameter grids, and the evolve search's candidates (`learn/evolve`), scored the same way. Adapted from the sibling projects (`terminal-wm/cloud`, `jepa/terminal-jepa`): `runpod.sh` and `lane.sh` keep every guard they learned the hard way; `jobpool.py` is jepa's, as is.

Status (2026-11-01): reviewed twice, never launched. The first launch is the speed probe (about $0.50), with the owner's go.

```bash
cd learn
uv run python ../cloud/wave.py probe probe-1                                  # the speed probe, first
uv run python ../cloud/wave.py grid bs-lr-1 --param optim.batch=256,1024,4096 --param optim.lr=3e-4,6e-4,1.2e-3
uv run python ../cloud/wave.py make cands-1 --genome g1=path/to/genome.json   # evolve candidates
uv run python ../cloud/wave.py estimate bs-lr-1                               # what it will cost, before any launch
cd ..
bash cloud/runpod.sh reap        # anything already billing on the account?
bash cloud/lane.sh run bs-lr-1   # provision -> bootstrap -> data -> wave -> poll -> download -> verify -> terminate
cd learn && uv run python ../cloud/wave.py summary bs-lr-1     # results side by side
uv run python ../cloud/wave.py verify bs-lr-1                  # every expected result came home
uv run python ../cloud/wave.py ingest cands-1                  # into the evolve archive
```

| File | What |
|---|---|
| `runpod.sh` | The provider client (curl + jq + ssh + rsync). Deploys a pod (recorded in `~/.runpod-pods.log` the moment it exists), ships the committed code (`git archive HEAD`: `learn/` + `cloud/`), syncs learn's environment, pulls the evolve search's sessions (`learn/eval/data_manifest.json`) at the pinned revision from the private Hugging Face dataset, starts a wave in tmux, pulls results, terminates. `reap` lists every pod on the account against that ledger. |
| `lane.sh` | One command per wave, end to end, with the pod confirmed stopped. |
| `wave.py` | On the Mac: make a wave (explicit genomes, a grid, or the probe), estimate its cost, verify and summarize results, ingest them into the evolve archive. |
| `wave_pod.py` | On the pod: `eval.preflight` once, `eval.smoke` and `eval.guard_leakage` per genome (ingesting skips the engine's own gates), `eval.adapter` per seed through the job pool, seeds folded into one record per genome. |
| `jobpool.py` | Runs a wave's trainings several at a time on one pod, threads sized to the pod's real CPU quota (pods report the host's 128 cores but are capped at ~13). |
| `probe.py` | How fast training is on the pod (GPU vs CPU, batch size, how many at once), to size waves before spending on them. |

`cloud/waves/NAME/` (made by wave.py) holds `wave.json`, `genomes/` and any new candidate impl under `chunks/`. Commit the first two; `chunks/` is git-ignored and goes to the campaign repo. Wave names are never reused.

**Safety.** Everything the lane does to cost money is bounded:
- A run's estimate must fit `RUNPOD_BUDGET_USD` (default $20) before a pod is rented, and the run stops when the pod's time reaches it.
- One pod at a time, SECURE cloud, on-demand.
- On any failure or at the budget, the lane pulls what finished, then terminates the pod and confirms it with RunPod (the terminate call alone exits 0 even on an API error). `LANE_KEEP_POD=1` keeps a failed pod up to inspect.
- Results come home on every poll, so a stop loses little.
- Deploys are never retried (a retried deploy is a second, untracked pod), every wait is bounded by wall clock, and `reap` finds a pod an interrupted launch left behind.

**SSH.** Nothing to configure: the pod gets the public keys in your ssh agent (`ssh-add -L`) through its `PUBLIC_KEY` variable, and ssh logs in with the agent. `RUNPOD_SSH_KEY` (a private key, with its `.pub` beside it) or an `~/.ssh/id_ed25519` / `id_rsa` with its `.pub` beside it is used instead when present. With no key at all, nothing is deployed. The keys in your RunPod account settings aren't enough: a pod created through the API without `PUBLIC_KEY` refused ssh (2026-11-01).

**Secrets.** `RUNPOD_API_KEY` and the Hugging Face token come from `~/.runpod.env`; neither is ever printed. The pod gets `HF_TOKEN` (or `HF_READ_TOKEN`, if set: a read-only one), as the sibling projects' pods do, through ssh's stdin into a mode-600 file on the pod, never on a command line; terminating the pod deletes it. The GraphQL API takes its key in the URL, so it can show in a local process listing while a call runs.

**Hosts.** `learn/` locks torch 2.15.0, whose only CUDA builds need a CUDA 13 driver (CUDA 12 wheels stop at 2.14.1), so `launch` re-rolls any pod whose driver is older rather than swap torch versions, which would change the scores. The deploy sends no CUDA filter by default (`RUNPOD_ALLOWED_CUDA=any`) and leaves it to the re-roll (a too-old host costs a few minutes of pod time before it's terminated): RunPod's API accepts filters only up to 13.0 (its enum, checked 2026-11-01), and a 13.0 filter left no RTX 4090 available while the first unfiltered host had a 13.2 driver (2026-11-01). The minimum driver is `RUNPOD_MIN_CUDA` (default 13.0). Each re-roll is a pod billed until it's terminated: usually a few minutes; at worst about 35 (up to 15 waiting for RunPod to open a port, then up to 20 for ssh), up to 4 tries.

**Comparing results.** Each record carries `env`, e.g. `runpod-geforce-rtx-4090-c3` (the GPU and how many trainings shared it: thread counts change float results). Scores compare only within one env; the evolve search selects within `fitness.selection_env`.

Tests (offline): `cd learn && uv run pytest ../cloud/tests -q`.
