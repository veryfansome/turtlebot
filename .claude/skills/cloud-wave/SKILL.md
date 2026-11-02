---
name: cloud-wave
description: Run a wave of world-model trainings on a RunPod GPU pod (cloud/) - a hyperparameter grid, the speed probe, or evolve candidates - and bring the results home with the pod confirmed stopped. Use when the user wants a grid search, a probe, or evolve candidates measured on RunPod. Launching costs money and needs the user's explicit go in the current conversation, with the printed cost estimate.
---

# Running a wave on RunPod

See [cloud/README.md](../../../cloud/README.md). Commands run from the repo root unless they start with `(cd learn && …)`. Run the shell scripts with `bash` (zsh doesn't word-split their ssh options).

## Steps

1. **Make the wave** (a record of what ran: never reuse a name):
   - the speed probe, before the first real wave on a GPU type: `(cd learn && uv run python ../cloud/wave.py probe NAME)`;
   - a grid: `(cd learn && uv run python ../cloud/wave.py grid NAME --param optim.batch=… --param optim.lr=…)`. The baseline genome is always included, as the point the grid is compared with; 3 seeds by default;
   - evolve candidates: `(cd learn && uv run python ../cloud/wave.py make NAME --genome ID=PATH … --meta LINEAGE.json)`, the lineage file giving each candidate's `parent`, `inventor`, `operator`, `axis_changed`, `rationale` (and `generation` when its parent isn't in this archive).
   It prints the estimate (jobs, hours, dollars). Pods run `git archive HEAD`, so any code the wave depends on must be committed first: ask the user to commit it (ship-changes skill) if it isn't.
2. **Check the data pin.** `git show HEAD:learn/data_source.json` and `git show HEAD:learn/eval/data_manifest.json`: every session in the manifest must be in the pinned revision (pods pull only those, `hub pull --manifest`, and refuse if one is missing). The pod's preflight then checks each file's fingerprint against the manifest; a session reconverted and pushed since the search was pinned fails there. If a session is missing, or `hub push` ran after a `convert --force`, every job fails preflight once the pod is paid for: don't launch, tell the user.
3. **Check what's already billing:** `bash cloud/runpod.sh reap`. Report anything UNTRACKED or unexpected to the user; never terminate a pod you didn't start without their say-so.
4. **Ask the user for a go**, giving the wave, the estimate and the budget (`RUNPOD_BUDGET_USD`, default $20). Their go is for that one wave.
5. **Run it in the background:** `bash cloud/lane.sh run NAME`. While it runs, report its progress lines now and then (records home, budget left).
6. **If the lane stopped early** (budget, failure), `(cd learn && uv run python ../cloud/wave.py fold NAME)` makes records for any genome whose seeds all came home. Then **report** from `(cd learn && uv run python ../cloud/wave.py summary NAME)`: per genome its fitness (higher is better: minus the log of the error relative to the baselines) and per-seed spread, against the baseline in the same wave. Differences within the seed spread, or under the measured noise floor, are noise. Confirm the lane ended with "terminate: CONFIRMED stopped".
7. **Evolve candidates:** `(cd learn && uv run python ../cloud/wave.py ingest NAME)` puts them in the archive under their env (e.g. `runpod-geforce-rtx-4090-c3`). Then `(cd learn && uv run python evolve/campaign.py push)`: the archive and impls otherwise live only on this disk.

## Rules

- Never launch without the user's go for that wave. A failed or stopped lane terminates its pod by itself; if it says termination wasn't confirmed, tell the user at once, with the pod id.
- Scores compare only within one env (GPU type and concurrency). Don't compare a pod's numbers with the Mac's.
- A grid's winner becomes the default only through the train-model skill (one change at a time, on all sessions).
- Results in `cloud/results/` are git-ignored. Commit a wave's `wave.json` and `genomes/` with the next change, as the record of what ran; its `chunks/` is git-ignored (impls go to the campaign repo, and this repo is public).
- Never print `~/.runpod.env` or the tokens in it, or put them on a command line.
- Never set `LANE_KEEP_POD=1` unless the user asked for it in this conversation.
