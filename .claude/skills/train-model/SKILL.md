---
name: train-model
description: Convert recordings and train the driving model (learn/: world model, then driver), then compare the run against the current best using its run record. Use after a new recording, after changing code in learn/, or to compare training settings.
---

# Training the driving model

Everything runs on the Mac. Commands below are from the repo root; `learn/` is a uv project, so its commands run as `(cd learn && uv run ...)`, which works whatever the current folder was. See [learn/README.md](../../../learn/README.md) and [learn/MODEL.md](../../../learn/MODEL.md).

## Steps

1. **Data.** `ops/pull-drives.sh` (if it fails, say so: don't train on stale data without telling the user), then `(cd learn && uv run python -m pet_learn.convert)` (converts only new sessions). If the converter or the steps format changed, `--force` reconverts every session: say so, because results before and after aren't comparable. Read each session's summary (driving time, rescues, how long the lidar was missing) and tell the user anything odd. Then `(cd learn && uv run python -m pet_learn.hub push)`: the sessions go to the private Hugging Face dataset RunPod pulls from, and the pinned revision in `learn/data_source.json` changes (commit it with the next change). Pods download only the evolve search's sessions (`learn/eval/data_manifest.json`, through `hub pull --manifest`), so new sessions don't change what waves measure: they're the search's final split. After a `convert --force`, don't push: the pushed files would replace the sessions the search is pinned to, and its preflight on every pod would refuse them. Ask the user first, since re-pinning the search (`eval.manifest --write`) starts a new measurement epoch and is their call. `(cd learn && uv run python -m pet_learn.hub status)` only reads.
2. **Train on every session.** Always all of `data/steps/`, new recordings included, unless the user says otherwise. Make sure nothing else is training first (`ps aux | grep '[p]et_learn'`): two runs share the CPU and GPU, which skews times. While a training runs, don't `git stash`, check out another branch or change files in `learn/`: the run loads code from disk.
   ```bash
   (cd learn && uv run python -m pet_learn.train_world)                                       # stops itself when val error stops improving
   (cd learn && uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt)   # the world model just trained
   ```
   Run them in the background (on the Mac's GPU, on 6 sessions: about 8 min for the world model and 14 min for the driver; each record's `seconds` has the exact time). Run them plainly: the defaults are the settings. If a setting needs to differ every time, change its default in the code rather than passing a flag, so no run can forget it. A flag is only for an experiment, and the report says which.
3. **Compare.** `(cd learn && uv run python -m pet_learn.runs)` lists every run's record (`data/models/<model>.json`), oldest first: code version, data, the settings that differ from the defaults, when and why it stopped, and test results. Compare against the current best and check the records first:
   - The same data (the `data` fingerprint: sessions and the exact steps files, so a reconversion shows), unless the new data is what's being tested.
   - The same code version (`commit+changes`, which covers only the training code, `learn/pet_learn` and its dependencies), unless the code is what's being tested. If the commits differ, `git diff <a> <b> -- learn/pet_learn` shows whether the training code did.
   - One difference at a time. If more than one differs, say so, since the comparison can't tell which one mattered.
4. **Report** the test results (the last 10% of each session's driving, never used to pick the model), in the same form every time: the world model's error relative to its baselines at 1 s per head, bump and slip AUC, the driver's value rank correlation, and the policy check (forward and turn for goals ahead, left, right, back-left, back-right and behind). Give the current best's numbers beside them, and say which model is now the best and why.

## Rules

- Never delete models or data without asking. A short smoke run (`--iters 200`) also saves a model and record: name them to the user.
- A world model trained before the inputs changed won't load into `train_driver` (it says so up front). Retrain the world model.
- Training defaults to the Mac's GPU (MPS). For timing, benchmark only while nothing else runs.
- Models without a `.json` beside them were saved before run records existed (up to world-20261101-145954 and driver-20261101-143434), and `pet_learn.runs` doesn't list them. Compare them only through the numbers printed when they were trained.
- The current best is world-20261102-152220 / driver-20261102-153924 (6 sessions, 88 inputs, the tuned defaults of 2026-11-02; numbers in learn/MODEL.md). The one before it, world-20261101-173810 / driver-20261101-175456 (the old settings), has a record that predates `test_score`: compare it by the per-head test results.
