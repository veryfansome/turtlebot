---
name: train-model
description: Convert recordings and train the driving model (learn/: world model, then driver), then compare the run against the current best using its run record. Use after a new recording, after changing code in learn/, or to compare training settings.
---

# Training the driving model

Everything runs on the Mac. Commands below are from the repo root; `learn/` is a uv project, so its commands run as `(cd learn && uv run ...)`, which works whatever the current folder was. See [learn/README.md](../../../learn/README.md) and [learn/MODEL.md](../../../learn/MODEL.md).

## Steps

1. **Data.** `ops/pull-drives.sh` (if it fails, say so: don't train on stale data without telling the user), then `(cd learn && uv run python -m pet_learn.convert)` (converts only new sessions). If the converter or the steps format changed, `--force` reconverts every session: say so, because results before and after aren't comparable. Read each session's summary (driving time, rescues, how long the lidar was missing) and tell the user anything odd.
2. **Train on every session.** Always all of `data/steps/`, new recordings included, unless the user says otherwise. Make sure nothing else is training first (`ps aux | grep pet_learn`): two runs share the CPU and GPU, which skews times.
   ```bash
   (cd learn && uv run python -m pet_learn.train_world)                                       # stops itself when val error stops improving
   (cd learn && uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt)   # the world model just trained
   ```
   Run them in the background (about 15–40 min each). Run them plainly: the defaults are the settings. If a setting needs to differ every time, change its default in the code rather than passing a flag, so no run can forget it. A flag is only for an experiment, and the report says which.
3. **Compare.** `(cd learn && uv run python -m pet_learn.runs)` lists every run's record (`data/models/<model>.json`), oldest first: code version, data, the settings that differ from the defaults, when and why it stopped, and test results. Compare against the current best and check the records first:
   - The same data (the `data` fingerprint: sessions and the exact steps files, so a reconversion shows), unless the new data is what's being tested.
   - The same code version (`commit+changes`, which covers only the training code, `learn/pet_learn` and its dependencies), unless the code is what's being tested. If the commits differ, `git diff <a> <b> -- learn/pet_learn` shows whether the training code did.
   - One difference at a time. If more than one differs, say so, since the comparison can't tell which one mattered.
4. **Report** the test results (the last 10% of each session's driving, never used to pick the model), in the same form every time: the world model's error relative to its baselines at 1 s per head, bump and slip AUC, the driver's value rank correlation, and the policy check (forward and turn for goals ahead, left, right, back-left, back-right and behind). Give the current best's numbers beside them, and say which model is now the best and why.

## Rules

- Never delete models or data without asking. A short smoke run (`--iters 200`) also saves a model and record: name them to the user.
- A world model trained before the inputs changed won't load into `train_driver` (it says so up front). Retrain the world model.
- Training defaults to the Mac's GPU (MPS). For timing, benchmark only while nothing else runs.
- Models without a `.json` beside them were saved before run records existed (up to world-20261101-152104 and driver-20261101-152320), and `pet_learn.runs` doesn't list them. Compare them only through the numbers printed when they were trained.
