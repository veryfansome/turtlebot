---
name: verify-changes
description: The canonical check after changing code in this repo, before committing or relying on it - tests, a real run, an independent review with each finding verified, docs, and a report. Use after any code change in learn/, src/ or ops/, and before a commit the user asks for.
---

# Verifying a code change

Work through these steps in order and report each. A change counts as done only when every step has run or has been explicitly skipped with a reason.

1. **Scope.** `git status` and `git diff`, untracked files included. State what changed and what it's meant to do. That's what the review will check against.
2. **Tests.** Add tests for new behaviour and for every bug fixed (one that fails without the fix). Then run, from the repo root (`cd "$(git rev-parse --show-toplevel)"` first if unsure):
   - learn: `(cd learn && uv run pytest -q)`
   - pet_drive's pure-Python tests on the Mac: `PYTHONPATH=src/pet_drive uv run --no-project --with pytest python -m pytest -q src/pet_drive/test`
   - ROS packages (pet_drive, pet_explorer) build and test on the VM, from pushed code: the vm-sync skill, after a commit the user asked for.
3. **Run it for real.** Tests use synthetic data; real data finds what they miss.
   - learn: a short run on the real sessions, `(cd learn && uv run python -m pet_learn.train_world --iters 200)` then `(cd learn && uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt --iters 200 --goals 300)`. Results are meaningless; crashes and NaN are what count. Both save a model and record: name them to the user.
   - Converter changes: reconvert (`--force` if the format changed) and read the summaries.
   - Robot-side code (src/, ops/): the VM build and tests; on the robot only through the field-test skill, with the user's go for that run.
4. **Review.** An independent review of the whole diff, not anchored on the intended design: `/code-review` at `high`, or reviewers with different lenses (fixes complete, correctness, devices/numerics, data edge cases, whether docs match the code). Then try to refute each finding: give it to a separate checker told to disprove it from the code and, where possible, a small check on the real data. Keep only what survives, with its real severity.
5. **Fix** what was confirmed, or say why not (for example, the cost is negligible). Each fix gets a test where one can catch it. Rerun steps 2–3. If the fixes were substantial, review them too (step 4, on the new diff).
6. **Docs.** The READMEs, `learn/MODEL.md`, `STEP_FIELDS` and docstrings must match what the code now does. A reviewer should check this too.
7. **Report.** What ran and its results (test counts, the run's outcome), findings fixed and skipped (with reasons), anything left unverified, and what the user needs to do (such as a VM sync). Changes stay uncommitted until the user asks for a commit.
