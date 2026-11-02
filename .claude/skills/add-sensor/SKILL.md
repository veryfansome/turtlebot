---
name: add-sensor
description: Add a recorded signal to the driving data end to end - relay it from the base if needed, record it, convert it into steps, and optionally make it a world-model input or target - keeping docs, tests, the evolve contract and the data pin consistent. Use when the user wants the robot to record a new topic or the model to use a new reading (IR, wheels_enabled, distress and slam_reset were added this way; camera features are next).
---

# Adding a sensor end to end

One layer at a time; each goes through verify-changes and ships as its own commit.

1. **Reach the VM.** A base topic needs relaying on the Pi: edit the `pi/create3-relays` unit's list (`ops/pi/create3-relays.sh`) and the ops/README.md table, applied through push-config with the user's go. Check with `ros2 topic hz` on the VM. Big messages (camera) risk starving `/scan` over Wi-Fi: measure bandwidth before recording them by default.
2. **Record it.** Add it to `TOPICS` in `src/pet_drive/launch/record.launch.py` (behind a launch argument if heavy) and to the topic table in src/pet_drive/README.md. vm-sync, then one short recording (field-test skill) to see it in the bag.
3. **Convert it.** `learn/pet_learn/convert.py`: read the topic, add the field to `STEP_FIELDS` with a plain description, a summary count if it's an event, a test in `learn/tests/test_convert.py`, a line in learn/README.md. Older sessions lack the topic: the field must come out missing, never as real zeros.
4. **Model input or target (optional).** Columns in `dataset.INPUT_COLUMNS`, with a real-or-missing flag and the masking and random-hiding rules; tests in `learn/tests/test_dataset.py`; update learn/MODEL.md and the input count in `jail_notes` in `learn/evolve/evolve.json` (88 now). These are protected files: harness maintenance (verify-changes). Old world models won't load into train_driver.
5. **Reconvert and retrain.** `(cd learn && uv run python -m pet_learn.convert --force)`, read every summary, then train-model on all sessions against the previous best (expect new data fingerprints).
6. **The searched data.** A reconversion changes every session's fingerprint, so the evolve search's preflight refuses them until the search is re-pinned. Ask the user before `eval.manifest --write` (a new measurement epoch) and before `pet_learn.hub push` plus committing `learn/data_source.json`: the two change together.
