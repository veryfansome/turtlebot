---
name: field-test
description: Record a driving session on the robot through ops/field-test.sh (SLAM, the safety layer, the pickup node and a bag of everything a driver could learn from), watch it, and stop it. Use when the user wants to record driving or try robot-side code (src/pet_drive) on the robot. Starting needs the user's explicit go in the current conversation; stopping and the e-stop never need permission.
---

# Field test

`ops/field-test.sh` controls a background run on the VM. Logs are in `~/field-tests/` on the VM, one per run. A run (`start record`) is SLAM, the safety layer, the pickup node and `ros2 bag record` into `~/drives/<time>/` (see [src/pet_drive](../../../src/pet_drive/README.md)). It drives nothing itself: the user drives with the controller, or a driver publishes on `drive/cmd_vel`. This is how training data gets collected. Run the ops scripts with `bash` from the repo root.

## Before starting

1. `bash ops/vm-sync.sh`, so the VM runs the code you mean to test, with tests passing.
2. `bash ops/health.sh` (robot-health skill). Don't start if the base is silent, the lidar isn't publishing, or `safety_override` is wrong. A recording doesn't use the camera, so a stuck camera alone doesn't block a run.
3. **Ask the user for a go,** and say what will run (branch and commit, launch arguments). Their go means they're watching the robot, stairs and drop-offs are blocked, cables and fragile things are out of the way, and the robot is undocked. A go is for one run: ask again before restarting it, even if they approved the last one.

## Running

```bash
bash ops/field-test.sh start record [args]   # e.g. camera:=true (raw frames, heavy on Wi-Fi)
bash ops/field-test.sh status                # running or not, the last safety-layer and pickup lines, the bag's size
bash ops/field-test.sh logs [lines]          # the end of the full log
bash ops/pull-drives.sh                      # after recording: copy bags to data/drives/ on the Mac
```

- `start` checks first and refuses if the base is silent, the battery is under 25%, the robot is docked or picked up, or there's no lidar data.
- While it runs, check `status` every minute or so and tell the user that the bag is growing, and about any safety-layer stops, calls for help or lidar alarms.
- For RViz, the user runs `ros2 launch turtlebot4_viz view_navigation.launch.py` in a terminal in the VM's own desktop.

## Stopping

```bash
bash ops/field-test.sh save-map      # save the SLAM map to ~/maps/<run>/ (stop does this too)
bash ops/field-test.sh stop          # normal stop: saves the map, then everything exits and the bag closes
bash ops/field-test.sh estop on      # emergency: cuts the drive motors' power at the base
bash ops/field-test.sh estop off     # restore motor power (only when the user says it's safe)
```

- Stop without asking when the user says stop, when something looks unsafe (repeated bumps, kidnapped, battery under 20%), or when you're about to end your turn without the user watching.
- Use `estop on` when a normal stop isn't enough or the robot is doing something dangerous; then `stop` as well. Never turn the e-stop off without the user's say-so. Never test the e-stop with circle (B) on the controller: it toggles the base's e-stop.
- Don't leave a run going silently: if it's still running when you reply, say so.
- If the user rejects a `stop` or `estop` you sent, it may already have run: check `bash ops/field-test.sh status` right away and tell them what state the run is in.
- A normal stop logs launch's Ctrl-C handling. If `stop` printed "no clean exit after Ctrl-C", the clean shutdown didn't happen: say so.

## Pickups

The pickup node logs as `[pickup]`. When the user lifts the robot and sets it down, expect `Picked up (#n): pausing SLAM`, `SLAM paused`, `Set down: resetting SLAM once it settles`, about 1 s later `SLAM reset after pickup #n: a fresh map`, then `SLAM resumed`. A `failed`, `not available`, `retrying` or `isn't following its toggles` line means the map may not have been reset: say so.

## Checking the robot-side features (built 2026-11-01, none field-tested yet)

One recording, with the user's go, can check them all; ask the user what they heard.

- Start tone: one long note a few seconds after `start`.
- Rescue: Y (triangle) before freeing a stuck robot: two low beeps.
- Lidar alarm: only on a real dropout (don't fake one): the lidar-silent tones every 20 s while undocked, two blips when scans are back. If it sounds, ask the user before stopping, and run the robot-fault skill's lidar checks first: the cause of the dropouts is still open.
- Call for help: a real stall, or the user holds the robot against something while driving it forward: a two-note siren within about 5 s.
- Pickup: one lift and set-down (above).
- Afterwards: `bash ops/pull-drives.sh`, then `(cd learn && uv run python -m pet_learn.convert)`. The session summary's counts of rescues, SLAM resets and calls for help must match what happened; a zero where an event happened means the topic wasn't recorded.

## After

Summarize the run from `bash ops/field-test.sh logs` and `status`: duration, stops, calls for help, pickups, anything new. Then `bash ops/health.sh` again. The camera in particular often needs a `turtlebot4` restart after docking and undocking.
