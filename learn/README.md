# learn

Learning to drive the TurtleBot 4 (stage 2 of the [roadmap](../ROADMAP.md)), on the Mac. A plain Python project, separate from the ROS packages under `src/`: it reads the recorded bags directly, so the Mac doesn't need ROS. Managed with [uv](https://docs.astral.sh/uv/).

## From a recording to training steps

```bash
ops/field-test.sh start record    # record a session on the VM (see src/pet_drive)
ops/field-test.sh stop
ops/pull-drives.sh                # copy it to data/drives/ on the Mac
cd learn
uv run python -m pet_learn.convert
```

`convert` turns every session in `data/drives/` that hasn't been converted yet into `data/steps/<session>.npz` (`--force` redoes them all), and prints a summary of each: how far it drove, bumps by side, time spent told to go forward without moving, wheel slip, area mapped, battery. Both folders are git-ignored. New recordings are just more sessions: pull, convert, and training picks them up.

Each steps file holds one row per step at 10 Hz, aligned on the time the VM received each message. The fields (listed in `STEP_FIELDS` in [pet_learn/convert.py](pet_learn/convert.py)):

| Field | Shape | What |
|---|---|---|
| `t` | (n,) | seconds since the first scan |
| `scan` | (n, 60) | lidar: nearest return in each of 60 equal sectors (m), starting at `scan_angle_min` |
| `cmd` | (n, 2) | what the wheels were told: forward m/s, turn rad/s |
| `joy_active` | (n,) | a person was driving (a controller button held) |
| `odom_vel` | (n, 2) | wheel odometry: forward m/s, turn rad/s |
| `wheel_vels`, `wheel_current` | (n, 2) | wheel speeds (rad/s) and motor currents (mA), left and right |
| `mouse_speed`, `mouse_quality` | (n,) | speed over the floor from the optical floor sensor (m/s), and its tracking quality |
| `slipping`, `stall`, `cliff_or_drop` | (n,) | the base's wheel-slip detection; wheel stall, cliff or wheel drop reported during the step |
| `bump` | (n, 3) | a bump began during the step: left, centre, right |
| `kidnapped`, `docked`, `battery` | (n,) | picked up, on the dock, battery fraction |
| `pose` | (n, 3) | SLAM pose: x, y (m), yaw (rad); nan before SLAM has one |
| `known_cells`, `new_cells` | (n,) | mapped cells in SLAM's map, and how many became mapped during the step |

Told to go forward while the floor sensor says it isn't moving is the "pushing" signal; the base reports a bump only at the moment of contact, so pushing afterwards shows up only this way.

## Tests

```bash
uv run pytest
```
