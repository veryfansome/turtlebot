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

`convert` turns every session in `data/drives/` that hasn't been converted yet into `data/steps/<session>.npz` and `<session>.maps.npz` (every SLAM map, about one a second) (`--force` redoes them all), and prints a summary of each: how far it drove, bumps by side, time spent told to go forward without moving, wheel slip, area mapped, battery. Both folders are git-ignored. New recordings are just more sessions: pull, convert, and training picks them up.

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
| `ir` | (n, 7) | front infrared proximity readings, left to right (higher is closer); nan in sessions recorded before they were relayed |
| `slipping`, `stall`, `cliff_or_drop` | (n,) | the base's wheel-slip detection; wheel stall, cliff or wheel drop reported during the step |
| `bump` | (n, 3) | a bump began during the step: left, centre, right |
| `kidnapped`, `docked`, `battery` | (n,) | picked up, on the dock, battery fraction |
| `pose` | (n, 3) | SLAM pose: x, y (m), yaw (rad); nan before SLAM has one |
| `known_cells`, `new_cells` | (n,) | mapped cells in SLAM's map, and how many became mapped during the step |

Told to go forward while the floor sensor says it isn't moving is the "pushing" signal; the base reports a bump only at the moment of contact, so pushing afterwards shows up only this way.

## Goals, the reward, and the report

Learning to drive means reaching a goal pose (position and heading) efficiently; see the [roadmap](../ROADMAP.md).

- [pet_learn/geodesic.py](pet_learn/geodesic.py): shortest-path distance to a goal through the SLAM map: mapped free space only, at least 12 cm from walls (a bit less than the robot's radius: SLAM's walls are fuzzy). The straight line would reward pushing into the couch to reach a goal behind it.
- [pet_learn/reward.py](pet_learn/reward.py): each step earns its progress (the drop in shortest-path distance, and near the goal in heading error, as a potential, so loops can't farm it), minus 0.1 per second and 0.1 per amp-second of motor current, plus 1.0 on arriving within 20 cm and 20 degrees. In the first session, normal driving drew about 0.19 A and pushing against something about 0.42 A.
- [pet_learn/episodes.py](pet_learn/episodes.py): cuts sessions into episodes after the fact (hindsight relabeling): a random start, and the pose the robot had 3–60 s later as the goal. Steps without a SLAM pose, docked, picked up, or where the pose jumped (a SLAM reset) are left out.
- [pet_learn/report.py](pet_learn/report.py): `uv run python -m pet_learn.report` prints, per session, success rate, SPL (success weighted by shortest path over path driven: 1.0 is perfect), time and energy per metre of shortest path, reward, bumps per minute, time stuck and rescues. A report, not a ranking.

First session (controller, 300 hindsight episodes): SPL 0.48, 19 s and 4.1 A·s per metre of shortest path, reward −1.41 per episode. That's expected: a person driving around wasn't heading for a point a random 3–60 s later, so most episodes meander. The direct ones (SPL ≥ 0.8, 22 of 300) earn +0.84 on average, the least direct (SPL < 0.3) −2.96. Recorded driving bootstraps the learner rather than being copied: meandering episodes still teach it what things are worth, because they're scored.

## The driving model

A world model with goal-reaching values and a policy on a shared state, trained offline: see [MODEL.md](MODEL.md) for the design and first results.

```bash
uv run python -m pet_learn.train_world                       # world model; prints held-out error vs baselines
uv run python -m pet_learn.train_driver --world ../data/models/world-<time>.pt   # values and policy on top
```

Models are saved to `data/models/` (git-ignored).

## Tests

```bash
uv run pytest
```
