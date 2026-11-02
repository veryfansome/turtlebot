# src/: notes for Claude

The ROS 2 Jazzy package `pet_drive` ([README](pet_drive/README.md)), built and run on the VM, never on the Mac or the Pi.

- The safety layer is the only path from a driver to `cmd_vel` (rules in `safety.py`, free of ROS); a driver publishes on `drive/cmd_vel`. Its tones, light ring, lidar alarm and calls for help are in `feedback.py`; the pickup node (pauses and resets SLAM, `/drive/slam_reset`) in `pickup.py`/`pickup_node.py`; the bag's topics in `launch/record.launch.py`.
- Keep the logic in ROS-free modules with unit tests. On the Mac: `PYTHONPATH=src/pet_drive uv run --no-project --with pytest python -m pytest -q src/pet_drive/test`. The full build and tests run on the VM (vm-sync skill), from pushed code.
- A new recorded topic: the add-sensor skill (the record list, the converter, the README's topic table, a relay unit if it comes from the base).
- On the robot only through the field-test skill, with the owner's go for that run. Built 2026-11-01, not yet tried on the robot: the start tone, the rescue beeps, the lidar-lost alarm, the call for help (`/drive/distress`) and the pickup node's reset (`/drive/slam_reset`).
- The Nav2 frontier explorer (`src/pet_explorer`) was removed on 2026-11-01; its frontier finder, the planned goal picker for stage 3 of the roadmap, is in git history (last at commit 374636d).
