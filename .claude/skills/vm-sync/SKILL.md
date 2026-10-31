---
name: vm-sync
description: Update the Ubuntu VM's checkout of this repo from GitHub, rebuild it with colcon and run the unit tests. Use after pushing code the robot should run, before a field test, or when the VM's build looks stale.
---

# Syncing the VM

```bash
ops/vm-sync.sh            # stay on the VM's current branch
ops/vm-sync.sh <branch>   # switch branch first
```

It runs on the VM over SSH (`turtlebot-vm`): fetch, fast-forward pull, `colcon build --symlink-install`, then the unit tests. Safe to run without asking; it only touches the VM's `~/turtlebot` checkout and build.

- Only pushed commits reach the VM. Commit and push first (with the user's go-ahead, as for any push).
- It refuses if the VM's checkout has local changes (the user may be experimenting there). Tell the user, don't discard them.
- It refuses while a field test is running, so a rebuild can't happen under a running robot.
- Finish by reporting the commit the VM is on and the test summary line. A failed build or test means the robot shouldn't run that code: say so before any field test.
- Running nodes don't pick up a rebuild. A running field test needs `ops/field-test.sh stop` and `start` (with the user's go-ahead).
