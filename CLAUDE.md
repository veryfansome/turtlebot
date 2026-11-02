# turtlebot: notes for Claude

A TurtleBot 4 "robot pet" (ROS 2 Jazzy; Create 3 base, RPLIDAR, OAK-D Pro camera, Raspberry Pi 4) learning to drive itself: a world model plus an IQL driver, trained offline on recorded driving, with every drive command passing through a hand-written safety layer. Plan, status, next steps and known issues: [ROADMAP.md](ROADMAP.md) (start with its "Next steps"). The repo root is a colcon workspace (the ROS 2 package is under `src/`). This repo is public: name machines by their SSH aliases, and never put home-network addresses, passwords or tokens in it.

## Where things run

- **Pi** (`ssh turtlebot-pi`): the robot's own software (the `turtlebot4` service). This repo changes it only through ops units ([ops/README.md](ops/README.md)).
- **VM** (`ssh turtlebot-vm`, Ubuntu 24.04 on the Mac, [vm/README.md](vm/README.md)): the ROS package in `src/` (the safety layer, the pickup node, the recorder), field tests, bags in `~/drives/`. It runs only pushed commits, rebuilt by the vm-sync skill.
- **Mac**: `learn/` (a uv project, torch on the Apple GPU), and the `ops/` and `cloud/` scripts, run from the repo root with `bash`.
- **RunPod**: world-model training waves through `cloud/` (built, never launched yet).
- `data/` (drives, steps, models) is git-ignored. Converted steps are also on the private Hugging Face dataset `veryfansome/turtlebot-steps` (pinned in `learn/data_source.json`). The evolve campaign lives in the private repo `veryfansome/turtlebot-evolve`.

## Branches and commits

- Work on `dev`; PRs go from `dev` to `main`; the owner merges. Commit, push or open a PR only when the owner asks (ship-changes skill).
- "Sync main and rebranch", after a merge, on `dev`: `git fetch origin main:main && git merge --ff-only main && git push origin dev`. Never check out `main` to do it.
- Never `git stash`, `checkout`, `switch`, `reset`, `rebase`, `merge` or `pull` (the sync above included) while a training runs (`ps aux | grep '[p]et_learn'`): it loads code from disk.
- Stage files by name, never a folder or `-A`, and read `git diff --cached --stat` before committing. A stray 17 MB pickle once had to be squashed out of pushed history.

## Hard rules

- **Robot and VM:** change them only through ops units (push-config skill), never with hand-typed edits over `ssh`. Network and Wi-Fi, firmware, `apt upgrade`, boot config and the Create 3's web config stay manual: write the steps for the owner.
- **Sudo on the Pi:** only the owner types the password. When `ops/run.sh` exits 3, give them the printed command to run in their own terminal.
- **Field tests:** start one only with the owner's go for that run (field-test skill). Stopping, and `ops/field-test.sh estop on`, never need permission; `estop off` only when the owner says it's safe. Never press or suggest circle (B) on the controller to test anything: it toggles the base's e-stop.
- **Money:** never launch a RunPod pod without the owner's go for that wave, given after the printed cost estimate (cloud-wave skill). Never terminate a pod you didn't start without the owner's say-so.
- **Data:** never delete recordings, steps or models without asking.

## Checking work

Every code change goes through the verify-changes skill before it counts as done: tests, a real run on real data, an independent review with each finding checked against the code, fixes, docs updated, a report. Training goes through the train-model skill: plain defaults (change a default rather than pass a flag every time), on every session, new recordings included.

## Skills (`.claude/skills/`)

| Skill | For |
|---|---|
| robot-health | Read-only check of the robot and VM; before and after any change |
| robot-fault | Why the robot won't drive, lidar scans missing on the VM, the camera not publishing |
| push-config | Apply, revert or write an ops unit on the Pi or VM |
| vm-sync | Pull, build and test the VM's checkout |
| field-test | Record a driving session, watch it, stop it |
| train-model | Convert, train the world model then the driver, compare run records |
| add-sensor | One new recorded signal, end to end |
| verify-changes | The check after any code change |
| ship-changes | Commit, push, PR, and sync after a merge |
| cloud-wave | A grid, the speed probe or evolve candidates on RunPod |
| evolve-runpod-round | The evolve search here, scored on RunPod (instead of plain `/evolve:round` scoring) |

## Writing

- Markdown: one line per paragraph or list item, no hard wraps.
- Plain words, for a reader with no context. Absolute dates. Say what was checked and what wasn't ("built, not yet tried on the robot"). A guess is labelled a guess.
- The docs must match the code; verify-changes checks this.
- The Mac's shell is zsh, which doesn't word-split an unquoted `$var`: run `ops/` and `cloud/` scripts with `bash`, and write loops over words in bash.
