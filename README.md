# turtlebot

A TurtleBot 4 (ROS 2 Jazzy: iRobot Create 3 base, RPLIDAR, OAK-D Pro camera, Raspberry Pi 4) on its way to becoming a robot pet that explores the house on its own. It's learning to drive itself: a world model plus an IQL driver, trained offline on recorded driving, with every drive command passing through a hand-written safety layer. The plan, what's built and what's next are in [ROADMAP.md](ROADMAP.md).

## Where things run

- **The robot** (Pi 4 + Create 3): only the TurtleBot 4's own software, changed only through reversible scripts in [ops/](ops/README.md).
- **An Ubuntu 24.04 VM on the Mac** ([vm/](vm/README.md)): SLAM, the safety layer, the pickup node and the recorder. It builds this repo as a colcon workspace (the ROS 2 package is under `src/`).
- **The Mac**: the ops scripts, and training ([learn/](learn/README.md): uv, PyTorch on the Apple GPU).
- **RunPod** (optional, costs money): batches of world-model trainings ([cloud/](cloud/README.md)).

## Map of the repo

| Path | What |
|---|---|
| [src/pet_drive](src/pet_drive/README.md) | The safety layer every drive command goes through (stops, tones, calls for help), the pickup node, and the launch file that records driving sessions |
| [config/nav2.yaml](config/nav2.yaml) | The Nav2 settings the old frontier explorer used (removed 2026-11-01, in git history), kept for comparing the learned driver against Nav2 |
| [ops/](ops/README.md) | Run from the Mac: health check, field tests, copying recordings home, syncing the VM, reversible config units |
| [vm/](vm/README.md) | Creating and provisioning the VM, SSH setup |
| [learn/](learn/README.md) | Recordings to 10 Hz training steps, reward and session report, the world model and IQL driver ([MODEL.md](learn/MODEL.md)), run records |
| [learn/eval](learn/eval), [learn/evolve](learn/evolve/EVOLVE.md) | The evolutionary search over the world model's training: scoring contract and config |
| [cloud/](cloud/README.md) | Training waves on RunPod, with the pod always confirmed stopped |
| [.claude/skills](.claude/skills) | Step-by-step procedures for Claude Code ([CLAUDE.md](CLAUDE.md) lists them) |
| `data/` (git-ignored) | `drives/` (bags), `steps/` (converted sessions), `models/` (checkpoints, each with a `.json` run record) |

## Kept outside this repo

- Converted sessions: the private Hugging Face dataset `veryfansome/turtlebot-steps`, pinned in [learn/data_source.json](learn/data_source.json).
- The evolve campaign: the private repo `veryfansome/turtlebot-evolve` (`learn/evolve/campaign.py push|pull`).
- Secrets in `~/.runpod.env`; SSH hosts `turtlebot-pi` and `turtlebot-vm` in `~/.ssh/config`.

## The usual loop

```bash
bash ops/health.sh                      # is the robot OK?
bash ops/vm-sync.sh                     # the VM runs the pushed code
bash ops/field-test.sh start record     # drive with the controller...
bash ops/field-test.sh stop
bash ops/pull-drives.sh                 # bags to data/drives/
cd learn && uv run python -m pet_learn.convert
uv run python -m pet_learn.train_world  # then train_driver --world ../data/models/world-<time>.pt
uv run python -m pet_learn.runs         # compare with the current best
```

Work happens on branch `dev`, merged to `main` by pull request.
