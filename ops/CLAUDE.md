# ops/: notes for Claude

Scripts that touch the robot and the VM. Read [README.md](README.md); use the robot-health, robot-fault, push-config, vm-sync and field-test skills. Run them with `bash` from the repo root.

- Read-only, safe without asking: `health.sh`, `run.sh ... status|diff`, `field-test.sh status|logs`, `pull-drives.sh` (never deletes on either side).
- Need the owner's yes in this conversation: `run.sh ... apply|revert`, `field-test.sh start`. Always allowed: `field-test.sh stop` and `estop on`. Only on the owner's say-so: `estop off`.
- A unit changes one file: copy `pi/create3-relays.sh`, define the transforms (revert(apply(x)) == x, each a no-op on its own output) and the verify hooks, test the transforms on a copy of the real file before anyone applies it, and add it to the table in README.md. `ops/vm/` has no units yet.
- Pi sudo: `run.sh` exits 3 and prints the command; the owner runs it in their own terminal (a `!` command in Claude Code can't answer the sudo prompt).
- Never script the Pi's network or Wi-Fi, firmware, `apt upgrade`, partitions or boot config, or the Create 3's web config.
- Machine addresses live in `~/.ssh/config` (`turtlebot-pi`, `turtlebot-vm`), not in this repo (it's public).
