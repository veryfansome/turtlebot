---
name: push-config
description: Apply, revert or check a reversible config change ("unit") on the TurtleBot 4's Pi or the Ubuntu VM through ops/run.sh. Use whenever a change to a file or service on the Pi or VM is needed, including writing a new unit. Never change those machines by hand-typed commands.
---

# Pushing config to the Pi or the VM

Every change to the robot or the VM is a unit under `ops/pi/` or `ops/vm/`, applied with `ops/run.sh`. Read `ops/README.md` first.

## Applying or reverting a unit

1. **Health first.** Run the robot-health skill (`ops/health.sh`). If something unrelated is broken, stop and tell the user before changing anything.
2. **Status and diff.** `ops/run.sh <pi|vm> <unit> status`, then `ops/run.sh <pi|vm> <unit> diff`. Both are read-only. Show the user the diff.
3. **Ask.** Say what will change, what gets restarted (e.g. restarting `turtlebot4` also restarts the camera and lidar, and stops the robot if it's driving), and how to undo it. Wait for a yes.
4. **Apply.** `ops/run.sh <pi|vm> <unit> apply`. On the Pi this needs the sudo password, so it stops with exit code 3 and prints the command. Then ask the user to run that command in their own terminal (commands run with `!` in Claude Code can't answer the sudo password prompt) and tell you when it's done.
5. **Check.** `ops/run.sh <pi|vm> <unit> status` should say `applied`; then run `ops/health.sh` again and compare with step 1.
6. **If it went wrong.** The unit restores the previous file by itself when its verification fails. Otherwise, `revert` is the undo. As a last resort, the untouched original is in `/var/lib/turtlebot-ops/backups/<unit>/` on the target, and `/var/lib/turtlebot-ops/changes.log` has the history.

Revert follows the same steps, with `revert` in place of `apply`.

## Writing a new unit

- Copy `ops/pi/create3-relays.sh`. One unit changes one file. Define `UNIT`, `TARGET_FILE`, `unit_apply_transform` and `unit_revert_transform` (stdin to stdout; each a no-op on its own output, and revert(apply(x)) == x for the original file), `unit_after_change` (what to restart) and `unit_verify_apply` / `unit_verify_revert` (return non-zero to trigger the automatic rollback).
- Test the transforms on the real file before anyone applies the unit: copy the target file to the scratchpad with `scp` and check apply(original) gives the intended file, revert(applied) gives the original byte for byte, and both are no-ops on their own output.
- Add the unit to the table in `ops/README.md`.

## Never

- Don't change the Pi's network or Wi-Fi settings, update firmware (Pi or Create 3), run `apt upgrade` on the Pi, touch partitions or boot config, or edit the Create 3's web config. Those stay manual: write down the steps for the user instead.
- Don't edit files on the Pi or VM directly with `ssh ... sed`/`tee`/editors, even for a quick test. Make a unit.
- Don't apply or revert without the user's yes in this conversation, even if they approved a similar change before.
