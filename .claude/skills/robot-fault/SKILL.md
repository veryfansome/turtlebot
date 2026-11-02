---
name: robot-fault
description: Diagnose why the TurtleBot 4 isn't behaving - won't drive (base silent, e-stop on, an autonomous behaviour holding the wheels), lidar scans missing on the VM, camera not publishing - by reading, not fixing. Use when the robot ignores the controller or drive commands, when the lidar-lost alarm sounds or a recording lost /scan, or when robot-health shows a fault.
---

# Robot faults

Start with `bash ops/health.sh` (robot-health skill). Everything here only reads. On the Pi, `source /etc/turtlebot4/setup.bash` first. Traps: hidden base topics need `--include-hidden-topics`; `ros2 node list` shows nothing on the Pi; publisher counts need `--no-daemon --spin-time 5`.

## Won't drive

1. Is the base talking? `ssh turtlebot-pi 'source /etc/turtlebot4/setup.bash; timeout 8 ros2 topic echo --once /battery_state --field percentage'`. No number means the base is silent. After a restart or docking, wait about 5 minutes (its startup once took about 2.5 min at full CPU) before calling it dead.
2. Talking but ignoring commands: read the base's own log, GET only (the page's buttons reboot it): `ssh turtlebot-pi 'curl -s http://192.168.186.2/logs-raw' | grep -iE 'e-stop|ignoring velocities' | tail` (192.168.186.2 is the base's fixed address on its USB link to the Pi). `E-Stop request: Set system E-Stop ON` with no OFF after it means the e-stop is on (circle/B on the controller toggles it). `Ignoring velocities commanded while an autonomous behavior is running!` means a dock or wall-follow action holds the wheels until it ends. The log is in UTC.
3. Silent: save that log page before anyone restarts the base; it may record the cause. Restarting the base's application (its web page) is the user's step.
4. E-stop on: tell the user. `bash ops/field-test.sh estop off` only when they say it's safe. A reboot also clears it, which hides the cause.

## Lidar scans missing on the VM

Do these while it's happening, before stopping the run, and note the times. The cause is still open (ROADMAP.md, Known issues): record which case it was.

1. Does the Pi see `/scan`? `ssh turtlebot-pi 'source /etc/turtlebot4/setup.bash; timeout 10 ros2 topic hz /scan'`. No: a lidar or driver fault; check `journalctl -u turtlebot4 --since "-10 min"` on the Pi.
2. The Pi sees it, the VM doesn't: test large packets. From the VM, `ping -c 20 -s 3000 <the Pi's address>` against `-s 1400`; on the Pi, `journalctl -k | grep brcmfmac`. Loss only at 3000 means the Pi's Wi-Fi is dropping split packets (a scan is about 9 KB) while small topics get through. A Pi reboot fixed this once.
3. Large pings pass: check whether the VM still sees the lidar's publisher: `ros2 topic info /scan --verbose --no-daemon --spin-time 5` and `timeout 10 ros2 topic hz /scan` on the VM.
4. Report the case with its evidence lines and times. Restarting `turtlebot4` needs the user's sudo.

## Camera not publishing undocked

`ssh turtlebot-pi 'journalctl -u turtlebot4 --since "-10 min" | grep -E "\[oakd\]|OAKD"'`: `start_camera ... sending request` with no `Starting camera.` after it is the known stuck start. `ssh turtlebot-pi 'lsusb | grep 03e7'`: `2485` means not started, `f63b` started. The fix is `sudo systemctl restart turtlebot4`, typed by the user; `Camera ready!` appears about 40 s later.

## Rules

- Report findings with evidence lines. Don't restart, reboot or change anything yourself.
- Never press or suggest circle (B) to test anything.
- When a cause is confirmed, offer to update this skill and ROADMAP.md's Known issues.
