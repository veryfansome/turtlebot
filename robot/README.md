# Robot-side setup

Changes made on the TurtleBot 4's Raspberry Pi, kept here so they can be re-applied after a reinstall or a TurtleBot 4 package update. Copy a script over and run it there, e.g. from the VM:

```bash
scp ~/turtlebot/robot/enable-create3-relays.sh ubuntu@<robot-ip>:
ssh ubuntu@<robot-ip> sudo bash enable-create3-relays.sh
ssh ubuntu@<robot-ip> sudo systemctl restart turtlebot4
```

- `enable-create3-relays.sh`: the Pi only relays the Create 3 base topics listed in `turtlebot4_bringup/config/republisher.yaml`. This turns on `hazard_detection` (bumps, used by pet_explorer) and `kidnap_status` (the robot was picked up), which are off by default.

Set on the Create 3 base itself (its web page, Application → Configuration), not by a script:

- `safety_override: "backup_only"` on `motion_control`, so the robot can reverse more than about an inch. See the comments in that config for the options. With the base's namespace (`/_do_not_use`), the key may need to be `/**/motion_control:` for it to apply; check with `ros2 service call /_do_not_use/motion_control/get_parameters rcl_interfaces/srv/GetParameters "{names: ['safety_override']}"` on the Pi after a restart.
