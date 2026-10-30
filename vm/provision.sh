#!/usr/bin/env bash
# Runs once inside the guest on first boot (started by cloud-init from create-vm.sh).
# Installs a desktop, UTM guest tools, ROS 2 Jazzy and the TurtleBot 4 desktop packages.
# Safe to re-run by hand if it fails: sudo bash /usr/local/sbin/provision.sh <user>
set -euxo pipefail

user=$1
export DEBIAN_FRONTEND=noninteractive
echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries

# The network-manager package makes NetworkManager the default netplan renderer, which takes the
# interface away from systemd-networkd mid-install and drops the connection. Keep networkd in
# charge until provisioning is done; NetworkManager takes over on the next boot.
netplan_override=/etc/netplan/90-provision-networkd.yaml
printf 'network:\n  version: 2\n  renderer: networkd\n' > "$netplan_override"
chmod 600 "$netplan_override"

apt-get update
apt-get install -y ubuntu-desktop-minimal spice-vdagent qemu-guest-agent mesa-utils curl
systemctl set-default graphical.target

# ROS 2 apt source, as in https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
ros_apt_version=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F'"' '{print $4}')
codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
curl -fsSL -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_apt_version}/ros2-apt-source_${ros_apt_version}.${codename}_all.deb"
apt-get install -y /tmp/ros2-apt-source.deb

apt-get update
apt-get upgrade -y
# irobot-create-msgs: needed to read the Create 3 base's topics/services (wheel_status, e_stop, ...)
# turtlebot4-navigation: SLAM mapping and Nav2 launch files (turtlebot4_navigation slam.launch.py, nav2.launch.py)
apt-get install -y ros-jazzy-desktop ros-jazzy-turtlebot4-desktop ros-jazzy-irobot-create-msgs ros-jazzy-turtlebot4-navigation ros-dev-tools

bashrc=/home/$user/.bashrc
grep -qF /opt/ros/jazzy/setup.bash "$bashrc" || echo 'source /opt/ros/jazzy/setup.bash' >> "$bashrc"
# Must match the robot's /etc/turtlebot4/setup.bash for ROS 2 discovery to find it.
grep -qF ROS_DOMAIN_ID "$bashrc" || cat >> "$bashrc" <<'EOF'
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF

rm "$netplan_override"
# The seed disk is detached after this run; don't let cloud-init reconfigure anything later.
touch /etc/cloud/cloud-init.disabled

# create-vm.sh waits for the VM to power off; cloud-init only powers off once this exists.
touch /var/lib/provision.done
