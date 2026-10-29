# Ubuntu VM for the TurtleBot 4

The robot runs Ubuntu 24.04 / ROS 2 Jazzy, so the Mac-side tools (rviz, teleop, Nav2 clients) run in an Ubuntu 24.04 arm64 VM in [UTM](https://mac.getutm.app/) with bridged networking.

## Creating the VM

Prerequisites (once):

```bash
brew install --cask utm
brew install qemu          # qemu-img, to grow the cloud image before UTM imports it
```

Then:

```bash
vm/create-vm.sh
```

It prompts for the guest password, then creates a UTM VM named `turtlebot-jazzy` (6 CPUs, 16 GiB RAM, 64 GiB disk, bridged networking, 3D-accelerated display), installs a GNOME desktop, ROS 2 Jazzy and `ros-jazzy-turtlebot4-desktop` via `provision.sh`, and leaves it shut down. Provisioning took about 6 minutes on a fast connection; the VM powers itself off when it's finished. Then start it from the UTM window (select it, press ▶).

Overrides via environment variables: `VM_NAME`, `VM_USER` (defaults to your macOS username), `CPUS`, `MEMORY_MIB`, `DISK_MIB`, `BRIDGE_IF` (defaults to the interface with the default route), `SSH_PUBKEY_FILE`, `VM_PASSWORD`. See the top of `create-vm.sh`.

Notes:

- The first run asks macOS to let your terminal control UTM (System Settings > Privacy & Security > Automation). The script stops with a message if that was denied.
- Bridged networking puts the VM on whatever network `BRIDGE_IF` is joined to. The robot must be on that same network (e.g. robot in Wi-Fi client mode on your home network).
- The cloud image is pinned to one build and cached in `~/Library/Caches/turtlebot-vm/`. To update, change `IMAGE_SERIAL` / `IMAGE_SHA256` in `create-vm.sh`.
- If provisioning fails, the VM stays running; log in on its console and check `/var/log/cloud-init-output.log`. `sudo bash /usr/local/sbin/provision.sh <user>` re-runs it.
- With GPU acceleration on, UTM can't suspend the VM, so closing its window force-kills it. Shut down from inside Ubuntu instead.
- `provision.sh` sets `ROS_DOMAIN_ID=0` and `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` in `~/.bashrc`, matching the robot's `/etc/turtlebot4/setup.bash` (simple discovery, no discovery server). If you change those on the robot, change them in the VM too.

## Install media

Manual-install ISO (downloaded to `~/Downloads/ubuntu-24.04.5-desktop-arm64.iso`):

- URL: https://cdimage.ubuntu.com/ubuntu/releases/noble/release/ubuntu-24.04.5-desktop-arm64.iso
- SHA-256: `2be09ca883921bff6d8e6b0bfbafd13e32436553b7086f33bce3a4c5bad8bd14`

`create-vm.sh` does not use this ISO. It boots the Ubuntu 24.04 arm64 cloud image (https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img), which can be provisioned unattended. The ISO is only needed for a by-hand install through the UTM wizard.
