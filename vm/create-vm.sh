#!/usr/bin/env bash
# Create an Ubuntu 24.04 + ROS 2 Jazzy desktop VM in UTM for working with the TurtleBot 4.
#
# Boots the official Ubuntu arm64 cloud image with a cloud-init seed that runs provision.sh,
# waits for provisioning to finish (the guest powers itself off) and detaches the seed.
# Networking is bridged so ROS 2 discovery reaches the robot.
#
# Usage: vm/create-vm.sh            (prompts for the guest password)
# Settings can be overridden with environment variables, see below.
set -euo pipefail

VM_NAME=${VM_NAME:-turtlebot-jazzy}
VM_USER=${VM_USER:-$(id -un)}
CPUS=${CPUS:-6}
MEMORY_MIB=${MEMORY_MIB:-16384}
DISK_MIB=${DISK_MIB:-65536}
# Host interface to bridge to; defaults to the one carrying the default route (e.g. Wi-Fi).
BRIDGE_IF=${BRIDGE_IF:-$(route -n get default | awk '/interface:/ {print $2}')}
# Optional public key to authorize for SSH into the guest.
SSH_PUBKEY_FILE=${SSH_PUBKEY_FILE:-}
# Set to skip the password prompt.
VM_PASSWORD=${VM_PASSWORD:-}
PROVISION_TIMEOUT_MIN=${PROVISION_TIMEOUT_MIN:-90}
CACHE_DIR=${CACHE_DIR:-$HOME/Library/Caches/turtlebot-vm}

# Pinned cloud image build so every VM starts from the same base and the download is cached.
# To move to a newer build, pick a serial from https://cloud-images.ubuntu.com/noble/ and copy
# the arm64 .img line from that build's SHA256SUMS.
IMAGE_SERIAL=20261017
IMAGE_SHA256=50e24e1fb79e89116e939d90aa7738f7af48c442fa1bc2510131e0b39c50348e
IMAGE_NAME=noble-server-cloudimg-arm64.img
IMAGE_URL=https://cloud-images.ubuntu.com/noble/$IMAGE_SERIAL/$IMAGE_NAME
UTMCTL=/Applications/UTM.app/Contents/MacOS/utmctl
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

die() { echo "error: $*" >&2; exit 1; }
log() { echo "==> $*"; }

[[ -x $UTMCTL ]] || die "UTM not found; install it with: brew install --cask utm"
# UTM's scripting interface can't resize disks, so the image is grown with qemu-img before import.
command -v qemu-img >/dev/null || die "qemu-img not found; install it with: brew install qemu"
[[ -n $BRIDGE_IF ]] || die "could not detect a network interface; set BRIDGE_IF (e.g. en1)"
if ! osascript -e 'tell application "UTM" to count virtual machines' >/dev/null 2>&1; then
  die "macOS blocked this terminal from controlling UTM. Allow it in System Settings >
       Privacy & Security > Automation (enable UTM under your terminal app), then re-run."
fi
if "$UTMCTL" list | awk 'NR > 1 {print $3}' | grep -qxF "$VM_NAME"; then
  die "a UTM VM named '$VM_NAME' already exists; delete it or set VM_NAME"
fi

if [[ -z $VM_PASSWORD ]]; then
  read -rsp "Password for '$VM_USER' in the VM: " VM_PASSWORD; echo
  read -rsp "Repeat password: " confirm; echo
  [[ $VM_PASSWORD == "$confirm" ]] || die "passwords do not match"
fi
password_hash=$(openssl passwd -6 "$VM_PASSWORD")

mkdir -p "$CACHE_DIR"
image=$CACHE_DIR/${IMAGE_NAME%.img}-$IMAGE_SERIAL.img
if [[ ! -f $image ]] || [[ $(shasum -a 256 "$image" | cut -d' ' -f1) != "$IMAGE_SHA256" ]]; then
  log "Downloading $IMAGE_NAME ($IMAGE_SERIAL)"
  curl -fL --progress-bar -o "$image.part" "$IMAGE_URL"
  [[ $(shasum -a 256 "$image.part" | cut -d' ' -f1) == "$IMAGE_SHA256" ]] || die "checksum mismatch for $IMAGE_URL"
  mv "$image.part" "$image"
fi

log "Building cloud-init seed"
seed_dir=$(mktemp -d)
trap 'rm -rf "$seed_dir"' EXIT
timezone=$(readlink /etc/localtime | sed 's|.*/zoneinfo/||')
ssh_keys=""
if [[ -n $SSH_PUBKEY_FILE ]]; then
  ssh_keys="    ssh_authorized_keys:
      - $(cat "$SSH_PUBKEY_FILE")"
fi
cat > "$seed_dir/meta-data" <<EOF
instance-id: $VM_NAME-$(date +%s)
local-hostname: $VM_NAME
EOF
cat > "$seed_dir/user-data" <<EOF
#cloud-config
hostname: $VM_NAME
timezone: $timezone
users:
  - name: $VM_USER
    gecos: $VM_USER
    groups: [adm, sudo, dialout, plugdev]
    shell: /bin/bash
    lock_passwd: false
    passwd: "$password_hash"
$ssh_keys
write_files:
  - path: /usr/local/sbin/provision.sh
    permissions: "0755"
    content: |
$(sed 's/^/      /' "$SCRIPT_DIR/provision.sh")
runcmd:
  - [bash, /usr/local/sbin/provision.sh, $VM_USER]
power_state:
  mode: poweroff
  condition: test -f /var/lib/provision.done
EOF
seed_iso=$seed_dir/seed.iso
hdiutil makehybrid -quiet -iso -joliet -default-volume-name cidata -o "$seed_iso" "$seed_dir"

disk=$seed_dir/$VM_NAME.qcow2
cp "$image" "$disk"
qemu-img resize -q -f qcow2 "$disk" "${DISK_MIB}M"

log "Creating UTM VM '$VM_NAME' (${CPUS} CPUs, ${MEMORY_MIB} MiB RAM, ${DISK_MIB} MiB disk, bridged to $BRIDGE_IF)"
# The seed is attached as a second VirtIO disk rather than a removable (USB) drive: cloud-init
# probes for it early in boot, before USB storage reliably shows up. UTM copies both disks
# into the VM bundle.
vm_id=$(osascript - "$VM_NAME" "$CPUS" "$MEMORY_MIB" "$disk" "$seed_iso" "$BRIDGE_IF" <<'EOF'
on run argv
  set {vmName, cpuCount, memMiB, diskPath, seedPath, bridgeIf} to argv
  set diskFile to POSIX file diskPath
  set seedFile to POSIX file seedPath
  tell application "UTM"
    set vm to make new virtual machine with properties {backend:qemu, configuration:{name:vmName, architecture:"aarch64", cpu cores:(cpuCount as integer), memory:(memMiB as integer), drives:{{source:diskFile}, {source:seedFile}}, network interfaces:{{mode:bridged, host interface:bridgeIf}}, displays:{{hardware:"virtio-gpu-gl-pci"}}}}
    return id of vm
  end tell
end run
EOF
)

log "Provisioning (desktop + ROS 2 Jazzy); this takes a while. The VM powers off when done."
"$UTMCTL" start "$vm_id"
deadline=$(( $(date +%s) + PROVISION_TIMEOUT_MIN * 60 ))
sleep 30
until [[ $("$UTMCTL" status "$vm_id") == stopped ]]; do
  # The guest agent is installed early in provision.sh; once it answers, catch failures right away.
  # (utmctl exec doesn't wait for the command, so read cloud-init's result file rather than run
  # the slower `cloud-init status`.) result.json appears when cloud-init finishes.
  result=$("$UTMCTL" exec "$vm_id" --cmd /bin/cat /run/cloud-init/result.json 2>/dev/null || true)
  if [[ -n $result && $result != *'"errors": []'* ]]; then
    echo
    "$UTMCTL" exec "$vm_id" --cmd /usr/bin/tail -n 30 /var/log/cloud-init-output.log || true
    die "provisioning failed (log above); the VM is still running for inspection"
  fi
  if (( $(date +%s) > deadline )); then
    die "provisioning did not finish within $PROVISION_TIMEOUT_MIN minutes; log in on the VM console and check /var/log/cloud-init-output.log"
  fi
  sleep 30
  printf '.'
done
echo

log "Detaching cloud-init seed"
osascript - "$vm_id" <<'EOF'
on run argv
  tell application "UTM"
    set vm to virtual machine id (item 1 of argv)
    set config to configuration of vm
    -- The seed is the second drive; keep only the system disk.
    set drives of config to {item 1 of (drives of config)}
    update configuration of vm with config
  end tell
end run
EOF

# Left stopped: a display window opened during provisioning stays attached to that run, so the
# VM is best started fresh from the UTM window.
log "Done. Start '$VM_NAME' from the UTM window and log in as '$VM_USER'."
