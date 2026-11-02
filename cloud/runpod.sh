#!/usr/bin/env bash
# cloud/runpod.sh — rent a GPU pod on RunPod, put learn/ on it, run a wave of world-model
# trainings, bring the results home. Pure curl + jq + ssh + rsync; no vendor CLI. Adapted from
# terminal-wm/cloud/runpod.sh: every guard there (the deploy ledger, never-retried deploys,
# bounded waits, the driver check) is kept; the project-specific parts are turtlebot's.
#
# A pod is a Docker container. You get `root` over SSH on an EXPOSED PUBLIC PORT (not user@ip:22);
# the ssh public key is injected through the pod's PUBLIC_KEY environment variable.
#
# Subcommands (each prints machine-parseable results on the LAST stdout line):
#   types [--available]    GPU types with price + stock (filter: $RUNPOD_TYPES_FILTER)
#   price                  the configured GPU type's on-demand $/hr, for a cost estimate
#   launch                 deploy a pod; prints "<podId> <ip> <port>"
#   host      <podId>      "<ip> <port>" of a running pod
#   ssh       <podId>      interactive ssh
#   bootstrap <podId>      apt deps, the COMMITTED code (git archive HEAD: learn/ + cloud/), uv sync,
#                          CUDA smoke test
#   data      <podId>      pull the pinned sessions from the private Hugging Face dataset on the pod
#                          (learn/data_source.json), with the token written to the pod's ~/.runpod.env
#                          (never passed on a command line)
#   wave      <podId> <cloud/waves/NAME>   ship a wave (cloud/wave.py make) and run it in tmux
#   pull      <podId> <NAME>               rsync the pod's cloud/results/NAME/ home
#   terminate <podId>      TERMINATE THE POD: what stops billing
#   status    <podId> | list
#   reap                   every pod on the account against the ids this machine deployed
#
# Auth: RUNPOD_API_KEY, from $RUNPOD_ENV_FILE (default ~/.runpod.env); never echoed. The pod gets the
# Hugging Face token from the same file: HF_READ_TOKEN if set (a read-only one), else HF_TOKEN, as the
# sibling projects' pods do.
#
# Env knobs:
#   RUNPOD_GPU_TYPE     default "NVIDIA GeForce RTX 4090"
#   RUNPOD_CLOUD        SECURE | COMMUNITY | ALL (default SECURE)
#   RUNPOD_IMAGE        default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (only the
#                       CUDA userspace and the driver matter: uv installs learn's own Python + torch)
#   RUNPOD_DISK_GB      default 60 (a CUDA torch venv is ~12 GB, the uv cache about as much again)
#   RUNPOD_POD_NAME     default tb-wave (the tb- prefix tells turtlebot's pods apart in `reap`)
#   RUNPOD_ALLOWED_CUDA host driver CUDA versions to accept: default 13.0,13.1,13.2. learn/ locks
#                       torch 2.15.0, whose only CUDA builds are CUDA 13 (CUDA-12 wheels stop at
#                       2.14.1), and swapping torch versions would change the scores
#   RUNPOD_SSH_KEY / RUNPOD_PUBKEY_FILE   ssh identity (auto-discovered otherwise)

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_REPO='$HOME/turtlebot'
POD_REPO_LIT="~/turtlebot"

RUNPOD_ENV_FILE="${RUNPOD_ENV_FILE:-$HOME/.runpod.env}"
[ -f "$RUNPOD_ENV_FILE" ] && { set +u; . "$RUNPOD_ENV_FILE"; set -u; }
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (put it in $RUNPOD_ENV_FILE)}"

GQL="https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY"
GPU_TYPE="${RUNPOD_GPU_TYPE:-NVIDIA GeForce RTX 4090}"
CLOUD="${RUNPOD_CLOUD:-SECURE}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
DISK_GB="${RUNPOD_DISK_GB:-60}"
POD_NAME="${RUNPOD_POD_NAME:-tb-wave}"
TYPES_FILTER="${RUNPOD_TYPES_FILTER:-4090|5090|A100|H100|L40|A40|RTX 6000}"
ALLOWED_CUDA="${RUNPOD_ALLOWED_CUDA:-13.0,13.1,13.2}"
MIN_CUDA="${ALLOWED_CUDA%%,*}"
TMUX_SESSION="tb-wave"
LEDGER="${RUNPOD_LEDGER:-$HOME/.runpod-pods.log}"

RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-}"
if [ -z "$RUNPOD_SSH_KEY" ]; then
    for c in "$HOME/.ssh/id_ed25519" "$HOME/.ssh/id_rsa"; do
        [ -f "$c" ] && RUNPOD_SSH_KEY="$c" && break
    done
fi
# accept-new: every pod is a fresh host key. The key is offered, not pinned: RunPod also honours the
# account-level console key, and a pod can accept that one instead (RUNPOD_SSH_PIN=1 pins anyway).
# Keepalive 30 s x 3 fails a dead peer in 90 s; ConnectTimeout bounds an unanswered handshake.
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=30${RUNPOD_SSH_KEY:+ -i $RUNPOD_SSH_KEY}${RUNPOD_SSH_PIN:+ -o IdentitiesOnly=yes}"

# Every API call is bounded. Reads retry (idempotent); mutations and deploys never do: a deploy
# that timed out may already exist server-side, and a retry would bill a second pod nobody tracks.
CURL_TIMEOUTS="--connect-timeout 15 --max-time 60"
CURL_READ="--connect-timeout 15 --max-time 25 --retry 3 --retry-connrefused --retry-max-time 120"
CURL_DEPLOY="--connect-timeout 15"   # no --max-time: losing the response to a created pod is worse

RSYNC_TIMEOUT=""
rsync --version 2>/dev/null | head -1 | grep -q '^rsync ' && RSYNC_TIMEOUT="--timeout=120"   # GNU rsync only

log() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

# A retried read writes to a file, so a half-sent body from a stalled attempt can't be concatenated
# ahead of the good one.
_curl_read() {
    local out rc=0
    out=$(mktemp) || return 1
    curl -sS $CURL_READ -o "$out" "$@" || rc=$?
    [ "$rc" = 0 ] && cat "$out"
    rm -f "$out"
    return "$rc"
}

gql() {
    local q; q=$(jq -Rn --arg q "$1" '{query:$q}')
    case "$1" in
        *mutation*) curl -sS $CURL_TIMEOUTS -X POST "$GQL" -H "Content-Type: application/json" -d "$q" ;;
        *)          _curl_read -X POST "$GQL" -H "Content-Type: application/json" -d "$q" ;;
    esac
}

REST_BASE="https://rest.runpod.io/v1"
rest() {
    local method="$1" path="$2" body="${3:-}"
    if [ "$method" = "GET" ]; then
        _curl_read -X GET "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
        return
    fi
    local flags="$CURL_TIMEOUTS"
    [ "$method" = "POST" ] && flags="$CURL_DEPLOY"
    if [ -n "$body" ]; then
        curl -sS $flags -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$body"
    else
        curl -sS $flags -X "$method" "$REST_BASE$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
}

_pubkey() {
    local f="${RUNPOD_PUBKEY_FILE:-${RUNPOD_SSH_KEY:+$RUNPOD_SSH_KEY.pub}}"
    [ -n "$f" ] && [ -f "$f" ] || die "no ssh public key (set RUNPOD_PUBKEY_FILE, or RUNPOD_SSH_KEY with a .pub beside it)"
    cat "$f"
}

_host() {
    local id="$1" js
    js=$(gql "query{ pod(input:{podId:\"$id\"}){ id desiredStatus runtime{ ports{ ip isIpPublic privatePort publicPort type } } } }")
    echo "$js" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | "\(.ip) \(.publicPort)"' | head -1
}

_ssh_to() { local ip="$1" port="$2"; shift 2; ssh $SSH_OPTS -p "$port" "root@$ip" "$@"; }

_need_host() {
    local id="$1"
    read -r ip port < <(_host "$id") || true
    [ -n "${ip:-}" ] && [ -n "${port:-}" ] || die "no public SSH port for $id (is it running? try: $0 status $id)"
}

_ledger() { mkdir -p "$(dirname "$LEDGER")" && echo "$(date -u +%FT%TZ) $1" >> "$LEDGER"; }

cmd_types() {
    local sel=""
    [ "${1:-}" = "--available" ] && sel='| select(.lowestPrice.stockStatus != null)'
    gql "query{ gpuTypes{ id displayName lowestPrice(input:{gpuCount:1}){ uninterruptablePrice stockStatus } } }" \
        | jq -r ".data.gpuTypes[]? | select(.displayName|test(\"$TYPES_FILTER\")) $sel | \"\(.displayName) | stock:\(.lowestPrice.stockStatus // \"none\") \$\(.lowestPrice.uninterruptablePrice // \"-\")/hr | id=\(.id)\""
}

cmd_price() {
    # securePrice / communityPrice: the on-demand rate for the configured cloud (SECURE by default).
    local field="securePrice"; [ "$CLOUD" = "COMMUNITY" ] && field="communityPrice"
    local p
    p=$(gql "query{ gpuTypes(input:{id:\"$GPU_TYPE\"}){ id $field } }" | jq -r ".data.gpuTypes[0].$field // empty")
    [ -n "$p" ] || die "no $CLOUD price for '$GPU_TYPE'"
    echo "$p"
}

# Poll a just-created pod until RUNNING with a public SSH port, wait for sshd, print "<id> <ip> <port>".
# Bounded by WALL CLOCK: a stalled API makes each retried read cost ~107 s.
_await_ssh() {
    local id="$1"
    log "pod id: $id — polling for RUNNING + a public SSH port (2-5 min)"
    local ip="" port="" st="" tries=0 t0=$SECONDS budget=900
    while :; do
        [ $((SECONDS - t0)) -lt "$budget" ] || die "pod never exposed a public SSH port within ${budget}s (check the RunPod console for $id, and terminate it there if it is billing)"
        st=$(gql "query{ pod(input:{podId:\"$id\"}){ desiredStatus } }" 2>/dev/null | jq -r '.data.pod.desiredStatus // "?"' 2>/dev/null || echo "?")
        ip=""; port=""; read -r ip port < <(_host "$id") || true   # no port yet: read fails, under set -e that would strand the pod
        echo "    status=$st ssh=${ip:+$ip:$port}" >&2
        [ -n "${ip:-}" ] && [ -n "${port:-}" ] && break
        tries=$((tries+1)); [ "$tries" -gt 60 ] && die "pod never exposed a public SSH port (check the RunPod console for $id)"
        sleep 10
    done
    log "waiting for sshd at $ip:$port"
    for _ in $(seq 1 30); do _ssh_to "$ip" "$port" true 2>/dev/null && break; sleep 10; done
    echo "$id $ip $port"
}

_deploy() {
    local pubkey; pubkey=$(_pubkey | tr -d '\n')
    log "deploying 1x '$GPU_TYPE' ($CLOUD), disk=${DISK_GB}GB, allowedCuda=[$ALLOWED_CUDA]"
    local body r id
    body=$(jq -n --arg name "$POD_NAME" --arg img "$IMAGE" --arg cloud "$CLOUD" --arg gt "$GPU_TYPE" \
        --argjson disk "$DISK_GB" --arg pk "$pubkey" --arg cuda "$ALLOWED_CUDA" \
        '{name:$name, imageName:$img, cloudType:$cloud, computeType:"GPU",
          gpuCount:1, gpuTypeIds:[$gt], volumeInGb:0, containerDiskInGb:$disk,
          ports:["22/tcp"], env:{PUBLIC_KEY:$pk},
          allowedCudaVersions:($cuda | split(","))}')
    r=$(rest POST /pods "$body")
    id=$(echo "$r" | jq -r '.id // empty')
    [ -n "$id" ] || { echo "$r" | jq -r '.error // .message // .' >&2; die "deploy failed"; }
    _ledger "$id"   # recorded the instant it exists: if anything later dies, `reap` still sees it
    echo "$id"
}

_driver_cuda() {  # <ip> <port>: the host driver's CUDA version; empty if it couldn't be read (unreachable)
    local i drv=""
    for i in 1 2 3 4 5; do
        drv=$(_ssh_to "$1" "$2" 'nvidia-smi | grep -oE "CUDA Version: [0-9]+\.[0-9]+"' 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' || true)
        [ -n "$drv" ] && break
        sleep 10
    done
    echo "$drv"
}

# Terminate and confirm against `list` (podTerminate exits 0 even on an API error). 0 when confirmed.
_terminate_confirmed() {
    local id="$1" lst t0=$SECONDS
    cmd_terminate "$id" || true
    while [ $((SECONDS - t0)) -lt 120 ]; do
        lst="$(cmd_list 2>/dev/null || true)"
        if printf '%s' "$lst" | jq -e 'type == "array"' >/dev/null 2>&1; then
            printf '%s' "$lst" | jq -e --arg id "$id" 'any(.[]; .id == $id and .desiredStatus == "RUNNING")' >/dev/null 2>&1 \
                || { log "CONFIRMED stopped: $id"; return 0; }
        fi
        sleep 5
    done
    echo "ERROR: pod $id did NOT confirm termination — stop it now: $0 terminate $id ; then $0 reap" >&2
    return 1
}

# The id of a pod being launched goes to $RUNPOD_ID_FILE (when set) the instant it exists, so a caller
# that dies or is interrupted mid-launch still knows what to stop. A pod that never becomes usable is
# terminated here, confirmed, before anything else is deployed.
cmd_launch() {
    local pid id ip port try drv
    for try in 1 2 3 4; do
        pid=$(_deploy)
        [ -n "${RUNPOD_ID_FILE:-}" ] && echo "$pid" > "$RUNPOD_ID_FILE"
        id=""; ip=""; port=""; read -r id ip port < <(_await_ssh "$pid" | tail -1) || true
        if [ -z "${ip:-}" ]; then
            _terminate_confirmed "$pid" && die "launch: $pid never became reachable over ssh (terminated)"
            die "launch: $pid never became reachable over ssh, and its termination is NOT confirmed: $0 terminate $pid"
        fi
        drv="$(_driver_cuda "$ip" "$port")"
        if [ -z "$drv" ]; then
            _terminate_confirmed "$pid" && die "launch: couldn't read $pid's driver version over ssh (terminated)"
            die "launch: couldn't read $pid's driver version, and its termination is NOT confirmed: $0 terminate $pid"
        fi
        log "host driver CUDA: $drv (need >= $MIN_CUDA)"
        if awk -v d="$drv" -v m="$MIN_CUDA" 'BEGIN{exit !(d >= m)}'; then echo "$pid $ip $port"; return; fi
        log "driver too old on $pid (try $try/4) — terminating and re-rolling"
        _terminate_confirmed "$pid" || die "re-roll: $pid not confirmed stopped; not deploying another"
        sleep 5
    done
    die "no pod with driver CUDA >= $MIN_CUDA after 4 tries (pick another GPU type, or RUNPOD_CLOUD=ALL)"
}

cmd_host()   { local id="${1:?host <podId>}"; _host "$id"; }
cmd_status() { local id="${1:?status <podId>}"; gql "query{ pod(input:{podId:\"$id\"}){ id name desiredStatus machineId } }" | jq '.data.pod'; }
cmd_list()   { gql 'query{ myself{ pods{ id name desiredStatus machine{ gpuDisplayName } } } }' | jq '.data.myself.pods'; }

cmd_ssh() {
    local id="${1:?ssh <podId>}" ip port
    _need_host "$id"
    exec ssh $SSH_OPTS -p "$port" "root@$ip"
}

cmd_bootstrap() {
    local id="${1:?bootstrap <podId>}" ip port
    _need_host "$id"
    log "ensuring rsync/curl/tmux on the pod"
    _ssh_to "$ip" "$port" 'timeout 600 bash -c "(command -v rsync >/dev/null && command -v tmux >/dev/null) || (apt-get update -qq && apt-get install -y -qq rsync curl tmux)"'

    # The COMMITTED code, as evolve scores it (git archive HEAD): learn/ and cloud/ only. Uncommitted
    # changes never reach a pod; a wave's new impls travel with the wave.
    if _ssh_to "$ip" "$port" "tmux has-session -t $TMUX_SESSION" 2>/dev/null; then
        echo "a wave is running on $id: not re-shipping code under it (resume polling it: STAGE=poll)" >&2
        exit 4   # lane.sh resumes on this
    fi
    local head; head="$(git -C "$REPO_DIR" rev-parse HEAD)"
    log "shipping HEAD (${head:0:7}): learn/ + cloud/"
    # Replace the code, keeping what the pod made: its results (cloud/results) and environment (learn/.venv).
    _ssh_to "$ip" "$port" "mkdir -p $POD_REPO/learn $POD_REPO/cloud && cd $POD_REPO && \
        find learn -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} + && \
        find cloud -mindepth 1 -maxdepth 1 ! -name results -exec rm -rf {} +"
    git -C "$REPO_DIR" archive --format=tar HEAD learn cloud | _ssh_to "$ip" "$port" "tar -x -C $POD_REPO"
    echo "$head" | _ssh_to "$ip" "$port" "cat > $POD_REPO/cloud/.commit"   # which commit ran, for the records

    # Bounded: a stalled wheel download keeps the ssh session alive, so nothing client-side breaks it.
    log "uv sync + CUDA smoke on the pod"
    _ssh_to "$ip" "$port" "
        set -e
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf --connect-timeout 15 --max-time 300 --retry 3 --retry-max-time 600 \
                 -o /tmp/uv-install.sh https://astral.sh/uv/install.sh \
                || { rm -f /tmp/uv-install.sh; echo 'uv installer download failed' >&2; exit 1; }
            sh /tmp/uv-install.sh
        fi
        export PATH=\"\$HOME/.local/bin:\$PATH\"
        cd $POD_REPO/learn && timeout 2400 uv sync
        uv run python -c \"import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok:', torch.__version__, torch.cuda.get_device_name(0))\"
    "
    echo "$id ready"
}

cmd_data() {
    local id="${1:?data <podId>}" ip port tok
    _need_host "$id"
    tok="${HF_READ_TOKEN:-${HF_TOKEN:-}}"
    [ -n "$tok" ] || die "no HF_TOKEN (or HF_READ_TOKEN) in $RUNPOD_ENV_FILE"
    # Through stdin into a 600 file: never on a command line, never in a process listing.
    printf 'HF_TOKEN=%s\n' "$tok" | _ssh_to "$ip" "$port" "umask 077 && cat > ~/.runpod.env"
    log "pulling the pinned sessions on the pod"
    _ssh_to "$ip" "$port" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd $POD_REPO/learn && timeout 1200 uv run python -m pet_learn.hub pull --to $POD_REPO/data/steps"
    echo "$id data ok"
}

cmd_wave() {
    local id="${1:?wave <podId> <cloud/waves/NAME>}" dir="${2:?<cloud/waves/NAME>}" ip port name
    _need_host "$id"
    [ -f "$dir/wave.json" ] || die "no $dir/wave.json (make one with cloud/wave.py make)"
    name="$(basename "$dir")"
    ! _ssh_to "$ip" "$port" "tmux has-session -t $TMUX_SESSION" 2>/dev/null \
        || die "a '$TMUX_SESSION' session is already running on $id — follow it with: $0 ssh $id  then  tmux attach -t $TMUX_SESSION"
    log "shipping wave $name"
    _ssh_to "$ip" "$port" "mkdir -p $POD_REPO/cloud/waves/$name"
    rsync -a $RSYNC_TIMEOUT -e "ssh $SSH_OPTS -p $port" "${dir%/}/" "root@$ip:$POD_REPO_LIT/cloud/waves/$name/"
    _ssh_to "$ip" "$port" "cat > $POD_REPO/tb_wave.sh" <<EOF
#!/usr/bin/env bash
# Written by cloud/runpod.sh wave. Runs under tmux on this pod.
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export PET_STEPS=$POD_REPO/data/steps
${TB_RUN_ID:+export TB_RUN_ID='$TB_RUN_ID'}
cd $POD_REPO/learn
mkdir -p ../cloud/results/$name
exec > >(tee -a ../cloud/results/$name/wave.log) 2>&1
echo "=== wave $name  \$(date -u +%FT%TZ)"
export PET_PYTHON="\$(uv run python -c 'import sys; print(sys.executable)')"
"\$PET_PYTHON" ../cloud/wave_pod.py ../cloud/waves/$name
echo "=== done  \$(date -u +%FT%TZ)"
EOF
    _ssh_to "$ip" "$port" "chmod +x $POD_REPO/tb_wave.sh && tmux new-session -d -s $TMUX_SESSION -c $POD_REPO 'bash tb_wave.sh'"
    log "wave $name running on $id in tmux '$TMUX_SESSION'"
    echo "$id $TMUX_SESSION $name"
}

cmd_pull() {
    local id="${1:?pull <podId> <NAME>}" name="${2:?<NAME>}" ip port
    _need_host "$id"
    local dest="$REPO_DIR/cloud/results/$name"; mkdir -p "$dest"
    rsync -a $RSYNC_TIMEOUT --partial-dir=.rsync-partial -e "ssh $SSH_OPTS -p $port" \
        "root@$ip:$POD_REPO_LIT/cloud/results/$name/" "$dest/"
    echo "$dest"
}

cmd_reap() {
    # Only ever REPORTS: deciding what dies stays with a person.
    local live tracked
    live=$(gql 'query{ myself{ pods{ id name desiredStatus } } }' | jq -r '.data.myself.pods[]? | "\(.id) \(.name) \(.desiredStatus)"')
    tracked=$( [ -f "$LEDGER" ] && awk '{print $2}' "$LEDGER" || true )
    [ -n "$live" ] || { log "no pods on this account"; return 0; }
    echo "$live" | while read -r id name st; do
        if echo "$tracked" | grep -qx "$id"; then echo "  tracked   $id  $name  $st"
        else echo "  UNTRACKED $id  $name  $st"; fi
    done
}

cmd_terminate() {
    local id="${1:?terminate <podId>}"
    log "terminating $id"
    # Exits 0 even on an API error: confirm with `list` (lane.sh does).
    gql "mutation{ podTerminate(input:{podId:\"$id\"}) }" | jq -r 'if .errors then .errors[0].message else "terminated" end' >&2
}

case "${1:-}" in
    ""|-h|--help) awk 'NR>1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0" >&2; exit 0 ;;
esac
CMD="$1"; shift
case "$CMD" in
    types) cmd_types "$@";; price) cmd_price "$@";;
    launch) cmd_launch "$@";; host) cmd_host "$@";; ssh) cmd_ssh "$@";;
    bootstrap) cmd_bootstrap "$@";; data) cmd_data "$@";; wave) cmd_wave "$@";;
    pull) cmd_pull "$@";; terminate) cmd_terminate "$@";;
    status) cmd_status "$@";; list) cmd_list "$@";; reap) cmd_reap "$@";;
    *) die "unknown command: $CMD";;
esac
