#!/usr/bin/env bash
# One command from a wave to its results on local disk, with the pod confirmed stopped at the end.
#
#   cloud/lane.sh run NAME          provision -> bootstrap -> data -> wave -> poll -> download -> verify -> terminate
#   STAGE=<name> cloud/lane.sh run NAME   one stage against the recorded pod
#   POD=<id> cloud/lane.sh run NAME       use a running pod instead of provisioning one
#
# NAME is a wave made by cloud/wave.py (cloud/waves/NAME/). Results land in cloud/results/NAME/.
# Run it with bash (zsh doesn't word-split $SSH_OPTS).
#
# Adapted from terminal-wm/cloud/lane.sh, keeping its rules: work is decided up front (a pod is never
# rented for nothing); "unreachable" is never evidence of anything; the completion marker is this run's
# (by nonce), because an earlier run's is indistinguishable from instant success; termination is
# confirmed against the provider, because the terminate call exits 0 on an API error.
#
# Turtlebot's failure policy differs (its jobs are cheap to rerun): a failure, or the budget running out,
# pulls whatever finished, then TERMINATES the pod and confirms it, rather than leaving it up to inspect.
# LANE_KEEP_POD=1 keeps it up instead. Results are pulled on every poll, so a stop loses little.
#
# Budget: RUNPOD_BUDGET_USD (default 20) per run. The wave's estimate (cloud/wave.py estimate) must fit
# it before anything is rented, and the run is stopped when the pod's time (counted from just before the
# deploy) reaches it, or after POLL_MAX_MIN minutes of polling (default 180), whichever comes first.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RP="bash cloud/runpod.sh"
POD_REPO="~/turtlebot"
TMUX_SESSION="tb-wave"
BUDGET_USD="${RUNPOD_BUDGET_USD:-20}"
POLL_MAX_MIN="${POLL_MAX_MIN:-180}"
POD="${POD:-}"
NAME=""
STATE=""

log() { printf '[lane %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[lane] ERROR: %s\n' "$*" >&2; exit 1; }
wave_py() { (cd learn && uv run python ../cloud/wave.py "$@"); }

# STATE (cloud/.lane-NAME) holds "<podId> <started epoch> <$/hr>", written once per pod: when it's
# deployed, or when the lane attaches to one (POD=). STATE.pending holds a launching pod's id before
# launch has finished, so a launch that fails or is interrupted can still be stopped.
_pod() {
    [ -n "$POD" ] && { echo "$POD"; return; }
    [ -s "$STATE" ] || die "no pod recorded in $STATE — run provision, or pass POD=<id>"
    head -1 "$STATE" | awk '{print $1}'
}

# Seconds of pod time the budget still allows; the full budget if the price isn't recorded (never an
# error: one failed lookup mustn't end a run).
_budget_left() {
    local started price
    started="$( [ -s "$STATE" ] && head -1 "$STATE" | awk '{print $2}' )"
    price="$( [ -s "$STATE" ] && head -1 "$STATE" | awk '{print $3}' )"
    awk -v b="$BUDGET_USD" -v p="${price:-0}" -v s="${started:-0}" -v now="$(date +%s)" \
        'BEGIN { if (p <= 0 || s <= 0) printf "%d", 1e9; else printf "%d", b / p * 3600 - (now - s) }'
}

_ssh() {   # 255 when the host can't be reached, which callers MUST tell apart
    local h ip port
    h="$($RP host "$(_pod)" 2>/dev/null | tail -1)" || return 255
    [ -n "$h" ] || return 255
    ip="${h%% *}"; port="${h##* }"
    ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=25 \
        ${RUNPOD_SSH_KEY:+-i "$RUNPOD_SSH_KEY"} -p "$port" "root@$ip" "$@"
}

# Cleanup on any exit that isn't a finished run: a die, a failed command, or a signal (Ctrl-C, kill, a
# closed terminal). Decided by LANE_OK, set only at the very end of a successful run, never by $?: bash 5
# gives an EXIT trap status 0 after a signal. Only a pod the lane has taken charge of is stopped: one in
# STATE (deployed, or attached with POD= and recorded) or STATE.pending (mid-launch), never a POD= pod
# a pre-flight check refused before anything ran.
LANE_OK=0
CLEANED=0
_on_exit() {
    local rc="${1:-$?}"
    [ "$LANE_OK" = 1 ] && return 0
    [ "$CLEANED" = 1 ] && return 0
    CLEANED=1
    local id=""
    [ -n "$STATE" ] && [ -s "$STATE" ] && id="$(head -1 "$STATE" | awk '{print $1}')"
    [ -n "$id" ] || id="$( [ -n "$STATE" ] && [ -s "$STATE.pending" ] && cat "$STATE.pending" || true )"   # mid-launch
    [ -n "$id" ] || return 0
    if [ "${LANE_KEEP_POD:-0}" = 1 ]; then
        printf '[lane] FAILED (rc=%s) — pod %s is STILL RUNNING (LANE_KEEP_POD=1).\n' "$rc" "$id" >&2
        printf '       Look: cloud/runpod.sh ssh %s ; stop: cloud/runpod.sh terminate %s && cloud/runpod.sh reap\n' "$id" "$id" >&2
        return 0
    fi
    printf '[lane] FAILED (rc=%s) — pulling what finished, then stopping pod %s.\n' "$rc" "$id" >&2
    $RP pull "$id" "$NAME" >/dev/null 2>&1 || printf '[lane] pull failed: results on the pod are lost with it\n' >&2
    _terminate_confirmed "$id" || true
}

_terminate_confirmed() {
    local id="$1" i lst t0=$SECONDS
    $RP terminate "$id" || true
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
        [ $((SECONDS - t0)) -lt 120 ] || break
        lst="$($RP list 2>/dev/null || true)"
        if printf '%s' "$lst" | jq -e 'type == "array"' >/dev/null 2>&1; then
            printf '%s' "$lst" | jq -e --arg id "$id" 'any(.[]; .id == $id and .desiredStatus == "RUNNING")' >/dev/null 2>&1 \
                || { : > "$STATE"; rm -f "$STATE.pending"; log "terminate: CONFIRMED stopped ($id)"; return 0; }
        fi
        sleep 5
    done
    printf '[lane] pod %s did NOT confirm termination — stop it now: cloud/runpod.sh terminate %s ; then cloud/runpod.sh reap\n' "$id" "$id" >&2
    return 1
}

stage_provision() {
    if [ -s "$STATE" ]; then log "provision: reusing $(_pod)"; return; fi
    local price out t0
    price="$($RP price | tail -1)"
    if [ -n "$POD" ]; then   # attaching: the budget counts from now, at today's price
        echo "$POD $(date +%s) $price" > "$STATE"; log "provision: attached to $POD"; return
    fi
    log "provision: deploying at \$$price/hr (budget \$$BUDGET_USD)"
    t0="$(date +%s)"   # the pod bills from its deploy, before it's reachable
    out="$(RUNPOD_ID_FILE="$STATE.pending" $RP launch | tail -1)"
    echo "${out%% *} $t0 $price" > "$STATE"
    rm -f "$STATE.pending"
    log "provision: $(_pod)"
}

# A wave already running on the pod (a rerun after the lane itself stopped): don't touch it, resume
# polling it (runpod.sh bootstrap exits 4 for that).
RESUMING=0
stage_bootstrap() {
    local rc=0
    log "bootstrap: $(_pod)"
    $RP bootstrap "$(_pod)" || rc=$?
    if [ "$rc" = 4 ]; then
        [ -s "$STATE.runid" ] || die "a wave is running on $(_pod) but this lane has no record of dispatching it: look before doing anything (cloud/runpod.sh ssh $(_pod))"
        TB_RUN_ID="$(cat "$STATE.runid")"; export TB_RUN_ID
        RESUMING=1; log "bootstrap: a wave is already running — resuming at poll (run id $TB_RUN_ID)"
    elif [ "$rc" != 0 ]; then
        die "bootstrap failed (exit $rc)"
    fi
}
stage_data()      { log "data: $(_pod)"; $RP data "$(_pod)"; }

stage_wave() {
    local i cleared=0
    # Cleared before dispatch, synchronously: the poller must never see an earlier run's marker.
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if _ssh "rm -f $POD_REPO/cloud/results/$NAME/.done" 2>/dev/null; then cleared=1; break; fi
        log "wave: pod unreachable clearing the marker (attempt $i) — retrying in 20s"; sleep 20
    done
    [ "$cleared" = 1 ] || die "pod unreachable for 10 attempts while clearing the completion marker"
    log "wave: $NAME, run id $TB_RUN_ID"
    echo "$TB_RUN_ID" > "$STATE.runid"   # so a later STAGE=poll looks for this run's marker
    $RP wave "$(_pod)" "cloud/waves/$NAME"
}

stage_poll() {
    local rc misses=0 left t0=$SECONDS
    while :; do
        left="$(_budget_left)"
        [ "$left" -gt 0 ] || die "budget \$$BUDGET_USD reached — stopping the run"
        [ $((SECONDS - t0)) -lt $((POLL_MAX_MIN * 60)) ] || die "still running after ${POLL_MAX_MIN} min of polling — stopping the run"
        rc=0; _ssh "grep -q '$TB_RUN_ID' $POD_REPO/cloud/results/$NAME/.done" >/dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            log "poll: this run's marker present ($(_ssh "cat $POD_REPO/cloud/results/$NAME/.done" 2>/dev/null | cut -d' ' -f2-))"
            return
        fi
        if [ "$rc" = 255 ]; then log "poll: unreachable (transient) — no evidence either way"; sleep 30; continue; fi
        rc=0; _ssh "tmux has-session -t $TMUX_SESSION" >/dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then
            misses=0
            $RP pull "$(_pod)" "$NAME" >/dev/null 2>&1 || true   # what's finished comes home as it finishes
            log "poll: running — $(ls cloud/results/$NAME/*.json 2>/dev/null | wc -l | tr -d ' ') record(s) home, $((left / 60)) min of budget left"
        elif [ "$rc" = 255 ]; then
            log "poll: unreachable (transient)"
        else
            misses=$((misses + 1))
            [ "$misses" -lt 3 ] || {
                _ssh "tail -40 $POD_REPO/cloud/results/$NAME/wave.log" >&2 || true
                die "reachable, no session, no marker after 3 polls: the wave died"
            }
        fi
        sleep 60
    done
}

stage_download() { log "download"; $RP pull "$(_pod)" "$NAME" >/dev/null; }

stage_verify() {
    local rc=0
    wave_py verify "$NAME" || rc=$?
    case "$rc" in
        0) log "verify: results complete" ;;
        3) [ "${LANE_ALLOW_ALL_FAILED:-0}" = 1 ] || die "every genome failed — that is the environment, not the candidates (LANE_ALLOW_ALL_FAILED=1 to accept)" ;;
        *) die "results are not all on local disk" ;;
    esac
}

stage_terminate() { _terminate_confirmed "$(_pod)" || die "termination not confirmed"; rm -f "$STATE.runid"; }

cmd_run() {
    NAME="${1:?usage: cloud/lane.sh run NAME}"
    [ -f "cloud/waves/$NAME/wave.json" ] || die "no cloud/waves/$NAME/wave.json (cloud/wave.py make|grid|probe)"
    STATE="${LANE_STATE:-cloud/.lane-$NAME}"
    # A single later stage resumes the dispatched run's id; a full run or a new dispatch gets a fresh one.
    if [ -z "${TB_RUN_ID:-}" ] && [ -n "${STAGE:-}" ] && [ "$STAGE" != wave ] && [ -s "$STATE.runid" ]; then
        TB_RUN_ID="$(cat "$STATE.runid")"
    fi
    : "${TB_RUN_ID:=lane-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
    export TB_RUN_ID
    # Pre-flight, before the cleanup trap exists: a refusal here stops nothing, even a POD= pod.
    if [ -z "${STAGE:-}" ] && [ ! -s "$STATE" ]; then
        local est
        est="$(wave_py estimate "$NAME" | tail -1)"
        log "estimate: $est"
        printf '%s' "$est" | jq -e --argjson b "$BUDGET_USD" '.dollars <= $b' >/dev/null \
            || die "the estimate exceeds the \$$BUDGET_USD budget (RUNPOD_BUDGET_USD)"
    fi
    trap '_on_exit' EXIT
    trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP   # each runs the EXIT trap
    if [ -n "${STAGE:-}" ]; then "stage_$STAGE"; LANE_OK=1; return; fi
    local s
    for s in provision bootstrap data wave poll download verify terminate; do
        if [ "$RESUMING" = 1 ] && { [ "$s" = data ] || [ "$s" = wave ]; }; then continue; fi
        "stage_$s"
    done
    LANE_OK=1
}

case "${1:-}" in
    run) shift; cmd_run "$@" ;;
    *) sed -n '2,24p' "$0"; exit 1 ;;
esac
