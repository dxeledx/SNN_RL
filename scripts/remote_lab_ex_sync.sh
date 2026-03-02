#!/usr/bin/env bash
set -euo pipefail

HOST="lab-external"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/home/wjx/workspace/Lab/bci-snn-rl"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

retry() {
  local -r max_tries="${1:-3}"
  shift
  local try=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( try >= max_tries )); then
      echo "[sync] failed after ${try} tries: $*" >&2
      return 1
    fi
    try=$((try + 1))
    sleep 1
  done
}

echo "[sync] ${LOCAL_ROOT} -> ${HOST}:${REMOTE_ROOT}"
#
# WAN-safe: sync code only; never ship datasets / runs / caches.
# Use rsync (more robust than scp+tar on flaky links).
#
retry 6 rsync -az --delete --partial \
  -e "ssh ${SSH_OPTS[*]}" \
  --rsync-path="mkdir -p '${REMOTE_ROOT}' && rsync" \
  --exclude '/.git/' \
  --exclude '/data/' \
  --exclude '/runs/' \
  --exclude '/results/cache/' \
  --exclude '/artifacts/' \
  --exclude '/trash/' \
  "${LOCAL_ROOT}/" \
  "${HOST}:${REMOTE_ROOT}/"

echo "[sync] done"
