#!/usr/bin/env bash
set -euo pipefail

HOST="lab-external"
REMOTE_ROOT="/home/wjx/workspace/Lab/bci-snn-rl"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command...>"
  exit 2
fi

CMD="$*"
echo "[run] ${HOST}:${REMOTE_ROOT} $CMD"
ssh "${SSH_OPTS[@]}" "${HOST}" "cd '${REMOTE_ROOT}' && ${CMD}"
