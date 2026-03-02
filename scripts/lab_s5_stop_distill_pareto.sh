#!/usr/bin/env bash
set -euo pipefail

# Local driver script:
# - sync code to lab-external
# - start 3 tmux sessions (seed0/1/2), each runs: LDA distill -> PPO pareto

HOST="${HOST:-lab-external}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/home/wjx/workspace/Lab/bci-snn-rl"

SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
VARIANT="${VARIANT:-iv2b_f8-30_t0-4_w3_s0.25_bp6_z1e-12}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
RESTART="${RESTART:-0}"  # set to 1 to kill and restart existing tmux sessions

CONDA="${CONDA:-/home/wjx/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-eeg}"

CONFIG_DISTILL="${CONFIG_DISTILL:-configs/exp/iv2b_stop_snn_distill_pretrain_sota.yaml}"
CONFIG_PARETO="${CONFIG_PARETO:-configs/exp/iv2b_stop_snn_ppo_spikefix_sota.yaml}"

retry() {
  local -r max_tries="${1:-3}"
  shift
  local try=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( try >= max_tries )); then
      echo "[s5] failed after ${try} tries: $*" >&2
      return 1
    fi
    try=$((try + 1))
    sleep 1
  done
}

echo "[s5] sync code to ${HOST} ..."
bash "${REPO_ROOT}/scripts/remote_lab_ex_sync.sh"

start_seed() {
  local -r seed="$1"
  local -r sess="codex-bci-s5-stop-seed${seed}"
  local -r distill_out="runs/s5_stop_distill_w3fb6_seed${seed}"
  local -r ppo_out="runs/s5_stop_snn_spikefix_distill_w3fb6_seed${seed}"

  # IMPORTANT: no single quotes in this string (it's wrapped by tmux as a single-quoted command).
  local -r inner_cmd="set -euo pipefail; \
cd ${REMOTE_ROOT}; \
export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH=src; \
mkdir -p ${distill_out} ${ppo_out}; \
${CONDA} run -n ${ENV_NAME} python -m bci_snn_rl.run_pretrain_stop_lda_distill \
  --config ${CONFIG_DISTILL} \
  --subjects ${SUBJECTS} \
  --variant ${VARIANT} \
  --override \
    project.seed=${seed} \
    project.out_dir=${distill_out} \
    project.overwrite=true \
  2>&1 | tee -a ${distill_out}/exp.log; \
${CONDA} run -n ${ENV_NAME} python -m bci_snn_rl.run_pareto \
  --config ${CONFIG_PARETO} \
  --subjects ${SUBJECTS} \
  --variant ${VARIANT} \
  --override \
    project.seed=${seed} \
    train.total_steps=${TOTAL_STEPS} \
    train.init_checkpoint=${distill_out}/checkpoints/distill_best.pt \
    project.out_dir=${ppo_out} \
    project.overwrite=true \
  2>&1 | tee -a ${ppo_out}/exp.log; \
echo [s5] done seed=${seed}"

  echo "[s5] start tmux ${sess}"
  if [[ "${RESTART}" == "1" ]]; then
    retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && tmux kill-session -t '${sess}' || true; tmux new -d -s '${sess}' '${inner_cmd}'"
    return 0
  fi

  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && echo '[s5] exists: ${sess} (set RESTART=1 to restart)' || tmux new -d -s '${sess}' '${inner_cmd}'"
}

start_seed 0 || true
start_seed 1 || true
start_seed 2 || true

echo "[s5] started."
echo "  ssh -t ${HOST} 'tmux ls'"
echo "  ssh -t ${HOST} 'tmux attach -t codex-bci-s5-stop-seed0'"
