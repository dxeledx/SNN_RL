#!/usr/bin/env bash
set -euo pipefail

# Local driver script:
# - sync code to lab-external
# - start 3 tmux sessions (seed0/1/2), each runs:
#     (1) policy distill pretrain (LDA -> 3-action soft policy)
#     (2) PPO pareto finetune with lr=1e-4

HOST="${HOST:-lab-external}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/home/wjx/workspace/Lab/bci-snn-rl"

SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
VARIANT="${VARIANT:-iv2b_f8-30_t0-4_w3_s0.25_bp6_z1e-12}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
RESTART="${RESTART:-0}" # set to 1 to kill and restart existing tmux sessions
SNN_OUTPUT_MODE="${SNN_OUTPUT_MODE:-}" # optional: "membrane" to use LIF membrane output instead of spikes
DISTILL_CONF_POWER="${DISTILL_CONF_POWER:-}" # optional: e.g. 4.0 to increase CONTINUE weight in distill teacher
DISTILL_RUN_NAME="${DISTILL_RUN_NAME:-s7_stop_policy_distill_w3fb6}"
PPO_RUN_NAME="${PPO_RUN_NAME:-s7_stop_snn_ppo_policy_distill_w3fb6}"

CONDA="${CONDA:-/home/wjx/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-eeg}"

CONFIG_DISTILL="${CONFIG_DISTILL:-configs/exp/iv2b_stop_snn_policy_distill_pretrain_sota.yaml}"
CONFIG_PARETO="${CONFIG_PARETO:-configs/exp/iv2b_stop_snn_ppo_spikefix_sota_finetune.yaml}"

retry() {
  local -r max_tries="${1:-3}"
  shift
  local try=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( try >= max_tries )); then
      echo "[s7] failed after ${try} tries: $*" >&2
      return 1
    fi
    try=$((try + 1))
    sleep 1
  done
}

echo "[s7] sync code to ${HOST} ..."
bash "${REPO_ROOT}/scripts/remote_lab_ex_sync.sh"

start_seed() {
  local -r seed="$1"
  local -r sess="codex-bci-s7-stop-seed${seed}"
  local -r distill_out="runs/${DISTILL_RUN_NAME}_seed${seed}"
  local -r ppo_out="runs/${PPO_RUN_NAME}_seed${seed}"

  local distill_extra_ovr=""
  local ppo_extra_ovr=""
  if [[ -n "${SNN_OUTPUT_MODE}" ]]; then
    distill_extra_ovr="${distill_extra_ovr} model.snn.output_mode=${SNN_OUTPUT_MODE}"
    ppo_extra_ovr="${ppo_extra_ovr} model.snn.output_mode=${SNN_OUTPUT_MODE}"
  fi
  if [[ -n "${DISTILL_CONF_POWER}" ]]; then
    distill_extra_ovr="${distill_extra_ovr} pretrain.policy_distill.conf_power=${DISTILL_CONF_POWER}"
  fi

  # IMPORTANT: no single quotes in this string (it's wrapped by tmux as a single-quoted command).
  local -r inner_cmd="set -euo pipefail; \
cd ${REMOTE_ROOT}; \
export PYTHONDONTWRITEBYTECODE=1; export PYTHONPATH=src; \
mkdir -p ${distill_out} ${ppo_out}; \
${CONDA} run -n ${ENV_NAME} python -m bci_snn_rl.run_pretrain_stop_lda_policy_distill \
  --config ${CONFIG_DISTILL} \
  --subjects ${SUBJECTS} \
  --variant ${VARIANT} \
  --override \
    project.seed=${seed} \
    project.out_dir=${distill_out} \
    project.overwrite=true \
    ${distill_extra_ovr} \
  2>&1 | tee -a ${distill_out}/exp.log; \
${CONDA} run -n ${ENV_NAME} python -m bci_snn_rl.run_pareto \
  --config ${CONFIG_PARETO} \
  --subjects ${SUBJECTS} \
  --variant ${VARIANT} \
  --override \
    project.seed=${seed} \
    train.total_steps=${TOTAL_STEPS} \
    train.init_checkpoint=${distill_out}/checkpoints/policy_distill_best.pt \
    project.out_dir=${ppo_out} \
    project.overwrite=true \
    ${ppo_extra_ovr} \
  2>&1 | tee -a ${ppo_out}/exp.log; \
echo [s7] done seed=${seed}"

  echo "[s7] start tmux ${sess}"
  if [[ "${RESTART}" == "1" ]]; then
    retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && tmux kill-session -t '${sess}' || true; tmux new -d -s '${sess}' '${inner_cmd}'"
    return 0
  fi

  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && echo '[s7] exists: ${sess} (set RESTART=1 to restart)' || tmux new -d -s '${sess}' '${inner_cmd}'"
}

start_seed 0 || true
start_seed 1 || true
start_seed 2 || true

echo "[s7] started."
echo "  ssh -t ${HOST} 'tmux ls'"
echo "  ssh -t ${HOST} 'tmux attach -t codex-bci-s7-stop-seed0'"
