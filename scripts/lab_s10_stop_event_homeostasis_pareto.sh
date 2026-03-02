#!/usr/bin/env bash
set -euo pipefail

# S10: EEG-only sigma-delta (eeg_only wrapper) + spike homeostasis + policy-distill -> PPO Pareto
# - Runs on lab-external
# - Starts 3 tmux sessions (seed0/1/2) in parallel
# - Monitors until all pareto_summary.csv exist, then cleans up tmux sessions

HOST="${HOST:-lab-external}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/home/wjx/workspace/Lab/bci-snn-rl"

SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
VARIANT="${VARIANT:-iv2b_f8-30_t0-4_w3_s0.25_bp6_z1e-12}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
RESTART="${RESTART:-0}" # set to 1 to kill+restart existing sessions

CONDA="${CONDA:-/home/wjx/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-eeg}"

CONFIG_DISTILL="${CONFIG_DISTILL:-configs/exp/iv2b_stop_snn_policy_distill_pretrain_event_sota.yaml}"
CONFIG_PARETO="${CONFIG_PARETO:-configs/exp/iv2b_stop_snn_ppo_event_sota_finetune.yaml}"

retry() {
  local -r max_tries="${1:-3}"
  shift
  local try=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( try >= max_tries )); then
      echo "[s10] failed after ${try} tries: $*" >&2
      return 1
    fi
    try=$((try + 1))
    sleep 2
  done
}

echo "[s10] sync code to ${HOST} ..."
bash "${REPO_ROOT}/scripts/remote_lab_ex_sync.sh"

start_seed() {
  local -r seed="$1"
  local -r sess="codex-bci-s10-stop-seed${seed}"
  local -r distill_out="runs/s10_stop_policy_distill_event_homeo_w3fb6_seed${seed}"
  local -r ppo_out="runs/s10_stop_snn_ppo_event_homeo_w3fb6_seed${seed}"

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
  2>&1 | tee -a ${ppo_out}/exp.log; \
echo [s10] done seed=${seed}"

  echo "[s10] start tmux ${sess}"
  if [[ "${RESTART}" == "1" ]]; then
    retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && tmux kill-session -t '${sess}' || true; tmux new -d -s '${sess}' '${inner_cmd}'"
    return 0
  fi

  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux has-session -t '${sess}' 2>/dev/null && echo '[s10] exists: ${sess} (set RESTART=1 to restart)' || tmux new -d -s '${sess}' '${inner_cmd}'"
}

start_seed 0 || true
start_seed 1 || true
start_seed 2 || true

echo "[s10] monitoring until done ..."

seed_done() {
  local -r seed="$1"
  local -r ppo_out="runs/s10_stop_snn_ppo_event_homeo_w3fb6_seed${seed}"
  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "test -f '${REMOTE_ROOT}/${ppo_out}/pareto/pareto_summary.csv'" >/dev/null 2>&1
}

seed_progress() {
  local -r seed="$1"
  local -r ppo_out="runs/s10_stop_snn_ppo_event_homeo_w3fb6_seed${seed}"
  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "ls -1t '${REMOTE_ROOT}/${ppo_out}/pareto'/tc*/metrics_train.csv 2>/dev/null | head -n 1" 2>/dev/null || true
}

seed_progress_tail() {
  local -r path="$1"
  retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tail -n 1 '${path}'" 2>/dev/null || true
}

while true; do
  done_all=1
  echo "==== $(date '+%F %T') ===="
  for seed in 0 1 2; do
    if seed_done "${seed}"; then
      echo "seed${seed}: DONE"
    else
      done_all=0
      p="$(seed_progress "${seed}")"
      if [[ -n "${p}" ]]; then
        last="$(seed_progress_tail "${p}")"
        echo "seed${seed}: RUNNING (${p}) last=${last}"
      else
        echo "seed${seed}: RUNNING (no metrics yet)"
      fi
    fi
  done
  if [[ "${done_all}" == "1" ]]; then
    break
  fi
  sleep 300
done

echo "[s10] done. cleaning tmux sessions ..."
retry 3 ssh "${SSH_OPTS[@]}" "${HOST}" "tmux kill-session -t codex-bci-s10-stop-seed0 2>/dev/null || true; tmux kill-session -t codex-bci-s10-stop-seed1 2>/dev/null || true; tmux kill-session -t codex-bci-s10-stop-seed2 2>/dev/null || true" || true

echo "[s10] ALL DONE"

