#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

CONDA="${CONDA:-/home/wjx/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-eeg}"
CONFIG="${CONFIG:-configs/exp/iv2b_cursor_snn_ppo.yaml}"

SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
VARIANT="${VARIANT:-iv2b_f8-30_t0-4_w0.5_s0.25_bp3_z1e-12}"
NORM_EPS="${NORM_EPS:-1e-12}"

TOTAL_STEPS="${TOTAL_STEPS:-300000}"
SEEDS="${SEEDS:-0 1 2}"

OUT_PREFIX="${OUT_PREFIX:-runs/iv2b_cursor_snn_ppo_sub1-9_z1e-12}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"

mkdir -p runs

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "[sweep] prepare subjects=${SUBJECTS} variant=${VARIANT} norm_eps=${NORM_EPS}"
  "${CONDA}" run -n "${ENV_NAME}" python -m bci_snn_rl.run_prepare_data \
    --config "${CONFIG}" \
    --subjects "${SUBJECTS}" \
    --variant "${VARIANT}" \
    --override "data.norm_eps=${NORM_EPS}" \
    2>&1 | tee -a "runs/prepare_${VARIANT}.log"
else
  echo "[sweep] skip prepare (SKIP_PREPARE=1)"
fi

for seed in ${SEEDS}; do
  out="${OUT_PREFIX}_seed${seed}"
  echo "[sweep] train+eval seed=${seed} out_dir=${out}"
  mkdir -p "${out}"

  "${CONDA}" run -n "${ENV_NAME}" python -m bci_snn_rl.run_train_rl \
    --config "${CONFIG}" \
    --subjects "${SUBJECTS}" \
    --variant "${VARIANT}" \
    --override \
      "data.norm_eps=${NORM_EPS}" \
      "project.seed=${seed}" \
      "train.total_steps=${TOTAL_STEPS}" \
      "project.out_dir=${out}" \
      "project.overwrite=true" \
    2>&1 | tee -a "${out}/exp.log"

  "${CONDA}" run -n "${ENV_NAME}" python -m bci_snn_rl.run_eval \
    --config "${CONFIG}" \
    --subjects "${SUBJECTS}" \
    --variant "${VARIANT}" \
    --override \
      "data.norm_eps=${NORM_EPS}" \
      "project.out_dir=${out}" \
    2>&1 | tee -a "${out}/exp.log"
done

echo "[sweep] done"

