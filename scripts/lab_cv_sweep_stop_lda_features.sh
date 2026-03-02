#!/usr/bin/env bash
set -euo pipefail

# Strict Screening-CV sweep for Stop-and-Decide (LDA threshold closed-loop baseline).
#
# - 3-fold session CV on screening sessions: 0train/1train/2train
# - Sweep window_s ∈ {1,1.5,2,2.5,3}
# - Sweep bands ∈ {bp3 (default), fb6}
# - For each (window_s, bands, fold): prepare_data(train-only norm) -> lda_threshold_pareto
#
# Outputs:
#   runs/cv_sweep_stop_lda/<tag>/fold_<val_session>/eval/lda_threshold_pareto.csv
#
# NOTE: Feedback (3test/4test) is NOT used here.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

CONDA="${CONDA:-/home/wjx/anaconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-eeg}"
CONFIG="${CONFIG:-configs/exp/iv2b_stop_snn_ppo_spikefix.yaml}"

SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9}"
NORM_EPS="${NORM_EPS:-1e-12}"
STRIDE_S="${STRIDE_S:-0.25}"

OUT_ROOT="${OUT_ROOT:-runs/cv_sweep_stop_lda}"

# Fixed sweep grid (decision complete).
WINDOWS_S=(1 1.5 2 2.5 3)

# Band sets.
BANDS_BP3="[[8,12],[13,20],[20,30]]"
BANDS_FB6="[[8,12],[12,16],[16,20],[20,24],[24,28],[28,30]]"

mkdir -p "${OUT_ROOT}"

run_prepare_fold() {
  local -r window_s="$1"
  local -r bands_name="$2"
  local -r n_bands="$3"
  local -r bands_expr="$4"
  local -r train_sessions="$5"
  local -r val_session="$6"

  local -r base_variant="iv2b_f8-30_t0-4_w${window_s}_s${STRIDE_S}_bp${n_bands}_z${NORM_EPS}"
  local -r variant="${base_variant}_cv${val_session}"

  echo "[cv] prepare window_s=${window_s} bands=${bands_name} fold_val=${val_session} variant=${variant}"
  "${CONDA}" run -n "${ENV_NAME}" python -m bci_snn_rl.run_prepare_data \
    --config "${CONFIG}" \
    --subjects "${SUBJECTS}" \
    --variant "${variant}" \
    --override \
      "data.norm_eps=${NORM_EPS}" \
      "dataset.iv2b.window_s=${window_s}" \
      "dataset.iv2b.stride_s=${STRIDE_S}" \
      "dataset.iv2b.bands=${bands_expr}" \
      "dataset.iv2b.train_sessions=${train_sessions}" \
      "dataset.iv2b.eval_sessions=[${val_session}]" \
    2>&1 | tee -a "${OUT_ROOT}/sweep.log"
}

run_lda_fold() {
  local -r window_s="$1"
  local -r bands_name="$2"
  local -r n_bands="$3"
  local -r val_session="$4"

  local -r base_variant="iv2b_f8-30_t0-4_w${window_s}_s${STRIDE_S}_bp${n_bands}_z${NORM_EPS}"
  local -r variant="${base_variant}_cv${val_session}"

  local -r tag="w${window_s}_${bands_name}"
  local -r out_dir="${OUT_ROOT}/${tag}/fold_${val_session}"
  mkdir -p "${out_dir}"

  echo "[cv] lda_pareto window_s=${window_s} bands=${bands_name} fold_val=${val_session} out_dir=${out_dir}"
  "${CONDA}" run -n "${ENV_NAME}" python -m bci_snn_rl.run_baseline_stop_lda_pareto \
    --config "${CONFIG}" \
    --subjects "${SUBJECTS}" \
    --variant "${variant}" \
    --override \
      "data.norm_eps=${NORM_EPS}" \
      "project.out_dir=${out_dir}" \
      "project.overwrite=true" \
    2>&1 | tee -a "${OUT_ROOT}/sweep.log"
}

# 3-fold session CV on screening.
# Fold A: train=1train,2train val=0train
# Fold B: train=0train,2train val=1train
# Fold C: train=0train,1train val=2train
FOLD_TRAINS=(
  "[1train,2train]"
  "[0train,2train]"
  "[0train,1train]"
)
FOLD_VALS=(
  "0train"
  "1train"
  "2train"
)

for window_s in "${WINDOWS_S[@]}"; do
  # bp3
  for fold_i in 0 1 2; do
    run_prepare_fold "${window_s}" "bp3" 3 "${BANDS_BP3}" "${FOLD_TRAINS[$fold_i]}" "${FOLD_VALS[$fold_i]}"
    run_lda_fold "${window_s}" "bp3" 3 "${FOLD_VALS[$fold_i]}"
  done

  # fb6
  for fold_i in 0 1 2; do
    run_prepare_fold "${window_s}" "fb6" 6 "${BANDS_FB6}" "${FOLD_TRAINS[$fold_i]}" "${FOLD_VALS[$fold_i]}"
    run_lda_fold "${window_s}" "fb6" 6 "${FOLD_VALS[$fold_i]}"
  done
done

echo "[cv] done -> ${OUT_ROOT}"

