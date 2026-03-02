<INSTRUCTIONS>
# bci-snn-rl — Research Infra (MI EEG → Closed-loop / Continuous Control)

## Hard constraints (do not violate)

### 1) Strict no-leakage protocol (most important)
- **Training / selection** uses only **training domain** (e.g. IV-2b screening / `train_sessions`).
- **Final reporting** uses only **evaluation domain** (e.g. IV-2b feedback / `eval_sessions`).
- Any normalization / feature scaling must be **fit on train_sessions only** and applied to eval_sessions.
- If requested sessions are missing from MOABB meta, **raise** (never silently fall back).

### 2) Main results are EEG-only
- Default configs must set `data.include_eog=false`.
- EOG may only be used for artifact removal (optional future ablation); never as classification/control input.

## Reproducibility requirements
- Every run directory must contain:
  - `config.snapshot.yaml`
  - `meta.json` (argv, seed, git commit, timestamps)
  - `metrics_train.csv` / `metrics_eval.csv` (as applicable)
  - `checkpoints/{best.pt,last.pt}`

## Remote execution (lab-external)
- Experiments should run on `lab-external`.
- Sync code only (respect `.gitignore`), avoid transferring `data/` and `runs/` over WAN.
</INSTRUCTIONS>

