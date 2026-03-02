# Reproducibility / Protocol

## Protocol (no leakage)
- `train_sessions` are used for training + any fitted preprocessing (normalization).
- `eval_sessions` are used only for final reporting.
- If a requested session name does not exist in MOABB meta, the code raises.

## Recommended run (IV-2b)

```bash
python -m bci_snn_rl.run_exp --config configs/exp/iv2b_stop_snn_ppo.yaml --subjects 1
```

## Data cache variants
- Processed data is written to `data/processed/<variant>/...`.
- The default IV-2b variant is derived from config (including `data.norm_eps` via the `_z...` suffix).
- When you change preprocessing, prefer using a new variant so caches never silently overwrite.

## Remote runs (lab-external)

Sync code (WAN-safe, code only):

```bash
bash scripts/remote_lab_ex_sync.sh
```

Start long-running jobs in tmux (so SSH disconnect doesn't kill them):

```bash
# ANN actor (verify PPO pipeline)
ssh lab-external "tmux new -d -s codex-bci-ann-s1 'cd /home/wjx/workspace/Lab/bci-snn-rl && mkdir -p runs && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/wjx/anaconda3/bin/conda run -n eeg python -m bci_snn_rl.run_exp --config configs/exp/iv2b_stop_ann_ppo.yaml --subjects 1 --override train.total_steps=200000 project.overwrite=true 2>&1 | tee -a runs/iv2b_stop_ann_ppo/exp.log'"

# SNN spikefix (make sure spike_rate>0)
ssh lab-external "tmux new -d -s codex-bci-snn-spikefix-s1 'cd /home/wjx/workspace/Lab/bci-snn-rl && mkdir -p runs && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/wjx/anaconda3/bin/conda run -n eeg python -m bci_snn_rl.run_exp --config configs/exp/iv2b_stop_snn_ppo_spikefix.yaml --subjects 1 --override train.total_steps=300000 project.overwrite=true 2>&1 | tee -a runs/iv2b_stop_snn_ppo_spikefix/exp.log'"
```

Three quick checks that it's really running:
- `runs/<out>/metrics_train.csv` keeps growing
- `runs/<out>/checkpoints/best.pt` exists
- `runs/<out>/eval/summary.csv` exists after finishing

## Pareto (accuracy/kappa vs MDT)

```bash
python -m bci_snn_rl.run_pareto --config configs/exp/iv2b_stop_snn_ppo.yaml --subjects 1
```

## Outputs
- `runs/<tag>/config.snapshot.yaml`
- `runs/<tag>/meta.json`
- `runs/<tag>/checkpoints/best.pt`
- `runs/<tag>/checkpoints/last.pt`
- `runs/<tag>/metrics_train.csv`
- `runs/<tag>/eval/metrics_per_subject.csv`
- `runs/<tag>/eval/summary.csv`
- `runs/<tag>/figures/pareto_acc_mdt.png`
- `runs/<tag>/figures/pareto_kappa_mdt.png`
