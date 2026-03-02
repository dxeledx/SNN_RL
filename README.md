# bci-snn-rl

Research infra for **public MI EEG → closed-loop / continuous control** using **SNN + RL** (Phase-0/1 focuses on IV-2b screening→feedback).

## Quickstart (local skeleton)

Create a clean Python env (recommended: conda, Python 3.10–3.12), then:

```bash
pip install -e ".[dev]"
```

## One-command experiment (IV-2b stop-and-decide)

```bash
python -m bci_snn_rl.run_exp --config configs/exp/iv2b_stop_snn_ppo.yaml --subjects 1
```

Outputs are written to `runs/<tag>/` with config snapshot + meta + metrics + figures.

## Remote (lab-external)

Sync code (WAN-safe) and run:

```bash
./scripts/remote_lab_ex_sync.sh
./scripts/remote_lab_ex_run.sh python -m bci_snn_rl.run_exp --config configs/exp/iv2b_stop_snn_ppo.yaml --subjects 1
```

See `docs/REPRO.md` for full reproducibility notes.

