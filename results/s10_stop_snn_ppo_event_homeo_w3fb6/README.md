# s10_stop_snn_ppo_event_homeo_w3fb6

Latest stop-and-decide experiment (SNN PPO + event encoder + spike homeostasis), exported as **small, paper-ready artifacts**.

## Setup (from `seed0/config.snapshot.yaml`)

- Dataset: IV-2b
- Subjects: 1–9
- Input: EEG-only (`include_eog: false`)
- Sessions:
  - Train: `0train, 1train, 2train`
  - Eval: `3test, 4test`
- Train:
  - Algo: PPO
  - `total_steps: 500000`
  - Init checkpoint (per-seed): `runs/s10_stop_policy_distill_event_homeo_w3fb6_seed{seed}/checkpoints/policy_distill_best.pt`
- Eval (Pareto sweep): `time_costs = [0.0, 0.005, 0.01, 0.02]`

## Results (mean ± std over seeds 0/1/2)

See `pareto_summary_mean_std.csv`.

| time_cost | acc (mean±std) | kappa (mean±std) | mdt_steps_mean (mean±std) | spike_rate_mean (mean±std) |
|---:|---:|---:|---:|---:|
| 0.000 | 0.7509 ± 0.0031 | 0.5019 ± 0.0063 | 2.5082 ± 0.2454 | 0.1764 ± 0.0051 |
| 0.005 | 0.7494 ± 0.0046 | 0.4987 ± 0.0091 | 2.3387 ± 0.1322 | 0.1782 ± 0.0054 |
| 0.010 | 0.7467 ± 0.0074 | 0.4935 ± 0.0149 | 2.0339 ± 0.1965 | 0.1754 ± 0.0038 |
| 0.020 | 0.7443 ± 0.0030 | 0.4886 ± 0.0060 | 1.8472 ± 0.0606 | 0.1684 ± 0.0028 |

## Files

- Per-seed summaries: `seed{0,1,2}/pareto_summary.csv`
- Per-seed Pareto figures:
  - `seed{0,1,2}/pareto_acc_mdt.png`
  - `seed{0,1,2}/pareto_kappa_mdt.png`
- Repro context:
  - `seed{0,1,2}/config.snapshot.yaml`
  - `seed{0,1,2}/meta.json`

