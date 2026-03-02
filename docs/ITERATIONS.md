# Iteration Log

简短记录每次迭代的关键改动与结果（便于回溯与写论文）。

格式建议（每次 3–6 行）：
- 日期 + tag
- 改动点（1 行）
- 关键数：`acc/kappa`、`MDT`、`spike_rate`
- 结论：追平/超过谁、差多少、下一步做什么

## 2026-03-02 — S9 membrane policy-distill+PPO
- Change: SNN output_mode=membrane；policy distill(3-action)→PPO pareto；追平 LDA
- LDA best: th=0.80 acc=0.762748 kappa=0.525496 MDT=3.301
- Ours best: tc=0 acc=0.762384±0.003021 kappa=0.524769 MDT=2.269 spike=0.194
- Gap: acc=-0.000364, MDT=-1.033 (ours - LDA)
