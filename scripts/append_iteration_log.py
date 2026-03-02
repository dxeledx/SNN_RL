#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _best_lda_row(rows: list[dict[str, float]]) -> dict[str, float]:
    # Max acc, tie-break by kappa.
    return max(rows, key=lambda r: (r.get("acc", float("-inf")), r.get("kappa", float("-inf"))))


def _aggregate_pareto(run_paths: list[Path]) -> dict[float, dict[str, float]]:
    by_tc: dict[float, list[dict[str, float]]] = {}
    for p in run_paths:
        rows = _read_rows(p)
        for r in rows:
            tc = float(r["time_cost"])
            by_tc.setdefault(tc, []).append(r)

    agg: dict[float, dict[str, float]] = {}
    for tc, rs in by_tc.items():
        n = float(len(rs))
        def mean(key: str) -> float:
            return sum(float(r[key]) for r in rs) / n
        def std(key: str) -> float:
            m = mean(key)
            return math.sqrt(sum((float(r[key]) - m) ** 2 for r in rs) / n)
        agg[tc] = {
            "n": float(len(rs)),
            "acc_mean": mean("acc"),
            "acc_std": std("acc"),
            "kappa_mean": mean("kappa"),
            "mdt_mean": mean("mdt_steps_mean"),
            "spike_mean": mean("spike_rate_mean"),
        }
    return agg


def _best_pareto_point(agg: dict[float, dict[str, float]]) -> tuple[float, dict[str, float]]:
    # Max mean acc, tie-break by mean kappa.
    best_tc = max(agg.keys(), key=lambda tc: (agg[tc]["acc_mean"], agg[tc]["kappa_mean"]))
    return float(best_tc), agg[best_tc]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append a short iteration record to docs/ITERATIONS.md")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--change", type=str, default="", help="1-line change summary (optional)")
    p.add_argument("--lda", type=str, required=True, help="Path to LDA baseline csv (lda_threshold_pareto.csv)")
    p.add_argument("--runs", type=str, nargs="+", required=True, help="One or more pareto_summary.csv (seeds)")
    p.add_argument("--out", type=str, default="docs/ITERATIONS.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lda_path = Path(args.lda)
    run_paths = [Path(p) for p in args.runs]
    if not lda_path.is_file():
        raise FileNotFoundError(lda_path)
    for p in run_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    lda_best = _best_lda_row(_read_rows(lda_path))
    agg = _aggregate_pareto(run_paths)
    best_tc, best = _best_pareto_point(agg)

    gap_acc = float(best["acc_mean"]) - float(lda_best["acc"])
    delta_mdt = float(best["mdt_mean"]) - float(lda_best["mdt_steps_mean"])

    change = args.change.strip()
    if not change:
        change = "(see configs/code)"

    entry = "\n".join(
        [
            "",
            f"## {date.today().isoformat()} — {args.tag}",
            f"- Change: {change}",
            (
                f"- LDA best: th={lda_best['threshold']:.2f} acc={lda_best['acc']:.6f} "
                f"kappa={lda_best['kappa']:.6f} MDT={lda_best['mdt_steps_mean']:.3f}"
            ),
            (
                f"- Ours best: tc={best_tc:g} acc={best['acc_mean']:.6f}±{best['acc_std']:.6f} "
                f"kappa={best['kappa_mean']:.6f} MDT={best['mdt_mean']:.3f} spike={best['spike_mean']:.3f}"
            ),
            f"- Gap: acc={gap_acc:+.6f}, MDT={delta_mdt:+.3f} (ours - LDA)",
            "",
        ]
    )

    if out_path.exists():
        prev = out_path.read_text(encoding="utf-8")
        if not prev.endswith("\n"):
            prev = prev + "\n"
        out_path.write_text(prev + entry.lstrip("\n"), encoding="utf-8")
    else:
        out_path.write_text("# Iteration Log\n" + entry.lstrip("\n"), encoding="utf-8")

    print(f"[iterlog] appended -> {out_path}")


if __name__ == "__main__":
    main()
