from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FoldPoint:
    threshold: float
    acc: float
    kappa: float
    mdt_steps_mean: float


def _parse_tag(tag: str) -> tuple[float, str]:
    """
    Tag format produced by `lab_cv_sweep_stop_lda_features.sh`:
      w{window_s}_{bands_name}
    """
    m = re.fullmatch(r"w([0-9]+(?:\.[0-9]+)?)_([a-z0-9]+)", tag)
    if not m:
        raise ValueError(f"Unrecognized tag: {tag}")
    return float(m.group(1)), str(m.group(2))


def _read_fold_csv(path: Path) -> list[FoldPoint]:
    rows: list[FoldPoint] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                FoldPoint(
                    threshold=float(r["threshold"]),
                    acc=float(r["acc"]),
                    kappa=float(r["kappa"]),
                    mdt_steps_mean=float(r["mdt_steps_mean"]),
                )
            )
    if not rows:
        raise ValueError(f"Empty fold csv: {path}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize CV sweep results for Stop LDA threshold baseline.")
    p.add_argument(
        "--root",
        type=str,
        default="runs/cv_sweep_stop_lda",
        help="Root directory containing <tag>/fold_*/eval/lda_threshold_pareto.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    fold_csvs = sorted(root.glob("*/fold_*/eval/lda_threshold_pareto.csv"))
    if not fold_csvs:
        raise FileNotFoundError(f"No fold csv files found under: {root}")

    cfg: dict[str, dict[str, list[FoldPoint]]] = {}
    for p in fold_csvs:
        # .../<tag>/fold_<val>/eval/lda_threshold_pareto.csv
        tag = p.parents[2].name
        fold = p.parents[1].name  # fold_<val>
        cfg.setdefault(tag, {})[fold] = _read_fold_csv(p)

    out_rows: list[dict[str, object]] = []
    best_overall: dict[str, object] | None = None

    for tag, folds in sorted(cfg.items()):
        window_s, bands_name = _parse_tag(tag)
        n_folds = len(folds)
        if n_folds != 3:
            raise ValueError(f"{tag}: expected 3 folds, got {sorted(folds.keys())}")

        by_th: dict[float, list[FoldPoint]] = {}
        for pts in folds.values():
            for pt in pts:
                by_th.setdefault(float(pt.threshold), []).append(pt)

        th_means: list[dict[str, float]] = []
        for th, pts in sorted(by_th.items()):
            if len(pts) != n_folds:
                raise ValueError(f"{tag}: threshold={th} missing folds ({len(pts)}/{n_folds})")
            th_means.append(
                {
                    "threshold": float(th),
                    "acc": sum(p.acc for p in pts) / float(n_folds),
                    "kappa": sum(p.kappa for p in pts) / float(n_folds),
                    "mdt_steps_mean": sum(p.mdt_steps_mean for p in pts) / float(n_folds),
                }
            )

        th_means_sorted = sorted(th_means, key=lambda r: (r["acc"], r["kappa"]), reverse=True)
        best_th = th_means_sorted[0]

        row = {
            "tag": tag,
            "window_s": float(window_s),
            "bands": bands_name,
            "best_threshold": float(best_th["threshold"]),
            "acc": float(best_th["acc"]),
            "kappa": float(best_th["kappa"]),
            "mdt_steps_mean": float(best_th["mdt_steps_mean"]),
        }
        out_rows.append(row)

        if (best_overall is None) or ((row["acc"], row["kappa"]) > (best_overall["acc"], best_overall["kappa"])):  # type: ignore[index]
            best_overall = dict(row)

    out_path = root / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["tag", "window_s", "bands", "best_threshold", "acc", "kappa", "mdt_steps_mean"],
        )
        w.writeheader()
        w.writerows(out_rows)

    assert best_overall is not None
    print(f"[cv_summary] wrote {out_path}")
    print(
        "[cv_best] "
        f"tag={best_overall['tag']} window_s={best_overall['window_s']} bands={best_overall['bands']} "
        f"best_threshold={best_overall['best_threshold']} acc={best_overall['acc']:.4f} "
        f"kappa={best_overall['kappa']:.4f} MDT={best_overall['mdt_steps_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
