from __future__ import annotations

import argparse
from pathlib import Path

from bci_snn_rl.data.prepare_iv2b import prepare_subject_iv2b
from bci_snn_rl.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare IV-2b data (MOABB) with strict screening->feedback split")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name (avoid overwriting)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    if args.subjects:
        cfg.setdefault("data", {})["subjects"] = [int(s) for s in args.subjects.split(",") if s.strip()]

    subjects = cfg["data"].get("subjects", [1])
    variant = args.variant

    for subject in subjects:
        out_dir = prepare_subject_iv2b(cfg=cfg, subject=int(subject), variant=variant)
        print(f"[prepare] subject={int(subject):02d} -> {out_dir}")

    print("[prepare] done")


if __name__ == "__main__":
    main()

