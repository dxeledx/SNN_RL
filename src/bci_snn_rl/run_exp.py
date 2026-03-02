from __future__ import annotations

import argparse

from bci_snn_rl.run_eval import main as run_eval_main
from bci_snn_rl.run_prepare_data import main as run_prepare_main
from bci_snn_rl.run_train_rl import main as run_train_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command experiment: prepare -> train -> eval")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", action="extend", nargs="+", default=[], help="YAML override(s), repeatable")
    p.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects override, e.g. 1,2,3")
    p.add_argument("--variant", type=str, default=None, help="Override variant name")
    return p.parse_args()


def main() -> None:
    # Delegate to subcommands by reusing their argv parsing via sys.argv patching.
    # This keeps a single source-of-truth for CLI flags.
    import sys

    args = parse_args()

    base_args = ["--config", args.config]
    if args.override:
        base_args += ["--override", *args.override]
    if args.subjects:
        base_args += ["--subjects", args.subjects]
    if args.variant:
        base_args += ["--variant", args.variant]

    sys.argv = ["run_prepare_data", *base_args]
    run_prepare_main()

    sys.argv = ["run_train_rl", *base_args]
    run_train_main()

    sys.argv = ["run_eval", *base_args]
    run_eval_main()


if __name__ == "__main__":
    main()

