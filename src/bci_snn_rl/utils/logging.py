from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:
        return None


def save_run_meta(path: str | Path, *, config_path: str, overrides: list[str], seed: int) -> None:
    path = Path(path)
    # Assume command is launched from within the repo; fall back to CWD for git info.
    repo_root = Path.cwd()
    meta = {
        "argv": [str(x) for x in sys.argv],
        "config_path": str(config_path),
        "overrides": list(overrides or []),
        "seed": int(seed),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _read_git_commit(repo_root),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


@dataclass
class CSVLogger:
    path: Path
    _fp: Any | None = None
    _writer: csv.DictWriter[str] | None = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        if self._fp is None:
            self._fp = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fp, fieldnames=list(row.keys()))
            self._writer.writeheader()
        assert self._writer is not None
        self._writer.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None
