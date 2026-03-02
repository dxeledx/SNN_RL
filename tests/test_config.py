from __future__ import annotations

from pathlib import Path

from bci_snn_rl.utils.config import load_yaml_with_base, parse_overrides


def test_parse_overrides_nested_types() -> None:
    out = parse_overrides(["train.total_steps=123", "data.subjects=[1,2,3]", "task.time_cost=0.01"])
    assert out["train"]["total_steps"] == 123
    assert out["data"]["subjects"] == [1, 2, 3]
    assert abs(out["task"]["time_cost"] - 0.01) < 1e-9


def test_load_yaml_with_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    child.write_text("base: base.yaml\nb:\n  d: 3\n", encoding="utf-8")

    cfg = load_yaml_with_base(child)
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2
    assert cfg["b"]["d"] == 3

