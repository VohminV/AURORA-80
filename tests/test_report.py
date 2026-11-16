from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aurora.report import (
    aggregate_reports,
    main,
    parse_report_inputs,
    render_markdown,
)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def _eval_report(ru: float, en: float, run_id: str = "run") -> dict[str, object]:
    return {
        "run_metadata": {
            "run_id": run_id,
            "git": {"commit": "abc123", "branch": "main", "dirty": False},
        },
        "checkpoint": {"step": 100, "dataset_versions": ["sha256:data"]},
        "results": {
            "ru": {"perplexity": ru},
            "en": {"perplexity": en},
        },
    }


def _zero_shot_report(mmlu: float, run_id: str = "run") -> dict[str, object]:
    return {
        "run_metadata": {"run_id": run_id, "git": {"commit": "def456"}},
        "checkpoint": {"step": 200},
        "tasks": {"mmlu": {"accuracy": mmlu}},
    }


def test_parse_report_inputs_validates_names_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    _write_report(path, _eval_report(2.0, 3.0))
    assert parse_report_inputs([f"seed0={path}"]) == [("seed0", path)]
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_report_inputs([str(path)])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_report_inputs([f"seed0={path}", f"seed0={path}"])
    with pytest.raises(ValueError, match="not found"):
        parse_report_inputs([f"seed0={tmp_path / 'missing.json'}"])


def test_aggregate_eval_reports_computes_mean_and_std(tmp_path: Path) -> None:
    first = tmp_path / "seed0.json"
    second = tmp_path / "seed1.json"
    _write_report(first, _eval_report(2.0, 4.0, run_id="run-0"))
    _write_report(second, _eval_report(4.0, 6.0, run_id="run-1"))

    summary = aggregate_reports([("seed0", first), ("seed1", second)])

    metrics = {metric["task"]: metric for metric in summary["metrics"]}
    assert set(metrics) == {"ru", "en"}
    assert metrics["ru"]["metric"] == "perplexity"
    assert metrics["ru"]["higher_is_better"] is False
    assert metrics["ru"]["count"] == 2
    assert metrics["ru"]["mean"] == pytest.approx(3.0)
    assert metrics["ru"]["std"] == pytest.approx(math.sqrt(2.0))
    assert metrics["ru"]["min"] == 2.0
    assert metrics["ru"]["max"] == 4.0
    assert summary["reports"][0]["run_id"] == "run-0"
    assert summary["reports"][0]["git_commit"] == "abc123"
    assert summary["reports"][0]["checkpoint_step"] == 100
    assert summary["reports"][0]["dataset_versions"] == ["sha256:data"]
    assert summary["limitations"]


def test_aggregate_zero_shot_reports(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _write_report(first, _zero_shot_report(0.5))
    _write_report(second, _zero_shot_report(0.7))

    summary = aggregate_reports([("a", first), ("b", second)])

    metric = summary["metrics"][0]
    assert metric["task"] == "mmlu"
    assert metric["metric"] == "accuracy"
    assert metric["higher_is_better"] is True
    assert metric["mean"] == pytest.approx(0.6)
    assert metric["std"] == pytest.approx(0.141421356)
    assert summary["reports"][0]["git_commit"] == "def456"


def test_aggregate_single_report_has_zero_std(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    _write_report(path, _eval_report(2.5, 3.5))

    summary = aggregate_reports([("only", path)])

    assert all(metric["std"] == 0.0 for metric in summary["metrics"])


def test_aggregate_rejects_unrecognized_reports(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    _write_report(path, {"results": {"ru": {"loss": 1.0}}})
    with pytest.raises(ValueError, match="no numeric perplexity"):
        aggregate_reports([("broken", path)])

    _write_report(path, {"something": {}})
    with pytest.raises(ValueError, match="unrecognized report"):
        aggregate_reports([("broken", path)])

    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        aggregate_reports([("broken", path)])


def test_render_markdown_contains_metrics_and_runs(tmp_path: Path) -> None:
    first = tmp_path / "seed0.json"
    second = tmp_path / "seed1.json"
    _write_report(first, _eval_report(2.0, 4.0, run_id="run-0"))
    _write_report(second, _eval_report(4.0, 6.0, run_id="run-1"))

    markdown = render_markdown(aggregate_reports([("seed0", first), ("seed1", second)]))

    assert "# AURORA results summary" in markdown
    assert "| ru | perplexity | 2 | 3.0000 | 1.4142 | 2.0000 | 4.0000 |" in markdown
    assert "| seed0 | perplexity | run-0 | abc123 | 100 | sha256:data |" in markdown
    assert "Limitations" in markdown


def test_report_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    first = tmp_path / "seed0.json"
    second = tmp_path / "seed1.json"
    _write_report(first, _eval_report(2.0, 4.0))
    _write_report(second, _eval_report(4.0, 6.0))
    markdown_path = tmp_path / "reports" / "results.md"
    json_path = tmp_path / "reports" / "results.json"

    main(
        [
            "--report",
            f"seed0={first}",
            "--report",
            f"seed1={second}",
            "--output",
            str(markdown_path),
            "--json",
            str(json_path),
        ]
    )

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["format_version"] == 1
    assert len(summary["metrics"]) == 2
    assert markdown_path.read_text(encoding="utf-8").startswith("# AURORA results summary")
