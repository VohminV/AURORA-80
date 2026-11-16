from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aurora.provenance import utc_now

FORMAT_VERSION = 1

_METRIC_DIRECTIONS = {
    "accuracy": True,
    "perplexity": False,
}


def parse_report_inputs(items: Sequence[str]) -> list[tuple[str, Path]]:
    reports: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Report must be NAME=PATH, got: {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name:
            raise ValueError(f"Report name is empty in: {item!r}")
        if name in seen:
            raise ValueError(f"Duplicate report name: {name}")
        if not path.is_file():
            raise ValueError(f"Report file not found: {path}")
        seen.add(name)
        reports.append((name, path))
    if not reports:
        raise ValueError("At least one --report NAME=PATH is required")
    return reports


def _extract_metrics(report: dict[str, Any], source: Path) -> tuple[str, dict[str, float]]:
    results = report.get("results")
    if isinstance(results, dict) and results:
        values: dict[str, float] = {}
        for task, payload in results.items():
            if not isinstance(payload, dict):
                raise ValueError(f"{source}: result {task!r} is not an object")
            value = payload.get("perplexity")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{source}: result {task!r} has no numeric perplexity")
            values[str(task)] = float(value)
        return "perplexity", values

    tasks = report.get("tasks")
    if isinstance(tasks, dict) and tasks:
        values = {}
        for task, payload in tasks.items():
            if not isinstance(payload, dict):
                raise ValueError(f"{source}: task {task!r} is not an object")
            value = payload.get("accuracy")
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{source}: task {task!r} has no numeric accuracy")
            values[str(task)] = float(value)
        return "accuracy", values

    raise ValueError(
        f"{source}: unrecognized report; expected eval 'results.<domain>.perplexity' "
        "or zero-shot 'tasks.<task>.accuracy'"
    )


def _report_metadata(report: dict[str, Any]) -> dict[str, Any]:
    run_metadata = report.get("run_metadata")
    if not isinstance(run_metadata, dict):
        run_metadata = {}
    git = run_metadata.get("git")
    if not isinstance(git, dict):
        evaluation = report.get("evaluation")
        git = evaluation.get("git") if isinstance(evaluation, dict) else {}
    if not isinstance(git, dict):
        git = {}
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    dataset_versions = report.get("dataset_versions")
    if not isinstance(dataset_versions, list):
        dataset_versions = checkpoint.get("dataset_versions")
    if not isinstance(dataset_versions, list):
        dataset_versions = []
    return {
        "run_id": run_metadata.get("run_id"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "checkpoint_step": checkpoint.get("step"),
        "dataset_versions": [str(version) for version in dataset_versions],
    }


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _std(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate_reports(reports: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    """Aggregate eval or zero-shot reports into mean/std metrics per task."""
    entries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for name, path in reports:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: report must be a JSON object")
        kind, values = _extract_metrics(raw, path)
        for task, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{path}: result {task!r} is not finite")
            grouped[(task, kind)].append(value)
        entries.append(
            {
                "name": name,
                "path": str(path),
                "kind": kind,
                "values": values,
                **_report_metadata(raw),
            }
        )

    metrics: list[dict[str, Any]] = []
    for (task, kind), metric_values in sorted(grouped.items()):
        mean = _mean(metric_values)
        metrics.append(
            {
                "task": task,
                "metric": kind,
                "higher_is_better": _METRIC_DIRECTIONS[kind],
                "count": len(metric_values),
                "mean": mean,
                "std": _std(metric_values, mean),
                "min": min(metric_values),
                "max": max(metric_values),
                "values": metric_values,
            }
        )

    return {
        "format_version": FORMAT_VERSION,
        "created_utc": utc_now(),
        "reports": entries,
        "metrics": metrics,
        "limitations": [
            "Each input report is treated as one independent run; the tool does not "
            "verify that runs share data, compute, tokenizer or seed.",
            "Standard deviation is the sample standard deviation (ddof=1) and is "
            "reported as 0 for a single run.",
            "No significance testing is performed; comparing models requires matched "
            "data, compute and evaluation protocols.",
        ],
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AURORA results summary",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "Mean and sample standard deviation across input reports. Each report is "
        "treated as one run.",
        "",
        "| Task | Metric | N | Mean | Std | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in summary["metrics"]:
        lines.append(
            f"| {_markdown_cell(metric['task'])} | {metric['metric']} | {metric['count']} | "
            f"{metric['mean']:.4f} | {metric['std']:.4f} | {metric['min']:.4f} | "
            f"{metric['max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Report | Kind | Run ID | Git commit | Step | Dataset versions |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for entry in summary["reports"]:
        versions = ", ".join(entry["dataset_versions"]) or "-"
        step = entry["checkpoint_step"]
        lines.append(
            f"| {_markdown_cell(entry['name'])} | {entry['kind']} | "
            f"{_markdown_cell(entry['run_id'] or '-')} | "
            f"{_markdown_cell(entry['git_commit'] or '-')} | "
            f"{_markdown_cell(step if step is not None else '-')} | "
            f"{_markdown_cell(versions)} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate eval and zero-shot JSON reports with mean/std"
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Eval or zero-shot report; repeat for multiple runs/seeds",
    )
    parser.add_argument("--output", default="", help="Optional Markdown output path")
    parser.add_argument("--json", default="", help="Optional aggregated JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        reports = parse_report_inputs(args.report)
        summary = aggregate_reports(reports)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not args.output and not args.json:
        raise SystemExit("At least one of --output or --json is required")

    for metric in summary["metrics"]:
        print(
            f"{metric['task']} ({metric['metric']}): "
            f"{metric['mean']:.4f} +/- {metric['std']:.4f} over {metric['count']} run(s)"
        )

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"json -> {json_path}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(summary), encoding="utf-8")
        print(f"report -> {output}")


if __name__ == "__main__":
    main()
