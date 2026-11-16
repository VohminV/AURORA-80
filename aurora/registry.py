from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from aurora.data.version import sha256_file
from aurora.provenance import utc_now
from aurora.training.checkpoint import validate_checkpoint

FORMAT_VERSION = 1


def _file_metadata(path: Path, verify: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path) if verify else None,
    }


def _checkpoint_entry(path: Path, verify: bool) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_checkpoint(checkpoint, check_finite=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = {}
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        model_config = {}
    training_config = config.get("training")
    if not isinstance(training_config, dict):
        training_config = {}
    run_metadata = checkpoint.get("run_metadata")
    if not isinstance(run_metadata, dict):
        run_metadata = {}
    git = run_metadata.get("git")
    if not isinstance(git, dict):
        git = {}
    dataset = run_metadata.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    dataset_versions = dataset.get("versions")
    if not isinstance(dataset_versions, list):
        dataset_versions = []
    return {
        **_file_metadata(path, verify),
        "kind": "checkpoint",
        "format_version": checkpoint.get("format_version"),
        "step": checkpoint.get("step"),
        "qat_active": checkpoint.get("qat_active"),
        "best_val": checkpoint.get("best_val"),
        "architecture": model_config.get("architecture"),
        "seed": training_config.get("seed"),
        "run_id": run_metadata.get("run_id"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "dataset_versions": [str(version) for version in dataset_versions],
        "metrics": checkpoint.get("metrics"),
        "config": config,
    }


def _report_kind(report: dict[str, Any]) -> str:
    if isinstance(report.get("metrics"), list) and isinstance(report.get("reports"), list):
        return "summary"
    if isinstance(report.get("trials"), list) and isinstance(report.get("configurations"), list):
        return "sweep"
    results = report.get("results")
    if isinstance(results, dict) and results:
        values = [value for value in results.values() if isinstance(value, dict)]
        if values and any("quantization" in value for value in values):
            return "quantization"
        if values and all("perplexity" in value for value in values):
            return "eval"
        if values and all("flags" in value and "validation" in value for value in values):
            return "ablation"
    if isinstance(results, list) and "peak_rss_mb" in report:
        return "state_stability"
    tasks = report.get("tasks")
    if isinstance(tasks, dict) and tasks:
        task_values = [value for value in tasks.values() if isinstance(value, dict)]
        if task_values and all("accuracy" in value for value in task_values):
            return "zero_shot"
    if {"tokenizer", "outputs", "dataset_version"} <= set(report):
        return "dataset_manifest"
    return "unknown"


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _report_summary(report: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "eval":
        metrics: dict[str, Any] = {}
        for task, value in report["results"].items():
            if isinstance(value, dict):
                metrics[str(task)] = {"perplexity": _number(value.get("perplexity"))}
        return {"metrics": metrics}
    if kind == "zero_shot":
        metrics = {}
        for task, value in report["tasks"].items():
            if isinstance(value, dict):
                metrics[str(task)] = {
                    "accuracy": _number(value.get("accuracy")),
                    "correct": value.get("correct"),
                    "evaluated": value.get("evaluated"),
                }
        return {"metrics": metrics}
    if kind == "quantization":
        metrics = {}
        for variant, value in report["results"].items():
            if not isinstance(value, dict):
                continue
            quantization = value.get("quantization")
            if not isinstance(quantization, dict):
                quantization = {}
            metrics[str(variant)] = {
                "perplexity": _number(value.get("perplexity")),
                "bits_per_weight": _number(quantization.get("bits_per_weight")),
            }
        return {"metrics": metrics}
    if kind == "summary":
        metric_rows = [
            {
                "task": metric.get("task"),
                "metric": metric.get("metric"),
                "count": metric.get("count"),
                "mean": _number(metric.get("mean")),
                "std": _number(metric.get("std")),
            }
            for metric in report["metrics"]
            if isinstance(metric, dict)
        ]
        return {"metrics": metric_rows}
    if kind == "state_stability":
        settings = report.get("settings")
        if not isinstance(settings, dict):
            settings = {}
        return {
            "lengths": settings.get("lengths"),
            "peak_rss_mb": _number(report.get("peak_rss_mb")),
        }
    if kind == "sweep":
        best = report.get("best")
        if not isinstance(best, dict):
            best = {}
        summary = report.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        return {
            "trials": len(report["trials"]),
            "completed": summary.get("completed"),
            "failed": summary.get("failed"),
            "best_overrides": best.get("overrides"),
            "best_mean_final_loss": _number(best.get("mean_final_loss")),
        }
    if kind == "dataset_manifest":
        outputs = report.get("outputs")
        if not isinstance(outputs, dict):
            outputs = {}
        return {
            "dataset_version": report.get("dataset_version"),
            "splits": {
                str(split): {
                    "tokens": payload.get("tokens"),
                    "bytes": payload.get("bytes"),
                }
                for split, payload in outputs.items()
                if isinstance(payload, dict)
            },
        }
    return {}


def _report_entry(path: Path, verify: bool) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    kind = _report_kind(report)
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
        **_file_metadata(path, verify),
        "kind": kind,
        "run_id": run_metadata.get("run_id"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "checkpoint_step": checkpoint.get("step"),
        "dataset_versions": [str(version) for version in dataset_versions],
        "summary": _report_summary(report, kind),
    }


def _run_key(entry: dict[str, Any]) -> str:
    run_id = entry.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return f"path:{Path(entry['path']).parent}"


def _collect_paths(paths: Sequence[str | Path], pattern: str) -> list[Path]:
    collected: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir():
            raise ValueError(f"Directory not found: {path}")
        collected.extend(sorted(path.rglob(pattern)))
    return collected


def build_registry(
    run_paths: Sequence[str | Path],
    report_paths: Sequence[str | Path],
    *,
    verify: bool = False,
) -> dict[str, Any]:
    """Collect checkpoint and report artifacts into a run-oriented registry."""
    checkpoints: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in _collect_paths(run_paths, "*.pt"):
        try:
            checkpoints.append(_checkpoint_entry(path, verify))
        except Exception as error:
            errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    for path in _collect_paths(report_paths, "*.json"):
        try:
            reports.append(_report_entry(path, verify))
        except Exception as error:
            errors.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})

    grouped: dict[str, dict[str, Any]] = {}
    for entry in checkpoints + reports:
        key = _run_key(entry)
        run = grouped.setdefault(
            key,
            {
                "run_id": entry.get("run_id"),
                "git_commit": None,
                "dataset_versions": set(),
                "checkpoints": [],
                "reports": [],
            },
        )
        if run["git_commit"] is None and entry.get("git_commit"):
            run["git_commit"] = entry["git_commit"]
        run["dataset_versions"].update(entry.get("dataset_versions") or [])
        collection = "checkpoints" if entry["kind"] == "checkpoint" else "reports"
        run[collection].append(entry["path"])

    runs = []
    for key in sorted(grouped):
        run = grouped[key]
        runs.append(
            {
                "run_id": run["run_id"],
                "group": key,
                "git_commit": run["git_commit"],
                "dataset_versions": sorted(run["dataset_versions"]),
                "checkpoints": run["checkpoints"],
                "reports": run["reports"],
            }
        )

    return {
        "format_version": FORMAT_VERSION,
        "created_utc": utc_now(),
        "settings": {
            "run_paths": [str(path) for path in run_paths],
            "report_paths": [str(path) for path in report_paths],
            "verify": verify,
        },
        "runs": runs,
        "artifacts": checkpoints + reports,
        "errors": errors,
        "limitations": [
            "Checkpoint tensors are loaded with weights_only and mmap but are not "
            "checked for finiteness; registry entries describe metadata, not quality.",
            "SHA-256 is computed only with --verify, which reads every artifact fully.",
            "Reports without a recognized schema are recorded with kind 'unknown'.",
            "Runs are grouped by run_metadata.run_id; legacy artifacts are grouped "
            "by their parent directory.",
        ],
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def render_markdown(registry: dict[str, Any]) -> str:
    artifacts = registry["artifacts"]
    checkpoint_count = sum(1 for artifact in artifacts if artifact["kind"] == "checkpoint")
    lines = [
        "# AURORA experiment registry",
        "",
        f"Generated: {registry['created_utc']}",
        "",
        f"Runs: {len(registry['runs'])} | checkpoints: {checkpoint_count} | "
        f"reports: {len(artifacts) - checkpoint_count} | errors: {len(registry['errors'])}",
        "",
        "## Runs",
        "",
        "| Run ID | Git commit | Dataset versions | Checkpoints | Reports |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run in registry["runs"]:
        versions = ", ".join(run["dataset_versions"]) or "-"
        lines.append(
            f"| {_cell(run['run_id'] or run['group'])} | {_cell(run['git_commit'] or '-')} | "
            f"{_cell(versions)} | {len(run['checkpoints'])} | {len(run['reports'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Path | Kind | Step | Bytes | SHA-256 |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for artifact in artifacts:
        step = artifact.get("step", artifact.get("checkpoint_step"))
        digest = artifact.get("sha256") or "-"
        lines.append(
            f"| {_cell(artifact['path'])} | {artifact['kind']} | "
            f"{_cell(step if step is not None else '-')} | {artifact['bytes']} | "
            f"{_cell(digest)} |"
        )
    if registry["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(
            f"- {_cell(error['path'])}: {_cell(error['error'])}" for error in registry["errors"]
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in registry["limitations"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect checkpoints and evaluation reports into an experiment registry"
    )
    parser.add_argument(
        "--runs",
        action="append",
        default=[],
        metavar="PATH",
        help="Directory scanned recursively for *.pt checkpoints; repeat for multiple roots",
    )
    parser.add_argument(
        "--reports",
        action="append",
        default=[],
        metavar="PATH",
        help="Directory scanned recursively for *.json reports; repeat for multiple roots",
    )
    parser.add_argument("--verify", action="store_true", help="Compute SHA-256 for every artifact")
    parser.add_argument("--output", default="", help="Optional JSON registry path")
    parser.add_argument("--markdown", default="", help="Optional Markdown registry path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_paths = args.runs or ["runs"]
    report_paths = args.reports or ["reports"]
    try:
        registry = build_registry(run_paths, report_paths, verify=args.verify)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.output and not args.markdown:
        raise SystemExit("At least one of --output or --markdown is required")

    print(
        f"runs: {len(registry['runs'])} | artifacts: {len(registry['artifacts'])} | "
        f"errors: {len(registry['errors'])}"
    )
    for failure in registry["errors"]:
        print(f"warning: {failure['path']}: {failure['error']}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"registry -> {output}")
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(registry), encoding="utf-8")
        print(f"markdown -> {markdown}")


if __name__ == "__main__":
    main()
