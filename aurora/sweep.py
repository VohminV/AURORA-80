from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from aurora.config import ExperimentConfig, load_config
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.provenance import collect_dataset_sources
from aurora.provenance import environment_metadata, git_metadata, utc_now
from aurora.training.trainer import Trainer
from aurora.utils import parameter_counts, resolve_device

FORMAT_VERSION = 1
DEFAULT_MAX_TRIALS = 128
SPEC_KEYS = frozenset({"base", "parameters", "seeds", "max_steps", "output_dir"})
TRAINING_TUNABLES = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "beta1",
        "beta2",
        "adam_eps",
        "warmup_steps",
        "min_lr_ratio",
        "grad_clip",
    }
)


@dataclass(frozen=True)
class SweepSpec:
    base: str
    parameters: dict[str, tuple[Any, ...]]
    seeds: tuple[int, ...] = ()
    max_steps: int | None = None
    output_dir: str | None = None


@dataclass(frozen=True)
class SweepTrial:
    name: str
    seed: int
    overrides: dict[str, Any]


def _split_parameter(name: str) -> tuple[str, str]:
    if name.startswith("model."):
        field = name[len("model.") :]
        if not field:
            raise ValueError("Model parameter name must not be empty")
        return "model", field
    if name.startswith("training."):
        field = name[len("training.") :]
        if field not in TRAINING_TUNABLES:
            allowed = ", ".join(sorted(TRAINING_TUNABLES))
            raise ValueError(f"Training parameter {name!r} is not tunable; allowed: {allowed}")
        return "training", field
    raise ValueError(f"Parameter {name!r} must start with 'model.' or 'training.'")


def _parse_parameters(raw: Any) -> dict[str, tuple[Any, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("Sweep 'parameters' must be a mapping")
    for key in raw:
        if not isinstance(key, str):
            raise ValueError(f"Parameter name must be a string, got: {key!r}")
    parameters: dict[str, tuple[Any, ...]] = {}
    for key in sorted(raw):
        _split_parameter(key)
        values = raw[key]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"Parameter {key!r} must be a non-empty list of values")
        parameters[key] = tuple(values)
    return parameters


def _parse_seeds(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Sweep 'seeds' must be a list of non-negative integers")
    seeds: list[int] = []
    for seed in raw:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Sweep seeds must be non-negative integers")
        if seed in seeds:
            raise ValueError(f"Duplicate sweep seed: {seed}")
        seeds.append(seed)
    return tuple(seeds)


def parse_sweep_spec(data: Mapping[str, Any]) -> SweepSpec:
    if not isinstance(data, Mapping):
        raise ValueError("Sweep spec root must be a mapping")
    unknown = set(data) - SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown sweep spec fields: {sorted(unknown)}")
    base = data.get("base")
    if not isinstance(base, str) or not base:
        raise ValueError("Sweep spec requires a non-empty 'base' config path")
    max_steps = data.get("max_steps")
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("Sweep 'max_steps' must be a positive integer")
    output_dir = data.get("output_dir")
    if output_dir is not None:
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("Sweep 'output_dir' must be a non-empty path")
    return SweepSpec(
        base=base,
        parameters=_parse_parameters(data.get("parameters")),
        seeds=_parse_seeds(data.get("seeds")),
        max_steps=max_steps,
        output_dir=output_dir,
    )


def load_sweep_spec(path: str | Path) -> SweepSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Sweep spec root must be a mapping")
    return parse_sweep_spec(data)


def _coerce_value(current: Any, value: Any, name: str) -> Any:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} expects a boolean")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} expects an integer")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} expects a number")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{name} expects a string")
        return value
    raise ValueError(f"{name} cannot be tuned")


def apply_overrides(config: ExperimentConfig, overrides: Mapping[str, Any]) -> ExperimentConfig:
    model_updates: dict[str, Any] = {}
    training_updates: dict[str, Any] = {}
    for name in sorted(overrides):
        section_name, field_name = _split_parameter(name)
        section = config.model if section_name == "model" else config.training
        if not hasattr(section, field_name):
            raise ValueError(f"Unknown parameter: {name}")
        value = _coerce_value(getattr(section, field_name), overrides[name], name)
        if section_name == "model":
            model_updates[field_name] = value
        else:
            training_updates[field_name] = value
    model = replace(config.model, **model_updates) if model_updates else config.model
    training = replace(config.training, **training_updates) if training_updates else config.training
    return ExperimentConfig(model=model, training=training)


def build_trials(
    spec: SweepSpec, base_seed: int, max_trials: int = DEFAULT_MAX_TRIALS
) -> list[SweepTrial]:
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    keys = list(spec.parameters)
    combinations = math.prod(len(values) for values in spec.parameters.values())
    if combinations == 0:
        combinations = 1
    seeds = spec.seeds if spec.seeds else (base_seed,)
    total = combinations * len(seeds)
    if total > max_trials:
        raise ValueError(
            f"Sweep expands to {total} trials, which exceeds --max-trials {max_trials}"
        )

    trials: list[SweepTrial] = []
    for combo_index, values in enumerate(
        itertools.product(*(spec.parameters[key] for key in keys))
    ):
        overrides = dict(zip(keys, values, strict=True))
        for seed in seeds:
            trials.append(
                SweepTrial(
                    name=f"trial_{combo_index:03d}_seed{seed}",
                    seed=seed,
                    overrides=overrides,
                )
            )
    return trials


def _require_single_train_path(config: ExperimentConfig, train_path: str | Path | None) -> str:
    if train_path is not None:
        return str(train_path)
    if config.training.train_paths:
        raise ValueError(
            "Sweep expects a single training file; pass --train-path instead of a mixture"
        )
    return config.training.train_path


def _require_val_path(config: ExperimentConfig, val_path: str | Path | None) -> str:
    if val_path is not None:
        return str(val_path)
    return config.training.val_path


def _trial_config(
    config: ExperimentConfig,
    trial: SweepTrial,
    *,
    train_path: str,
    val_path: str,
    output_root: Path,
    max_steps: int,
    device: str,
) -> ExperimentConfig:
    tuned = apply_overrides(config, trial.overrides)
    training = replace(
        tuned.training,
        train_path=train_path,
        train_paths=(),
        train_weights=(),
        val_path=val_path,
        output_dir=str(output_root / trial.name),
        max_steps=max_steps,
        device=device,
        seed=trial.seed,
    )
    return ExperimentConfig(model=tuned.model, training=training)


def _write_config(path: Path, config: ExperimentConfig) -> None:
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _std(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _trial_stem(trial: SweepTrial) -> dict[str, Any]:
    return {
        "name": trial.name,
        "seed": trial.seed,
        "overrides": dict(trial.overrides),
    }


def _summarize_configurations(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in trials:
        if entry.get("status") != "completed":
            continue
        grouped[json.dumps(entry["overrides"], sort_keys=True)].append(entry)

    configurations: list[dict[str, Any]] = []
    for key, group in grouped.items():
        completed = [entry for entry in group if entry["validation"]["final_loss"] is not None]
        losses = [float(entry["validation"]["final_loss"]) for entry in completed]
        perplexities = [
            float(entry["validation"]["final_perplexity"])
            for entry in completed
            if entry["validation"]["final_perplexity"] is not None
        ]
        mean_loss = _mean(losses) if losses else None
        std_loss = _std(losses, mean_loss) if mean_loss is not None else None
        best_entry = (
            min(completed, key=lambda entry: entry["validation"]["final_loss"])
            if completed
            else None
        )
        configurations.append(
            {
                "key": key,
                "overrides": group[0]["overrides"],
                "seeds": sorted({int(entry["seed"]) for entry in group}),
                "runs": len(group),
                "mean_final_loss": mean_loss,
                "std_final_loss": std_loss,
                "mean_final_perplexity": _mean(perplexities) if perplexities else None,
                "best_trial": best_entry["name"] if best_entry is not None else None,
                "best_final_loss": (
                    best_entry["validation"]["final_loss"] if best_entry is not None else None
                ),
            }
        )

    configurations.sort(
        key=lambda item: (
            item["mean_final_loss"] is None,
            item["mean_final_loss"] if item["mean_final_loss"] is not None else math.inf,
            item["key"],
        )
    )
    for configuration in configurations:
        configuration.pop("key")
    return configurations


def _select_best(configurations: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not configurations:
        return None
    top = configurations[0]
    if top["mean_final_loss"] is None:
        return None
    return {
        "overrides": top["overrides"],
        "seeds": top["seeds"],
        "runs": top["runs"],
        "mean_final_loss": top["mean_final_loss"],
        "std_final_loss": top["std_final_loss"],
        "mean_final_perplexity": top["mean_final_perplexity"],
        "best_trial": top["best_trial"],
    }


def _build_report(
    spec: SweepSpec,
    *,
    spec_path: str | Path | None,
    config: ExperimentConfig,
    trials: list[SweepTrial],
    entries: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    configurations = _summarize_configurations(entries)
    best = _select_best(configurations)
    status_counts = {
        status: sum(1 for entry in entries if entry.get("status") == status)
        for status in ("completed", "failed", "planned")
    }
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": utc_now(),
        "spec": str(spec_path) if spec_path is not None else None,
        "base_config": spec.base,
        "architecture": config.model.architecture,
        "settings": settings,
        "summary": {
            "trials": len(trials),
            **status_counts,
        },
        "trials": entries,
        "configurations": configurations,
        "best": best,
        "limitations": [
            "All trials share the same train/validation files and token budget, but "
            "tuning model.* fields changes parameter count and compute per token.",
            "The best configuration is selected by mean final validation loss across "
            "seeds; no significance testing is performed.",
            "Failed trials are recorded and skipped; a clean sweep has zero failures.",
        ],
    }


def run_sweep(
    spec: SweepSpec,
    *,
    spec_path: str | Path | None = None,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    max_steps: int | None = None,
    output_root: str | Path | None = None,
    device: str | None = None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Expand a grid, train every trial and summarize validation results."""
    config = load_config(spec.base)
    if max_steps is not None:
        resolved_max_steps = max_steps
    elif spec.max_steps is not None:
        resolved_max_steps = spec.max_steps
    else:
        resolved_max_steps = config.training.max_steps
    if resolved_max_steps <= 0:
        raise ValueError("max_steps must be positive")
    resolved_device = device if device is not None else config.training.device
    root = Path(output_root) if output_root is not None else Path(spec.output_dir or "runs/sweep")
    resolved_train = _require_single_train_path(config, train_path)
    resolved_val = _require_val_path(config, val_path)
    trials = build_trials(spec, config.training.seed, max_trials)

    tokens_per_step = (
        config.training.batch_size * config.training.grad_accum_steps * config.training.seq_len
    )
    settings = {
        "parameters": {name: list(values) for name, values in spec.parameters.items()},
        "seeds": list(spec.seeds) if spec.seeds else [config.training.seed],
        "max_steps": resolved_max_steps,
        "seq_len": config.training.seq_len,
        "batch_size": config.training.batch_size,
        "grad_accum_steps": config.training.grad_accum_steps,
        "tokens_per_step": tokens_per_step,
        "max_tokens": resolved_max_steps * tokens_per_step,
        "device": str(resolve_device(resolved_device)) if not dry_run else resolved_device,
        "output_root": str(root),
        "max_trials": max_trials,
        "dry_run": dry_run,
        "train_path": resolved_train,
        "val_path": resolved_val,
    }

    entries: list[dict[str, Any]] = []
    if dry_run:
        for trial in trials:
            entry = _trial_stem(trial)
            trial_dir = root / trial.name
            trial_dir.mkdir(parents=True, exist_ok=True)
            try:
                trial_config = _trial_config(
                    config,
                    trial,
                    train_path=resolved_train,
                    val_path=resolved_val,
                    output_root=root,
                    max_steps=resolved_max_steps,
                    device=resolved_device,
                )
            except ValueError as error:
                entry.update({"status": "failed", "error": str(error)})
                entries.append(entry)
                continue
            config_path = trial_dir / "config.yaml"
            _write_config(config_path, trial_config)
            entry.update(
                {
                    "status": "planned",
                    "output_dir": str(trial_dir),
                    "config_path": str(config_path),
                    "config": trial_config.to_dict(),
                }
            )
            entries.append(entry)
        return _build_report(
            spec,
            spec_path=spec_path,
            config=config,
            trials=trials,
            entries=entries,
            settings=settings,
        )

    if not resolved_train:
        raise ValueError("Sweep requires a training token file")
    if not resolved_val:
        raise ValueError("Sweep requires a validation token file")
    train_file = Path(resolved_train)
    val_file = Path(resolved_val)
    if not train_file.is_file():
        raise FileNotFoundError(f"Training token file not found: {train_file}")
    if not val_file.is_file():
        raise FileNotFoundError(f"Validation token file not found: {val_file}")

    dataset_sources = collect_dataset_sources(
        (resolved_train,),
        resolved_val,
        verify_files=config.training.verify_data,
        expected_dtype=config.training.dtype,
    )
    train_dataset = MemmapTokenDataset(
        train_file, config.training.seq_len, dtype=config.training.dtype
    )
    val_dataset = MemmapTokenDataset(val_file, config.training.seq_len, dtype=config.training.dtype)

    resolved = resolve_device(resolved_device)
    for trial in trials:
        entry = _trial_stem(trial)
        trial_dir = root / trial.name
        trial_dir.mkdir(parents=True, exist_ok=True)
        try:
            trial_config = _trial_config(
                config,
                trial,
                train_path=resolved_train,
                val_path=resolved_val,
                output_root=root,
                max_steps=resolved_max_steps,
                device=resolved_device,
            )
        except ValueError as error:
            entry.update({"status": "failed", "error": str(error)})
            entries.append(entry)
            continue

        config_path = trial_dir / "config.yaml"
        _write_config(config_path, trial_config)
        entry.update(
            {
                "output_dir": str(trial_dir),
                "config_path": str(config_path),
                "config": trial_config.to_dict(),
            }
        )
        started = time.perf_counter()
        trainer: Trainer | None = None
        try:
            trainer = Trainer(
                trial_config,
                train_dataset,
                val_dataset,
                dataset_sources=dataset_sources,
            )
            trainer.fit()
            final_loss = trainer.evaluate()
            seconds = time.perf_counter() - started
            final_perplexity = (
                math.exp(min(final_loss, 20.0)) if math.isfinite(final_loss) else None
            )
            total_parameters, active_parameters = parameter_counts(trainer.model)
            entry.update(
                {
                    "status": "completed",
                    "run_id": trainer.run_metadata.get("run_id"),
                    "checkpoint": str(trial_dir / "latest.pt"),
                    "parameters": {
                        "total": total_parameters,
                        "active": active_parameters,
                    },
                    "training": {
                        "steps": trainer.step,
                        "tokens_seen": trainer.tokens_seen,
                        "seconds": seconds,
                        "tokens_per_second": trainer.tokens_seen / max(seconds, 1e-9),
                    },
                    "validation": {
                        "final_loss": (final_loss if math.isfinite(final_loss) else None),
                        "final_perplexity": final_perplexity,
                        "best_loss": (
                            trainer.best_val if math.isfinite(trainer.best_val) else None
                        ),
                    },
                }
            )
        except Exception as error:
            entry.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            if trainer is not None:
                del trainer
        entries.append(entry)

    report = _build_report(
        spec,
        spec_path=spec_path,
        config=config,
        trials=trials,
        entries=entries,
        settings=settings,
    )
    report["evaluation"] = {
        "git": git_metadata(),
        "environment": environment_metadata(resolved),
    }
    report["dataset"] = {"sources": [source.to_dict() for source in dataset_sources]}
    return report


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_overrides(overrides: Mapping[str, Any]) -> str:
    if not overrides:
        return "(base)"
    return ", ".join(f"{name}={overrides[name]}" for name in sorted(overrides))


def render_markdown(report: dict[str, Any]) -> str:
    settings = report["settings"]
    summary = report["summary"]
    lines = [
        "# AURORA hyperparameter sweep",
        "",
        f"Generated: {report['created_utc']}",
        "",
        f"Base config: `{report['base_config']}`  ",
        f"Trials: {summary['trials']} | completed: {summary['completed']} | "
        f"failed: {summary['failed']} | planned: {summary['planned']}",
        "",
    ]
    if settings["dry_run"]:
        lines.extend(["Dry run: configs were generated without training.", ""])
    lines.extend(
        [
            "## Configurations",
            "",
            "Ranked by mean final validation loss across completed seeds.",
            "",
            "| Rank | Parameters | Seeds | Runs | Mean loss | Std | Best trial |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, configuration in enumerate(report["configurations"], start=1):
        mean = configuration["mean_final_loss"]
        std = configuration["std_final_loss"]
        lines.append(
            f"| {rank} | {_cell(_format_overrides(configuration['overrides']))} | "
            f"{_cell(', '.join(str(seed) for seed in configuration['seeds']))} | "
            f"{configuration['runs']} | "
            f"{_cell('-' if mean is None else f'{mean:.4f}')} | "
            f"{_cell('-' if std is None else f'{std:.4f}')} | "
            f"{_cell(configuration['best_trial'] or '-')} |"
        )
    if not report["configurations"]:
        lines.append("| - | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Trials",
            "",
            "| Trial | Seed | Status | Val loss | Perplexity | Tokens | Tok/s |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial in report["trials"]:
        validation = trial.get("validation") or {}
        training = trial.get("training") or {}
        loss = validation.get("final_loss")
        perplexity = validation.get("final_perplexity")
        tokens_per_second = training.get("tokens_per_second")
        tokens_seen = training.get("tokens_seen", "-")
        lines.append(
            f"| {_cell(trial['name'])} | {trial['seed']} | {trial['status']} | "
            f"{_cell('-' if loss is None else f'{loss:.4f}')} | "
            f"{_cell('-' if perplexity is None else f'{perplexity:.2f}')} | "
            f"{_cell(tokens_seen)} | "
            f"{_cell('-' if tokens_per_second is None else f'{tokens_per_second:.1f}')} |"
        )
    errors = [trial for trial in report["trials"] if trial.get("error")]
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {_cell(trial['name'])}: {_cell(trial['error'])}" for trial in errors)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter sweep with per-trial configs and a summary"
    )
    parser.add_argument("--spec", required=True, help="Path to a sweep YAML spec")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate per-trial configs without training",
    )
    parser.add_argument("--output", default="reports/sweep.json")
    parser.add_argument("--markdown", default="", help="Optional Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        spec = load_sweep_spec(args.spec)
        report = run_sweep(
            spec,
            spec_path=args.spec,
            train_path=args.train_path,
            val_path=args.val_path,
            max_steps=args.max_steps,
            output_root=args.output_root,
            device=args.device,
            max_trials=args.max_trials,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not args.output and not args.markdown:
        raise SystemExit("At least one of --output or --markdown is required")

    summary = report["summary"]
    print(
        f"trials: {summary['trials']} | completed: {summary['completed']} | "
        f"failed: {summary['failed']} | planned: {summary['planned']}"
    )
    if report["best"] is not None:
        best = report["best"]
        print(
            f"best: {_format_overrides(best['overrides'])} | "
            f"mean loss {best['mean_final_loss']:.4f} +/- "
            f"{best['std_final_loss']:.4f} over {best['runs']} run(s)"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json -> {output}")
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
        print(f"report -> {markdown}")


if __name__ == "__main__":
    main()
