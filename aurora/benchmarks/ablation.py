from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aurora.config import ExperimentConfig, load_config
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.provenance import collect_dataset_sources
from aurora.provenance import environment_metadata, git_metadata, utc_now
from aurora.training.trainer import Trainer
from aurora.utils import parameter_counts, resolve_device

VARIANTS = ("core", "core_memory", "full")


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_memory: bool
    use_moe: bool


def build_variant(name: str) -> AblationVariant:
    if name == "core":
        return AblationVariant(name, use_memory=False, use_moe=False)
    if name == "core_memory":
        return AblationVariant(name, use_memory=True, use_moe=False)
    if name == "full":
        return AblationVariant(name, use_memory=True, use_moe=True)
    raise ValueError(f"Unknown variant: {name}")


def _resolve_variants(variants: Sequence[str]) -> list[AblationVariant]:
    requested = list(dict.fromkeys(variants))
    if not requested:
        raise ValueError("At least one variant is required")
    return [build_variant(name) for name in requested]


def _require_single_train_path(config: ExperimentConfig, train_path: str | Path | None) -> Path:
    if train_path is not None:
        value = str(train_path)
    elif config.training.train_paths:
        raise ValueError(
            "Ablation expects a single training file; pass --train-path instead of a mixture"
        )
    else:
        value = config.training.train_path
    if not value:
        raise ValueError("Ablation requires a training token file")
    return Path(value)


def _require_val_path(config: ExperimentConfig, val_path: str | Path | None) -> Path:
    value = str(val_path) if val_path is not None else config.training.val_path
    if not value:
        raise ValueError("Ablation requires a validation token file")
    return Path(value)


def _variant_config(
    config: ExperimentConfig,
    variant: AblationVariant,
    *,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    max_steps: int,
    device: str,
    seed: int,
) -> ExperimentConfig:
    model = replace(
        config.model,
        use_memory=variant.use_memory,
        use_moe=variant.use_moe,
    )
    training = replace(
        config.training,
        train_path=str(train_path),
        train_paths=(),
        train_weights=(),
        val_path=str(val_path),
        output_dir=str(output_dir),
        max_steps=max_steps,
        device=device,
        seed=seed,
    )
    return ExperimentConfig(model=model, training=training)


def run_ablation(
    config: ExperimentConfig,
    *,
    variants: Sequence[str] = VARIANTS,
    max_steps: int | None = None,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    output_root: str | Path = "runs/ablation",
    device: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Train core-only, core+memory and full variants with an equal token budget."""
    if config.model.architecture != "aurora":
        raise ValueError("Ablation is defined for the aurora architecture only")
    resolved_variants = _resolve_variants(variants)
    resolved_max_steps = max_steps if max_steps is not None else config.training.max_steps
    if resolved_max_steps <= 0:
        raise ValueError("max_steps must be positive")
    resolved_device = device if device is not None else config.training.device
    resolved_seed = seed if seed is not None else config.training.seed
    resolved_train = _require_single_train_path(config, train_path)
    resolved_val = _require_val_path(config, val_path)
    if not resolved_train.is_file():
        raise FileNotFoundError(f"Training token file not found: {resolved_train}")
    if not resolved_val.is_file():
        raise FileNotFoundError(f"Validation token file not found: {resolved_val}")

    dataset_sources = collect_dataset_sources(
        (str(resolved_train),),
        str(resolved_val),
        verify_files=config.training.verify_data,
        expected_dtype=config.training.dtype,
    )
    train_dataset = MemmapTokenDataset(
        resolved_train,
        config.training.seq_len,
        dtype=config.training.dtype,
    )
    val_dataset = MemmapTokenDataset(
        resolved_val,
        config.training.seq_len,
        dtype=config.training.dtype,
    )
    output_root = Path(output_root)
    tokens_per_step = (
        config.training.batch_size * config.training.grad_accum_steps * config.training.seq_len
    )

    results: dict[str, Any] = {}
    for variant in resolved_variants:
        variant_dir = output_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config = _variant_config(
            config,
            variant,
            train_path=resolved_train,
            val_path=resolved_val,
            output_dir=variant_dir,
            max_steps=resolved_max_steps,
            device=resolved_device,
            seed=resolved_seed,
        )
        started = time.perf_counter()
        trainer = Trainer(
            variant_config,
            train_dataset,
            val_dataset,
            dataset_sources=dataset_sources,
        )
        trainer.fit()
        seconds = time.perf_counter() - started
        final_loss = trainer.evaluate()
        total_parameters, active_parameters = parameter_counts(trainer.model)
        final_perplexity = math.exp(min(final_loss, 20.0)) if math.isfinite(final_loss) else None
        best_loss = trainer.best_val if math.isfinite(trainer.best_val) else None
        results[variant.name] = {
            "use_memory": variant.use_memory,
            "use_moe": variant.use_moe,
            "run_id": trainer.run_metadata.get("run_id"),
            "output_dir": str(variant_dir),
            "checkpoint": str(variant_dir / "latest.pt"),
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
                "final_loss": final_loss if math.isfinite(final_loss) else None,
                "final_perplexity": final_perplexity,
                "best_loss": best_loss,
            },
        }
        del trainer

    baseline = results.get("core", {}).get("validation", {})
    baseline_loss = baseline.get("final_loss")
    baseline_perplexity = baseline.get("final_perplexity")
    if baseline_loss is not None:
        for name, result in results.items():
            validation = result["validation"]
            loss = validation["final_loss"]
            if name == "core" or loss is None:
                continue
            validation["loss_delta_vs_core"] = loss - baseline_loss
            if baseline_perplexity is not None and validation["final_perplexity"] is not None:
                validation["perplexity_ratio_vs_core"] = (
                    validation["final_perplexity"] / baseline_perplexity
                )

    return {
        "created_utc": utc_now(),
        "architecture": config.model.architecture,
        "evaluation": {
            "git": git_metadata(),
            "environment": environment_metadata(resolve_device(resolved_device)),
        },
        "dataset": {"sources": [source.to_dict() for source in dataset_sources]},
        "settings": {
            "variants": [variant.name for variant in resolved_variants],
            "max_steps": resolved_max_steps,
            "seq_len": config.training.seq_len,
            "batch_size": config.training.batch_size,
            "grad_accum_steps": config.training.grad_accum_steps,
            "tokens_per_step": tokens_per_step,
            "max_tokens": resolved_max_steps * tokens_per_step,
            "device": str(resolve_device(resolved_device)),
            "seed": resolved_seed,
            "train_path": str(resolved_train),
            "val_path": str(resolved_val),
            "output_root": str(output_root),
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare core, core+memory and full AURORA configurations"
    )
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--output-root", default="runs/ablation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="reports/ablation.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    try:
        report = run_ablation(
            config,
            variants=args.variants,
            max_steps=args.max_steps,
            train_path=args.train_path,
            val_path=args.val_path,
            output_root=args.output_root,
            device=args.device,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    for name, result in report["results"].items():
        validation = result["validation"]
        loss = validation["final_loss"]
        loss_text = f"{loss:.4f}" if loss is not None else "n/a"
        perplexity = validation["final_perplexity"]
        perplexity_text = f"{perplexity:.2f}" if perplexity is not None else "n/a"
        print(
            f"{name}: loss {loss_text} | perplexity {perplexity_text} | "
            f"params {result['parameters']['total']:,} total / "
            f"{result['parameters']['active']:,} active | "
            f"{result['training']['tokens_per_second']:.0f} tok/s"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report -> {output}")


if __name__ == "__main__":
    main()
