from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.provenance import resolve_dataset_source
from aurora.inference import inference_autocast, load_model_from_checkpoint, model_output
from aurora.provenance import environment_metadata, git_metadata, utc_now
from aurora.quantization import dequantize_ternary, quantize_ternary
from aurora.quantization.baselines import (
    QuantizationStats,
    WeightTransform,
    apply_weight_quantization,
    quantize_symmetric,
    round_weights_fp16,
)
from aurora.quantization.qat import DEFAULT_SKIP_PREFIXES, TernaryLinear
from aurora.training.checkpoint import load_checkpoint
from aurora.utils import resolve_device

VARIANTS = ("fp32", "fp16", "ternary", "ternary_calibrated", "int8", "int4")


@dataclass(frozen=True)
class QuantizationVariant:
    name: str
    transform: WeightTransform | None
    bits: int
    group_size: int


def build_variant(
    name: str,
    group_size: int,
    calibration_steps: int,
) -> QuantizationVariant:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")
    if name == "fp32":
        return QuantizationVariant(name, None, 32, 0)
    if name == "fp16":
        return QuantizationVariant(name, round_weights_fp16, 16, 0)
    if name == "ternary":
        return QuantizationVariant(
            name,
            lambda weight: dequantize_ternary(quantize_ternary(weight, group_size)),
            2,
            group_size,
        )
    if name == "ternary_calibrated":
        return QuantizationVariant(
            name,
            lambda weight: dequantize_ternary(
                quantize_ternary(weight, group_size, calibration_steps=calibration_steps)
            ),
            2,
            group_size,
        )
    bits = 8 if name == "int8" else 4
    return QuantizationVariant(
        name,
        lambda weight: quantize_symmetric(weight, bits=bits, group_size=group_size),
        bits,
        group_size,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: ExperimentConfig,
    data_path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = MemmapTokenDataset(data_path, seq_len, dtype=config.training.dtype)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    with inference_autocast(device, config.training.dtype_amp):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model_output(model, inputs).logits
            loss = F.cross_entropy(
                logits.reshape(-1, config.model.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += targets.numel()
            batches += 1
            if max_batches > 0 and batches >= max_batches:
                break
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "tokens": total_tokens,
        "batches": batches,
    }


def run_quantization_benchmark(
    checkpoint_path: str | Path,
    data_path: str | Path,
    *,
    variants: Sequence[str] = VARIANTS,
    group_size: int = 128,
    calibration_steps: int = 4,
    min_features: int = 256,
    skip_prefixes: Sequence[str] = DEFAULT_SKIP_PREFIXES,
    seq_len: int | None = None,
    batch_size: int = 4,
    max_batches: int = 0,
    device: str | torch.device = "auto",
    verify_data: bool = True,
) -> dict[str, Any]:
    """Compare weight-only quantization variants on one checkpoint and dataset."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if calibration_steps < 0:
        raise ValueError("calibration_steps must be non-negative")
    if min_features <= 0:
        raise ValueError("min_features must be positive")
    if batch_size <= 0 or max_batches < 0:
        raise ValueError("batch_size must be positive and max_batches non-negative")
    if seq_len is not None and seq_len <= 0:
        raise ValueError("seq_len must be positive")

    requested = list(dict.fromkeys(variants))
    if not requested:
        raise ValueError("At least one variant is required")
    for name in requested:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant: {name}")
    if "fp32" not in requested:
        requested.insert(0, "fp32")

    resolved_device = resolve_device(device) if isinstance(device, str) else device
    probe_model, config = load_model_from_checkpoint(checkpoint_path, resolved_device)
    if any(isinstance(module, TernaryLinear) for module in probe_model.modules()):
        raise ValueError(
            "Quantization comparison expects a non-QAT checkpoint; "
            "export the QAT model to AURORA-TQ instead"
        )
    total_parameters = sum(parameter.numel() for parameter in probe_model.parameters())
    del probe_model
    resolved_seq_len = seq_len or config.training.seq_len
    source = resolve_dataset_source(
        data_path,
        role="eval",
        verify_files=verify_data,
        expected_dtype=config.training.dtype,
    )
    checkpoint = load_checkpoint(checkpoint_path, validate=False)
    run_metadata = checkpoint.get("run_metadata")
    if not isinstance(run_metadata, dict):
        run_metadata = None

    results: dict[str, Any] = {}
    baseline_perplexity: float | None = None
    for name in requested:
        model, _ = load_model_from_checkpoint(checkpoint_path, resolved_device)
        variant = build_variant(name, group_size, calibration_steps)
        stats: QuantizationStats | None = None
        if variant.transform is not None:
            stats = apply_weight_quantization(
                model,
                variant.transform,
                bits=variant.bits,
                group_size=variant.group_size,
                min_features=min_features,
                skip_prefixes=skip_prefixes,
            )
            if stats.modules == 0:
                results[name] = {
                    "skipped": "no qualifying linear layers",
                    "quantization": stats.to_dict(),
                }
                del model
                continue
        metrics = evaluate_model(
            model,
            config,
            data_path,
            seq_len=resolved_seq_len,
            batch_size=batch_size,
            max_batches=max_batches,
            device=resolved_device,
        )
        result: dict[str, Any] = {
            **metrics,
            "quantization": stats.to_dict() if stats is not None else None,
        }
        if baseline_perplexity is None:
            baseline_perplexity = metrics["perplexity"]
        else:
            result["perplexity_delta"] = metrics["perplexity"] - baseline_perplexity
            result["perplexity_ratio"] = metrics["perplexity"] / baseline_perplexity
        results[name] = result
        del model

    report: dict[str, Any] = {
        "created_utc": utc_now(),
        "run_metadata": run_metadata,
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": checkpoint.get("step"),
            "qat_active": checkpoint.get("qat_active"),
            "architecture": config.model.architecture,
            "total_parameters": total_parameters,
            "run_id": run_metadata.get("run_id") if run_metadata else None,
        },
        "evaluation": {
            "git": git_metadata(),
            "environment": environment_metadata(resolved_device),
        },
        "dataset": source.to_dict(),
        "settings": {
            "seq_len": resolved_seq_len,
            "batch_size": batch_size,
            "batches": max_batches,
            "group_size": group_size,
            "calibration_steps": calibration_steps,
            "min_features": min_features,
            "skip_prefixes": list(skip_prefixes),
            "device": str(resolved_device),
            "verify_data": verify_data,
        },
        "results": results,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ternary, integer and FP16 weight-only quantization"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Tokenized .bin file")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
        help="Variants to evaluate; fp32 is always included as the baseline",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-steps", type=int, default=4)
    parser.add_argument("--min-features", type=int, default=256)
    parser.add_argument(
        "--skip-prefix",
        action="append",
        default=None,
        dest="skip_prefixes",
        help="Module-name fragment to leave dense; repeat to override the defaults",
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=0, help="0 means the full file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-verify-data", action="store_true")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = run_quantization_benchmark(
            args.checkpoint,
            args.data,
            variants=args.variants,
            group_size=args.group_size,
            calibration_steps=args.calibration_steps,
            min_features=args.min_features,
            skip_prefixes=(
                tuple(args.skip_prefixes)
                if args.skip_prefixes is not None
                else DEFAULT_SKIP_PREFIXES
            ),
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_batches=args.batches,
            device=args.device,
            verify_data=not args.no_verify_data,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    for name, result in report["results"].items():
        if "skipped" in result:
            print(f"{name}: skipped ({result['skipped']})")
            continue
        quantization = result.get("quantization")
        details = ""
        if quantization is not None:
            details = (
                f" | weight MSE {quantization['weight_mse']:.3e}"
                f" | {quantization['bits_per_weight']:.2f} bit/weight"
            )
        print(f"{name}: loss {result['loss']:.4f} | perplexity {result['perplexity']:.4f}{details}")
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
