from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
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
from aurora.training.checkpoint import load_checkpoint
from aurora.utils import resolve_device


def parse_datasets(items: Sequence[str]) -> list[tuple[str, Path]]:
    datasets: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Dataset must be NAME=PATH, got: {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name:
            raise ValueError(f"Dataset name is empty in: {item!r}")
        if name in seen:
            raise ValueError(f"Duplicate dataset name: {name}")
        if not path.is_file():
            raise ValueError(f"Dataset file not found: {path}")
        seen.add(name)
        datasets.append((name, path))
    if not datasets:
        raise ValueError("At least one --data NAME=PATH is required")
    return datasets


def _manifest_metadata(
    dataset_version: str | None,
    split: str | None,
    sha256: str | None,
    tokens: int | None,
    manifest_path: str | None,
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    return {
        "path": manifest_path,
        "split": split,
        "sha256": sha256,
        "tokens": tokens,
        "dataset_version": dataset_version,
    }


@torch.no_grad()
def evaluate_file(
    model: torch.nn.Module,
    config: ExperimentConfig,
    path: Path,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    verify_data: bool = True,
) -> dict[str, Any]:
    source = resolve_dataset_source(
        path,
        role="eval",
        verify_files=verify_data,
        expected_dtype=config.training.dtype,
    )
    dataset = MemmapTokenDataset(path, seq_len, dtype=config.training.dtype)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    result: dict[str, Any] = {
        "path": str(path),
        "tokens": total_tokens,
        "batches": batches,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "dataset": source.to_dict(),
    }
    if source.dataset_version is not None:
        result["dataset_version"] = source.dataset_version
    manifest = _manifest_metadata(
        source.dataset_version,
        source.split,
        source.sha256,
        source.tokens,
        source.manifest_path,
    )
    if manifest is not None:
        result["manifest"] = manifest
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an AURORA or Transformer checkpoint on token files"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named tokenized .bin file; repeat for multiple domains",
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
        datasets = parse_datasets(args.data)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device=device)
    seq_len = args.seq_len or config.training.seq_len
    checkpoint = load_checkpoint(args.checkpoint, validate=False)

    results: dict[str, Any] = {}
    for name, path in datasets:
        result = evaluate_file(
            model,
            config,
            path,
            seq_len=seq_len,
            batch_size=args.batch_size,
            max_batches=args.batches,
            device=device,
            verify_data=not args.no_verify_data,
        )
        results[name] = result
        version = result.get("dataset_version")
        version_text = f" | dataset {version}" if version else ""
        print(
            f"{name}: loss {result['loss']:.4f} | perplexity {result['perplexity']:.2f} "
            f"| tokens {result['tokens']:,}{version_text}"
        )

    if args.output:
        run_metadata = checkpoint.get("run_metadata")
        if not isinstance(run_metadata, dict):
            run_metadata = None
        checkpoint_dataset = run_metadata.get("dataset") if run_metadata else None
        if not isinstance(checkpoint_dataset, dict):
            checkpoint_dataset = {}
        dataset_versions = sorted(
            {
                str(result["dataset_version"])
                for result in results.values()
                if result.get("dataset_version")
            }
        )
        report = {
            "created_utc": utc_now(),
            "run_metadata": run_metadata,
            "checkpoint": {
                "path": str(args.checkpoint),
                "step": checkpoint.get("step"),
                "qat_active": checkpoint.get("qat_active"),
                "architecture": config.model.architecture,
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "run_id": run_metadata.get("run_id") if run_metadata else None,
                "dataset_versions": checkpoint_dataset.get("versions", []),
                "metrics": checkpoint.get("metrics"),
            },
            "evaluation": {
                "git": git_metadata(),
                "environment": environment_metadata(device),
            },
            "settings": {
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "batches": args.batches,
                "device": str(device),
                "dtype_amp": config.training.dtype_amp,
                "verify_data": not args.no_verify_data,
            },
            "dataset_versions": dataset_versions,
            "results": results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {output}")


if __name__ == "__main__":
    main()
