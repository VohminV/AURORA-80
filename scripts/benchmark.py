from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from aurora.architecture.factory import build_model, set_gradient_checkpointing
from aurora.config import ExperimentConfig, load_config
from aurora.generation import generate
from aurora.inference import model_output
from aurora.utils import (
    parameter_counts,
    resolve_amp_dtype,
    resolve_device,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark AURORA throughput and memory")
    parser.add_argument("--config", default="configs/aurora_100m.yaml")
    parser.add_argument("--output", default="reports/benchmark.json")
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=3)
    parser.add_argument("--generation-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the PyTorch default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        choices=["inductor", "aot_eager", "eager"],
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, dtype_amp: str) -> Any:
    dtype = resolve_amp_dtype(device, dtype_amp)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _peak_vram_mb(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0),
    }


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _hardware(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        info["gpu"] = {
            "name": properties.name,
            "total_memory_mb": properties.total_memory / (1024.0 * 1024.0),
            "capability": f"{properties.major}.{properties.minor}",
        }
    return info


def _model_loss(
    model: nn.Module, config: ExperimentConfig, input_ids: torch.Tensor
) -> torch.Tensor:
    output = model_output(model, input_ids)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    loss = F.cross_entropy(
        output.logits.reshape(-1, config.model.vocab_size),
        targets.reshape(-1),
    )
    loss = loss + output.aux_loss
    if config.model.memory_loss_coef > 0.0:
        loss = loss + config.model.memory_loss_coef * output.memory_loss
    if config.model.consistency_loss_coef > 0.0:
        loss = loss + config.model.consistency_loss_coef * output.consistency_loss
    return loss


def _benchmark_training(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    input_ids: torch.Tensor,
    steps: int,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    if steps <= 0:
        return {"steps": 0}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.adam_eps,
        weight_decay=config.training.weight_decay,
    )
    model.train()
    _reset_peak_vram(device)
    _sync(device)
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.training.dtype_amp):
            loss = _model_loss(model, config, input_ids)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
    _sync(device)
    elapsed = time.perf_counter() - started
    tokens = steps * input_ids.numel()
    result: dict[str, Any] = {
        "steps": steps,
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "seconds_per_step": elapsed / max(steps, 1),
        "peak_rss_mb": _peak_rss_mb(),
    }
    vram = _peak_vram_mb(device)
    if vram is not None:
        result.update(vram)
    return result


@torch.no_grad()
def _benchmark_inference(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    input_ids: torch.Tensor,
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        return {"steps": 0}
    model.eval()
    with _autocast(device, config.training.dtype_amp):
        model_output(model, input_ids)
    _reset_peak_vram(device)
    _sync(device)
    started = time.perf_counter()
    with _autocast(device, config.training.dtype_amp):
        for _ in range(steps):
            model_output(model, input_ids)
    _sync(device)
    elapsed = time.perf_counter() - started
    tokens = steps * input_ids.numel()
    result: dict[str, Any] = {
        "steps": steps,
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "milliseconds_per_token": 1000.0 * elapsed / max(tokens, 1),
        "peak_rss_mb": _peak_rss_mb(),
    }
    vram = _peak_vram_mb(device)
    if vram is not None:
        result.update(vram)
    return result


@torch.no_grad()
def _benchmark_generation(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        return {"new_tokens": 0}
    model.eval()
    _reset_peak_vram(device)
    _sync(device)
    started = time.perf_counter()
    with _autocast(device, config.training.dtype_amp):
        tokens, _ = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
    _sync(device)
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "new_tokens": len(tokens),
        "prompt_tokens": prompt.numel(),
        "seconds": elapsed,
        "tokens_per_second": len(tokens) / max(elapsed, 1e-9),
        "milliseconds_per_token": 1000.0 * elapsed / max(len(tokens), 1),
        "peak_rss_mb": _peak_rss_mb(),
    }
    vram = _peak_vram_mb(device)
    if vram is not None:
        result.update(vram)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    config = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model(config.model).to(device)
    if args.gradient_checkpointing:
        set_gradient_checkpointing(model, True)
    if args.compile:
        model = torch.compile(model, backend=args.compile_backend)  # type: ignore[assignment]
    total_parameters, active_parameters = parameter_counts(model)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda"
        and resolve_amp_dtype(device, config.training.dtype_amp) == torch.float16
        else None
    )

    input_ids = torch.randint(
        0, config.model.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    prompt = torch.randint(0, config.model.vocab_size, (1, min(args.seq_len, 32)), device=device)

    results: dict[str, Any] = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "hardware": _hardware(device),
        "parameters": {"total": total_parameters, "active": active_parameters},
        "settings": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "train_steps": args.train_steps,
            "inference_steps": args.inference_steps,
            "generation_tokens": args.generation_tokens,
            "compile": args.compile,
            "compile_backend": args.compile_backend if args.compile else None,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dtype_amp": config.training.dtype_amp,
        },
        "train": _benchmark_training(model, config, device, input_ids, args.train_steps, scaler),
        "inference": _benchmark_inference(model, config, device, input_ids, args.inference_steps),
        "generation": _benchmark_generation(model, config, device, prompt, args.generation_tokens),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"device: {device}")
    print(f"parameters: {total_parameters:,} total / {active_parameters:,} active")
    train = results["train"]
    if train.get("steps", 0) > 0:
        print(
            f"train: {train['tokens_per_second']:.1f} tok/s, "
            f"{train['seconds_per_step']:.2f} s/step, peak RSS {train['peak_rss_mb']:.0f} MB"
        )
    inference = results["inference"]
    if inference.get("steps", 0) > 0:
        print(
            f"inference: {inference['tokens_per_second']:.1f} tok/s, "
            f"{inference['milliseconds_per_token']:.2f} ms/token"
        )
    generation = results["generation"]
    if generation.get("new_tokens", 0) > 0:
        print(
            f"generation: {generation['tokens_per_second']:.1f} tok/s, "
            f"{generation['milliseconds_per_token']:.2f} ms/token"
        )
    print(f"results -> {output}")


if __name__ == "__main__":
    main()
