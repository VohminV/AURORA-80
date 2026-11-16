from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig, load_config
from aurora.inference import inference_autocast, load_model_from_checkpoint, model_output
from aurora.provenance import environment_metadata, git_metadata, utc_now
from aurora.utils import resolve_device, set_seed

DEFAULT_LENGTHS = (4096, 8192, 16384, 32768)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _state_tensors(state: Any) -> list[torch.Tensor]:
    if state is None:
        return []
    layers = getattr(state, "blocks", None)
    if layers is None:
        layers = getattr(state, "layers", None)
    if layers is None:
        return []
    tensors: list[torch.Tensor] = []
    for layer in layers:
        if not is_dataclass(layer):
            continue
        for field in fields(layer):
            value = getattr(layer, field.name)
            if torch.is_tensor(value):
                tensors.append(value)
    return tensors


def _state_stats(state: Any) -> tuple[float, float, bool]:
    tensors = _state_tensors(state)
    if not tensors:
        return 0.0, 0.0, True
    max_abs = 0.0
    squared = 0.0
    finite = True
    for tensor in tensors:
        values = tensor.detach().float()
        if values.numel() == 0:
            continue
        if not bool(torch.isfinite(values).all()):
            finite = False
            continue
        max_abs = max(max_abs, float(values.abs().max()))
        squared += float(values.pow(2).sum())
    return math.sqrt(squared), max_abs, finite


@torch.no_grad()
def measure_state_stability(
    model: torch.nn.Module,
    config: ExperimentConfig,
    lengths: Sequence[int],
    *,
    chunk_size: int = 256,
    reset_interval: int = 0,
    tokens: torch.Tensor | None = None,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    """Run one state continuation per length and report loss/state diagnostics."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if reset_interval < 0:
        raise ValueError("reset_interval must be non-negative")
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must be positive")
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    max_length = max(lengths)
    if tokens is None:
        generator = torch.Generator().manual_seed(seed)
        sequence = torch.randint(
            0,
            config.model.vocab_size,
            (max_length,),
            generator=generator,
        )
    else:
        sequence = tokens.detach().to(torch.long).flatten().cpu()
        if sequence.numel() < max_length:
            raise ValueError(
                f"Token stream has {sequence.numel()} tokens, need at least {max_length}"
            )
        sequence = sequence[:max_length]

    model.eval()
    results: list[dict[str, Any]] = []
    for length in lengths:
        state: Any = None
        total_loss = 0.0
        scored_tokens = 0
        chunks = 0
        first_loss: float | None = None
        last_loss: float | None = None
        first_norm: float | None = None
        final_norm: float | None = None
        max_norm = 0.0
        max_abs = 0.0
        finite = True
        started = time.perf_counter()
        with inference_autocast(resolved_device, config.training.dtype_amp):
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                if reset_interval and start > 0 and start % reset_interval == 0:
                    state = None
                input_ids = sequence[start:end].unsqueeze(0).to(resolved_device)
                output = model_output(model, input_ids, state=state)
                state = output.state
                targets = sequence[start + 1 : end + 1].to(resolved_device)
                if targets.numel() > 0:
                    logits = output.logits[:, : targets.numel(), :]
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, config.model.vocab_size),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    chunk_loss = float(loss_sum) / targets.numel()
                    if not math.isfinite(chunk_loss):
                        finite = False
                        break
                    total_loss += float(loss_sum)
                    scored_tokens += targets.numel()
                    if first_loss is None:
                        first_loss = chunk_loss
                    last_loss = chunk_loss

                chunk_norm, chunk_abs, chunk_finite = _state_stats(state)
                finite = finite and chunk_finite
                max_norm = max(max_norm, chunk_norm)
                max_abs = max(max_abs, chunk_abs)
                if first_norm is None:
                    first_norm = chunk_norm
                final_norm = chunk_norm
                chunks += 1
                if not finite:
                    break
        elapsed = time.perf_counter() - started
        mean_loss = total_loss / scored_tokens if scored_tokens else None
        result: dict[str, Any] = {
            "length": length,
            "chunks": chunks,
            "scored_tokens": scored_tokens,
            "seconds": elapsed,
            "tokens_per_second": length / max(elapsed, 1e-9),
            "finite": finite,
            "loss": mean_loss,
            "perplexity": (
                math.exp(min(mean_loss, 20.0))
                if mean_loss is not None and math.isfinite(mean_loss)
                else None
            ),
            "first_chunk_loss": first_loss,
            "last_chunk_loss": last_loss,
            "state_norm_first": first_norm,
            "state_norm_final": final_norm,
            "state_norm_max": max_norm,
            "state_max_abs": max_abs,
            "state_growth": (
                final_norm / first_norm
                if first_norm is not None and final_norm is not None and first_norm > 0.0
                else None
            ),
            "reset_interval": reset_interval,
        }
        results.append(result)
    return results


def _load_tokens(path: str | Path, dtype: str, count: int) -> torch.Tensor:
    array = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if len(array) < count:
        raise ValueError(f"{path} contains {len(array)} tokens, need at least {count}")
    return torch.from_numpy(np.asarray(array[:count], dtype=np.int64))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check recurrent state stability on long token sequences"
    )
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data", default="", help="Optional tokenized .bin file")
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_LENGTHS),
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=0,
        help="Reset state every N tokens; 0 keeps one state for the whole sequence",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output", default="reports/state_stability.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = resolve_device(args.device)

    try:
        if args.checkpoint:
            model, config = load_model_from_checkpoint(args.checkpoint, device)
        else:
            config = load_config(args.config)
            model = build_model(config.model).to(device)
        tokens = None
        if args.data:
            tokens = _load_tokens(args.data, config.training.dtype, max(args.lengths))
        results = measure_state_stability(
            model,
            config,
            args.lengths,
            chunk_size=args.chunk_size,
            reset_interval=args.reset_interval,
            tokens=tokens,
            seed=args.seed,
            device=device,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    report: dict[str, Any] = {
        "created_utc": utc_now(),
        "config": args.config,
        "checkpoint": args.checkpoint or None,
        "architecture": config.model.architecture,
        "device": str(device),
        "evaluation": {
            "git": git_metadata(),
            "environment": environment_metadata(device),
        },
        "settings": {
            "lengths": args.lengths,
            "chunk_size": args.chunk_size,
            "reset_interval": args.reset_interval,
            "seed": args.seed,
            "data": args.data or None,
        },
        "peak_rss_mb": _peak_rss_mb(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for result in results:
        loss = result["loss"]
        loss_text = f"{loss:.4f}" if loss is not None else "n/a"
        print(
            f"{result['length']:,}: loss {loss_text} | finite {result['finite']} | "
            f"max|state| {result['state_max_abs']:.3e} | "
            f"norm {result['state_norm_final']:.3e} | "
            f"{result['tokens_per_second']:.0f} tok/s"
        )
    print(f"report -> {output}")


if __name__ == "__main__":
    main()
