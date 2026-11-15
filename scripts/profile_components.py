from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig, load_config
from aurora.inference import model_output
from aurora.utils import resolve_device, set_seed

DEFAULT_TARGETS = ("RecurrentCore", "AssociativeMemory", "MoE", "CausalSelfAttention")


class BranchTimer:
    def __init__(self, targets: tuple[str, ...] = DEFAULT_TARGETS) -> None:
        self.targets = set(targets)
        self.forward: dict[str, float] = defaultdict(float)
        self.backward: dict[str, float] = defaultdict(float)
        self._handles: list[Any] = []
        self._restore: list[tuple[nn.Module, str, Any]] = []

    def attach(self, model: nn.Module) -> None:
        for module in model.modules():
            name = type(module).__name__
            if name not in self.targets:
                continue
            if name == "AssociativeMemory":
                self._wrap_memory(module)
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_forward_pre(name)))
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._handles.append(
                module.register_full_backward_pre_hook(self._make_backward_pre(name))
            )
            self._handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))

    def _wrap_memory(self, module: nn.Module) -> None:
        original = module.forward_with_losses

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self.forward["AssociativeMemory"] += time.perf_counter() - started
            return result

        self._restore.append((module, "forward_with_losses", original))
        module.forward_with_losses = wrapped  # type: ignore[attr-defined]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, attribute, original in self._restore:
            setattr(module, attribute, original)
        self._restore.clear()

    def reset(self) -> None:
        self.forward.clear()
        self.backward.clear()
        for module_name in self.targets:
            self.forward[module_name] += 0.0
            self.backward[module_name] += 0.0

    @staticmethod
    def _make_forward_pre(name: str) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            del inputs
            module._timer_forward_start = time.perf_counter()  # type: ignore[attr-defined]

        return hook

    def _make_forward_hook(self, name: str) -> Any:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            del inputs, output
            started = getattr(module, "_timer_forward_start", None)
            if started is not None:
                self.forward[name] += time.perf_counter() - started

        return hook

    @staticmethod
    def _make_backward_pre(name: str) -> Any:
        def hook(module: nn.Module, grad_output: tuple[Any, ...]) -> None:
            del grad_output
            module._timer_backward_start = time.perf_counter()  # type: ignore[attr-defined]

        return hook

    def _make_backward_hook(self, name: str) -> Any:
        def hook(
            module: nn.Module,
            grad_input: tuple[Any, ...],
            grad_output: tuple[Any, ...],
        ) -> None:
            del grad_input, grad_output
            started = getattr(module, "_timer_backward_start", None)
            if started is not None:
                self.backward[name] += time.perf_counter() - started

        return hook


def _loss(model: nn.Module, config: ExperimentConfig, input_ids: torch.Tensor) -> torch.Tensor:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile core/memory/MoE branches separately")
    parser.add_argument("--config", default="configs/aurora_100m.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/profile_components.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    config = load_config(args.config)
    device = resolve_device(args.device)
    model = build_model(config.model).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    input_ids = torch.randint(
        0, config.model.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    timer = BranchTimer()
    timer.attach(model)
    total_forward = 0.0
    total_backward = 0.0
    total_step = 0.0
    measured_steps = 0

    try:
        for step in range(args.warmup_steps + args.steps):
            if step == args.warmup_steps:
                timer.reset()
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            forward_started = time.perf_counter()
            loss = _loss(model, config, input_ids)
            forward_elapsed = time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            loss.backward()
            backward_elapsed = time.perf_counter() - backward_started
            optimizer.step()
            total_step += time.perf_counter() - step_started
            if step >= args.warmup_steps:
                total_forward += forward_elapsed
                total_backward += backward_elapsed
                measured_steps += 1
    finally:
        timer.close()

    if measured_steps == 0:
        raise SystemExit("--steps must be positive")

    total_forward /= measured_steps
    total_backward /= measured_steps
    total_step /= measured_steps

    branches: dict[str, dict[str, float | None]] = {}
    for name in timer.targets:
        forward = timer.forward.get(name, 0.0) / measured_steps
        backward = timer.backward.get(name, 0.0) / measured_steps
        if forward == 0.0 and backward == 0.0:
            continue
        branches[name] = {
            "forward_seconds": forward,
            "backward_seconds": backward,
            "forward_share": forward / max(total_forward, 1e-9),
            "backward_share": backward / max(total_backward, 1e-9),
        }

    report: dict[str, Any] = {
        "config": str(args.config),
        "architecture": config.model.architecture,
        "device": str(device),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "steps": measured_steps,
        "warmup_steps": args.warmup_steps,
        "torch_threads": torch.get_num_threads(),
        "total": {
            "forward_seconds": total_forward,
            "backward_seconds": total_backward,
            "step_seconds": total_step,
            "tokens_per_second": args.batch_size * args.seq_len / max(total_step, 1e-9),
        },
        "branches": branches,
        "note": (
            "Forward timings are measured per module. Backward timings are measured "
            "through module hooks; AssociativeMemory uses a direct method call and is "
            "therefore only timed in the forward pass."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"architecture: {config.model.architecture} | device: {device}")
    print(
        f"step: {total_step * 1000:.1f} ms | "
        f"forward {total_forward * 1000:.1f} ms | backward {total_backward * 1000:.1f} ms"
    )
    for name, values in branches.items():
        print(
            f"{name}: forward {values['forward_seconds'] * 1000:.1f} ms "
            f"({values['forward_share'] * 100:.1f}%), "
            f"backward {values['backward_seconds'] * 1000:.1f} ms "
            f"({values['backward_share'] * 100:.1f}%)"
        )
    print(f"results -> {output}")


if __name__ == "__main__":
    main()
