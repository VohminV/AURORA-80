from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aurora.quantization.qat import DEFAULT_SKIP_PREFIXES, TernaryLinear

WeightTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class QuantizationStats:
    modules: int
    parameters: int
    weight_mse: float
    estimated_bytes: int
    dense_bytes: int

    @property
    def bits_per_weight(self) -> float:
        return self.estimated_bytes * 8.0 / self.parameters if self.parameters else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "modules": self.modules,
            "parameters": self.parameters,
            "weight_mse": self.weight_mse,
            "estimated_bytes": self.estimated_bytes,
            "dense_bytes": self.dense_bytes,
            "bits_per_weight": self.bits_per_weight,
        }


def quantize_symmetric(
    weight: torch.Tensor,
    bits: int = 8,
    group_size: int = 128,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Fake-quantize a tensor with symmetric per-group integer levels."""
    if not 2 <= bits <= 16:
        raise ValueError("bits must be in [2, 16]")
    if weight.numel() == 0:
        raise ValueError("Cannot quantize an empty tensor")
    original_shape = tuple(weight.shape)
    columns = weight.shape[-1]
    effective_group = min(group_size, columns) if group_size > 0 else columns
    flat = weight.detach().float().reshape(-1, columns)
    padding = (-columns) % effective_group
    if padding:
        flat = F.pad(flat, (0, padding))
    groups = flat.reshape(-1, effective_group)
    qmax = float(2 ** (bits - 1) - 1)
    scales = (groups.abs().amax(dim=-1) / qmax).clamp_min(epsilon)
    levels = torch.round(groups / scales.unsqueeze(-1)).clamp(-qmax, qmax)
    values = levels * scales.unsqueeze(-1)
    values = values.reshape(-1, columns + padding)[:, :columns]
    return values.reshape(original_shape)


def quantized_tensor_bytes(
    shape: tuple[int, ...],
    bits: int,
    group_size: int,
    scale_bytes: int = 2,
) -> int:
    """Estimate storage for packed levels plus FP16 group scales."""
    if not 2 <= bits <= 16:
        raise ValueError("bits must be in [2, 16]")
    if len(shape) < 1 or any(dimension <= 0 for dimension in shape):
        raise ValueError("shape must contain positive dimensions")
    if group_size < 0 or scale_bytes < 0:
        raise ValueError("group_size and scale_bytes must be non-negative")
    numel = math.prod(shape)
    columns = shape[-1]
    weight_bytes = math.ceil(numel * bits / 8)
    if group_size == 0:
        return weight_bytes
    rows = numel // columns
    groups = rows * math.ceil(columns / group_size)
    return weight_bytes + groups * scale_bytes


def round_weights_fp16(weight: torch.Tensor) -> torch.Tensor:
    return weight.half().float()


def apply_weight_quantization(
    model: nn.Module,
    transform: WeightTransform,
    *,
    bits: int,
    group_size: int,
    min_features: int = 256,
    skip_prefixes: Iterable[str] = DEFAULT_SKIP_PREFIXES,
) -> QuantizationStats:
    """Apply a weight-only transform in place to eligible ``nn.Linear`` layers."""
    skipped = tuple(skip_prefixes)
    modules = 0
    parameters = 0
    estimated_bytes = 0
    dense_bytes = 0
    squared_error = 0.0
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            raise ValueError("Quantization comparison expects a non-QAT checkpoint")
        if not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in skipped):
            continue
        if min(module.in_features, module.out_features) < min_features:
            continue
        with torch.no_grad():
            original = module.weight.detach().float()
            quantized = transform(original).to(module.weight.dtype)
            squared_error += float((original - quantized.float()).pow(2).sum())
            module.weight.data.copy_(quantized)
        modules += 1
        parameters += original.numel()
        estimated_bytes += quantized_tensor_bytes(
            tuple(original.shape),
            bits,
            min(group_size, original.shape[-1]) if group_size > 0 else 0,
        )
        dense_bytes += original.numel() * original.element_size()
    if parameters == 0:
        return QuantizationStats(0, 0, 0.0, 0, 0)
    return QuantizationStats(
        modules=modules,
        parameters=parameters,
        weight_mse=squared_error / parameters,
        estimated_bytes=estimated_bytes,
        dense_bytes=dense_bytes,
    )
