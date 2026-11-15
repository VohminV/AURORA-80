from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TernaryTensor:
    levels: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int


def _ternary_levels(groups: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.round(groups / scales.unsqueeze(-1)), -1.0, 1.0)


def _reconstruction_error(groups: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    levels = _ternary_levels(groups, scales)
    return (groups - levels * scales.unsqueeze(-1)).pow(2).sum(dim=-1)


def calibrate_group_scales(
    groups: torch.Tensor,
    scales: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    steps: int = 4,
) -> torch.Tensor:
    """Refine per-group scales by alternating assignments and least squares.

    For a fixed ternary assignment the optimal scale is ``sum(x*q) / sum(q*q)``.
    Starting from the mean-absolute scale, each step recomputes the assignment
    and scale, keeping the scale with the lowest reconstruction error. The
    result is never worse than the initial mean-absolute scale.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return scales
    best_scales = scales
    best_error = _reconstruction_error(groups, best_scales)
    current = scales
    for _ in range(steps):
        levels = _ternary_levels(groups, current)
        numerator = (groups * levels).sum(dim=-1)
        denominator = (levels * levels).sum(dim=-1)
        candidate = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            current,
        ).clamp_min(epsilon)
        error = _reconstruction_error(groups, candidate)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scales = torch.where(improved, candidate, best_scales)
        if torch.allclose(candidate, current, rtol=0.0, atol=epsilon):
            break
        current = candidate
    return best_scales


def quantize_ternary(
    weight: torch.Tensor,
    group_size: int = 128,
    epsilon: float = 1e-8,
    *,
    calibration_steps: int = 0,
) -> TernaryTensor:
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
    scales = groups.abs().mean(dim=-1).clamp_min(epsilon)
    if calibration_steps:
        scales = calibrate_group_scales(
            groups,
            scales,
            epsilon=epsilon,
            steps=calibration_steps,
        )
    levels = _ternary_levels(groups, scales)
    levels = levels.reshape(-1, columns + padding)[:, :columns].reshape(original_shape)
    return TernaryTensor(
        levels=levels.to(torch.int8),
        scales=scales.float(),
        shape=original_shape,
        group_size=effective_group,
    )


def dequantize_ternary(quantized: TernaryTensor) -> torch.Tensor:
    columns = quantized.shape[-1]
    flat_levels = quantized.levels.float().reshape(-1, columns)
    padding = (-columns) % quantized.group_size
    if padding:
        flat_levels = F.pad(flat_levels, (0, padding))
    groups = flat_levels.reshape(-1, quantized.group_size)
    scales = quantized.scales.to(groups.device).unsqueeze(-1)
    values = groups * scales
    values = values.reshape(-1, columns + padding)[:, :columns]
    return values.reshape(quantized.shape)


def ternary_ste(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    quantized = quantize_ternary(weight, group_size)
    dequantized = dequantize_ternary(quantized).to(weight.dtype)
    return weight + (dequantized - weight).detach()


def pack_ternary(levels: torch.Tensor) -> tuple[torch.Tensor, int]:
    numel = levels.numel()
    flat = levels.reshape(-1).to(torch.int16)
    padding = (-numel) % 4
    if padding:
        flat = F.pad(flat, (0, padding))
    codes = (flat + 1).to(torch.uint8).reshape(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed, numel


def unpack_ternary(packed: torch.Tensor, numel: int) -> torch.Tensor:
    codes = torch.stack(
        [
            packed & 0b11,
            (packed >> 2) & 0b11,
            (packed >> 4) & 0b11,
            (packed >> 6) & 0b11,
        ],
        dim=-1,
    ).reshape(-1)
    codes = codes[:numel]
    levels = torch.where(codes == 3, torch.zeros_like(codes), codes.to(torch.int16) - 1)
    return levels.to(torch.int8)


def theoretical_bits_per_weight(group_size: int, bits_per_scale: int = 16) -> float:
    return 2.0 + bits_per_scale / group_size


def theoretical_size_bytes(num_parameters: int, group_size: int = 128) -> int:
    return math.ceil(num_parameters * theoretical_bits_per_weight(group_size) / 8.0)
