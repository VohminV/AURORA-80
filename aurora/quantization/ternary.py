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


def quantize_ternary(
    weight: torch.Tensor, group_size: int = 128, epsilon: float = 1e-8
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
    levels = torch.clamp(torch.round(groups / scales.unsqueeze(-1)), -1.0, 1.0)
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
