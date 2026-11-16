from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(dtype=x.dtype)


def init_module_weights(module: nn.Module, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def scale_residual_outputs(model: nn.Module, n_layers: int) -> None:
    scale = 1.0 / math.sqrt(2.0 * max(n_layers, 1))
    for block in getattr(model, "blocks", []):
        with torch.no_grad():
            block.core.out_proj.weight.mul_(scale)
            if getattr(block, "use_memory", False):
                block.memory.out_proj.weight.mul_(scale)
            if getattr(block, "use_moe", False):
                for expert in block.moe.experts:
                    expert.down_proj.weight.mul_(scale)
