from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from aurora.quantization.ternary import ternary_ste

DEFAULT_SKIP_PREFIXES = ("embed", "lm_head", "router")


class TernaryLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group_size: int = 128) -> None:
        super().__init__()
        self.linear = linear
        self.group_size = group_size

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = ternary_ste(self.linear.weight, self.group_size)
        return F.linear(x, weight, self.linear.bias)


def convert_to_ternary(
    module: nn.Module,
    group_size: int = 128,
    min_features: int = 256,
    skip_prefixes: Iterable[str] = DEFAULT_SKIP_PREFIXES,
    prefix: str = "",
) -> int:
    skipped = tuple(skip_prefixes)
    converted = 0
    for name, child in module.named_children():
        if isinstance(child, TernaryLinear):
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        should_skip = any(token in full_name for token in skipped)
        if isinstance(child, nn.Linear):
            small = min(child.in_features, child.out_features) < min_features
            if not should_skip and not small:
                setattr(module, name, TernaryLinear(child, group_size))
                converted += 1
            continue
        converted += convert_to_ternary(
            child,
            group_size=group_size,
            min_features=min_features,
            skip_prefixes=skipped,
            prefix=full_name,
        )
    return converted
