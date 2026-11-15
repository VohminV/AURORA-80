from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from torch import nn

from aurora.architecture.model import AuroraModel
from aurora.architecture.output import LanguageModelOutput
from aurora.architecture.transformer import TransformerModel
from aurora.config import ModelConfig


class CausalLanguageModel(Protocol):
    def __call__(self, input_ids: torch.Tensor, state: Any = None) -> LanguageModelOutput: ...

    def forward(self, input_ids: torch.Tensor, state: Any = None) -> LanguageModelOutput: ...


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "aurora":
        return AuroraModel(config)
    if config.architecture == "transformer":
        return TransformerModel(config)
    raise ValueError(f"Unknown architecture: {config.architecture}")


def set_gradient_checkpointing(model: nn.Module, enabled: bool = True) -> int:
    """Enable activation checkpointing on every block that supports it."""
    updated = 0
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            cast(Any, module).gradient_checkpointing = enabled
            updated += 1
    return updated
