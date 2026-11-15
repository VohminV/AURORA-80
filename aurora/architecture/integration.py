from __future__ import annotations

import torch
from torch import nn

from aurora.architecture.layers import LayerScale


class ResidualIntegration(nn.Module):
    def __init__(self, d_model: int, scale_init: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.scale = LayerScale(d_model, scale_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return residual + self.dropout(self.scale(branch))
