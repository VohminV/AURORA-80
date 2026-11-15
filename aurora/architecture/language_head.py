from __future__ import annotations

import torch
from torch import nn

from aurora.architecture.layers import RMSNorm


class LanguageHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        norm_eps: float = 1e-6,
        embedding_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, norm_eps)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
        if embedding_weight is not None:
            self.projection.weight = embedding_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(x))
