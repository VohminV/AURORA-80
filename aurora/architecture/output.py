from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LanguageModelOutput:
    """Common output of every causal language model in AURORA.

    ``state`` is model-specific: AURORA uses an :class:`AuroraState` with
    recurrent and memory tensors, the Transformer baseline uses a KV-cache.
    """

    logits: torch.Tensor
    aux_loss: torch.Tensor
    memory_loss: torch.Tensor
    consistency_loss: torch.Tensor
    state: Any
