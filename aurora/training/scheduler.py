from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from aurora.config import TrainingConfig


def create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    warmup_steps = config.warmup_steps
    total_steps = config.max_steps
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, float(step - warmup_steps + 1) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
