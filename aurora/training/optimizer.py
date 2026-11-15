from __future__ import annotations

import torch
from torch.nn import Module
from torch.optim import AdamW

from aurora.config import TrainingConfig


def create_optimizer(model: Module, config: TrainingConfig, device: torch.device) -> AdamW:
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_no_decay = (
            parameter.ndim < 2 or name.endswith(".bias") or "norm" in name or "embed" in name
        )
        if is_no_decay:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    groups = [
        {"params": decay_parameters, "weight_decay": config.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        fused=device.type == "cuda",
    )
