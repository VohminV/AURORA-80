from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aurora.config import ExperimentConfig


def save_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: Any,
    step: int,
    config: ExperimentConfig,
    qat_active: bool,
    best_val: float | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "step": step,
        "config": config.to_dict(),
        "qat_active": qat_active,
        "best_val": best_val,
        "rng_state": rng_state,
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)
