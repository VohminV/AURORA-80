from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from aurora.config import ExperimentConfig

CHECKPOINT_FORMAT_VERSION = 1

REQUIRED_CHECKPOINT_KEYS = (
    "model",
    "optimizer",
    "scheduler",
    "step",
    "config",
    "qat_active",
)


def validate_checkpoint(checkpoint: Any, check_finite: bool = True) -> None:
    """Validate the checkpoint schema and reject corrupted model tensors."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    version = checkpoint.get("format_version")
    if version is not None:
        if not isinstance(version, int) or version < 1:
            raise ValueError(f"Invalid checkpoint format_version: {version!r}")
        if version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format version {version} is newer than supported "
                f"version {CHECKPOINT_FORMAT_VERSION}"
            )

    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint 'model' entry must be a state dict")
    if not check_finite:
        return
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Checkpoint tensor {name!r} contains non-finite values")


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
        "format_version": CHECKPOINT_FORMAT_VERSION,
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


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    validate: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if validate:
        validate_checkpoint(checkpoint)
    return checkpoint
