from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from aurora.architecture.model import AuroraModel
from aurora.config import ExperimentConfig
from aurora.quantization import convert_to_ternary
from aurora.training.checkpoint import load_checkpoint
from aurora.utils import resolve_amp_dtype, resolve_device


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "auto"
) -> tuple[AuroraModel, ExperimentConfig]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])

    model = AuroraModel(config.model).to(resolved_device)
    if checkpoint.get("qat_active", False):
        convert_to_ternary(
            model,
            group_size=config.training.qat_group_size,
            min_features=config.training.qat_min_features,
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def inference_autocast(device: torch.device, dtype_amp: str) -> Any:
    dtype = resolve_amp_dtype(device, dtype_amp)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
