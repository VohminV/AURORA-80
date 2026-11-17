from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from aurora.architecture.factory import CausalLanguageModel, build_model
from aurora.config import ExperimentConfig
from aurora.data.dataset import document_reset_mask
from aurora.quantization import convert_to_ternary
from aurora.training.checkpoint import load_checkpoint
from aurora.utils import resolve_amp_dtype, resolve_device


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "auto"
) -> tuple[nn.Module, ExperimentConfig]:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])

    model = build_model(config.model).to(resolved_device)
    if checkpoint.get("qat_active", False):
        convert_to_ternary(
            model,
            group_size=config.training.qat_group_size,
            min_features=config.training.qat_min_features,
        )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def reset_mask_for_config(input_ids: torch.Tensor, config: ExperimentConfig) -> torch.Tensor | None:
    if not config.training.document_reset:
        return None
    return document_reset_mask(input_ids, config.training.eos_token_id)


def model_output(
    model: nn.Module,
    input_ids: torch.Tensor,
    state: Any = None,
    reset_mask: torch.Tensor | None = None,
) -> Any:
    language_model = cast(CausalLanguageModel, model)
    if reset_mask is None:
        return language_model(input_ids, state=state)
    return language_model(input_ids, state=state, reset_mask=reset_mask)


def inference_autocast(device: torch.device, dtype_amp: str) -> Any:
    dtype = resolve_amp_dtype(device, dtype_amp)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
