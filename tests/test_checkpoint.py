from __future__ import annotations

import pytest
import torch

from aurora.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    validate_checkpoint,
)


def _checkpoint() -> dict[str, object]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": {"weight": torch.zeros(2)},
        "optimizer": {},
        "scheduler": {},
        "step": 0,
        "config": {},
        "qat_active": False,
        "run_metadata": {"run_id": "test-run"},
        "metrics": {},
    }


def test_valid_checkpoint_passes_validation() -> None:
    validate_checkpoint(_checkpoint())


def test_v1_checkpoint_without_provenance_still_loads() -> None:
    checkpoint = _checkpoint()
    checkpoint["format_version"] = 1
    del checkpoint["run_metadata"]
    del checkpoint["metrics"]
    validate_checkpoint(checkpoint)


def test_missing_required_key_is_rejected() -> None:
    checkpoint = _checkpoint()
    del checkpoint["optimizer"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_checkpoint(checkpoint)


def test_missing_provenance_is_rejected_for_new_format() -> None:
    checkpoint = _checkpoint()
    del checkpoint["run_metadata"]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_checkpoint(checkpoint)


def test_newer_format_version_is_rejected() -> None:
    checkpoint = _checkpoint()
    checkpoint["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="newer than supported"):
        validate_checkpoint(checkpoint)


def test_non_finite_tensor_is_rejected() -> None:
    checkpoint = _checkpoint()
    checkpoint["model"] = {"weight": torch.tensor([1.0, float("nan")])}
    with pytest.raises(ValueError, match="non-finite"):
        validate_checkpoint(checkpoint)
    validate_checkpoint(checkpoint, check_finite=False)
