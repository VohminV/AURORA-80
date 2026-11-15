from __future__ import annotations

import pytest
import torch

from aurora.architecture.model import AuroraModel
from aurora.config import ExperimentConfig, ModelConfig, load_config
from aurora.utils import parameter_counts


def test_load_smoke_config() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.model.vocab_size == 260
    assert config.training.train_path.endswith("train.bin")


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(
            {
                "model": {"d_model": 32, "unknown": 1},
                "training": {},
            }
        )


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(n_experts=4, top_k=5)


def test_80b_config_is_in_target_range() -> None:
    config = load_config("configs/aurora_80b.yaml")
    with torch.device("meta"):
        model = AuroraModel(config.model)
    total, active = parameter_counts(model)
    assert 75_000_000_000 <= total <= 85_000_000_000
    assert active < total
