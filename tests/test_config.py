from __future__ import annotations

import pytest
import torch

from aurora.architecture.model import AuroraModel
from aurora.config import ExperimentConfig, ModelConfig, TrainingConfig, load_config
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


def test_document_reset_configuration() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.training.document_reset is True
    assert config.training.eos_token_id == 2
    with pytest.raises(ValueError):
        TrainingConfig(document_reset=True, eos_token_id=-1)
    with pytest.raises(ValueError):
        TrainingConfig(eos_token_id=-2)


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(n_experts=4, top_k=5)


def test_negative_auxiliary_loss_coefficients_are_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(memory_loss_coef=-0.1)
    with pytest.raises(ValueError):
        ModelConfig(consistency_loss_coef=-0.1)


def test_invalid_architecture_and_attention_shape_are_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(architecture="mamba")
    with pytest.raises(ValueError):
        ModelConfig(d_model=32, attention_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, attention_heads=6)


def test_transformer_config_loads() -> None:
    config = load_config("configs/transformer_100m.yaml")
    assert config.model.architecture == "transformer"
    assert config.model.d_model == 768
    assert config.model.n_layers == 11


def test_80b_config_is_in_target_range() -> None:
    config = load_config("configs/aurora_80b.yaml")
    with torch.device("meta"):
        model = AuroraModel(config.model)
    total, active = parameter_counts(model)
    assert 75_000_000_000 <= total <= 85_000_000_000
    assert active < total


def test_mixture_config_loads() -> None:
    config = load_config("configs/aurora_100m_mixture.yaml")
    assert config.training.train_path == ""
    assert config.training.train_paths == (
        "data/processed/ru/train.bin",
        "data/processed/en/train.bin",
        "data/processed/code/train.bin",
    )
    assert config.training.train_weights == (0.6, 0.3, 0.1)
