from __future__ import annotations

import pytest

from aurora.config import ExperimentConfig, ModelConfig, TrainingConfig


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_levels=2,
        d_state=16,
        memory_heads=2,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        dropout=0.0,
        tie_embeddings=True,
    )


@pytest.fixture
def tiny_training_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        seq_len=8,
        batch_size=2,
        grad_accum_steps=2,
        max_steps=3,
        warmup_steps=1,
        learning_rate=1e-3,
        eval_interval=2,
        eval_batches=1,
        log_interval=1,
        save_interval=2,
        output_dir=str(tmp_path / "run"),
        device="cpu",
        dtype_amp="fp32",
        seed=7,
    )


@pytest.fixture
def tiny_experiment_config(
    tiny_model_config: ModelConfig, tiny_training_config: TrainingConfig
) -> ExperimentConfig:
    return ExperimentConfig(model=tiny_model_config, training=tiny_training_config)
