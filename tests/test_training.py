from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.training.checkpoint import load_checkpoint
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 64, size=4096, dtype=np.uint16)
    tokens.tofile(path)


def test_training_checkpoint_and_resume(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 1)
    _write_tokens(val_path, 2)

    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path=str(val_path),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    train_dataset = MemmapTokenDataset(train_path, training.seq_len)
    val_dataset = MemmapTokenDataset(val_path, training.seq_len)

    trainer = Trainer(config, train_dataset, val_dataset)
    trainer.fit()

    latest = Path(training.output_dir) / "latest.pt"
    best = Path(training.output_dir) / "best.pt"
    assert latest.is_file()
    assert best.is_file()
    checkpoint = load_checkpoint(latest)
    assert checkpoint["step"] == training.max_steps
    assert checkpoint["qat_active"] is False

    resumed_training = replace(training, max_steps=training.max_steps + 1, resume=str(latest))
    resumed_config = ExperimentConfig(model=config.model, training=resumed_training)
    resumed = Trainer(resumed_config, train_dataset, val_dataset)
    resumed.fit()
    resumed_checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert resumed_checkpoint["step"] == resumed_training.max_steps


def test_qat_activation(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 3)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=1,
        output_dir=str(tmp_path / "qat_run"),
        qat_enabled=True,
        qat_start_step=1,
        qat_min_features=16,
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    train_dataset = MemmapTokenDataset(train_path, training.seq_len)

    trainer = Trainer(config, train_dataset)
    trainer.fit()

    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == training.max_steps
    assert checkpoint["qat_active"] is True
    assert any(key.endswith(".linear.weight") for key in checkpoint["model"])
