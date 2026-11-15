from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.mixture import build_token_mixture
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
    assert checkpoint["format_version"] == 1
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


def test_auto_resume_uses_latest_checkpoint(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 4)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        output_dir=str(tmp_path / "auto_resume_run"),
        eval_interval=0,
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    dataset = MemmapTokenDataset(train_path, training.seq_len)
    Trainer(config, dataset).fit()

    continued = replace(
        training,
        max_steps=training.max_steps + 1,
        auto_resume=True,
    )
    resumed = Trainer(ExperimentConfig(model=config.model, training=continued), dataset)
    resumed.fit()
    assert resumed.step == continued.max_steps
    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == continued.max_steps


def test_max_tokens_stops_training_early(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 6)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=10,
        max_tokens=32,
        eval_interval=0,
        save_interval=1,
        output_dir=str(tmp_path / "budget_run"),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    trainer = Trainer(config, MemmapTokenDataset(train_path, training.seq_len))
    trainer.fit()
    assert trainer.step == 1
    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == 1


def test_early_stopping_patience(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 7)
    _write_tokens(val_path, 8)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path=str(val_path),
        max_steps=10,
        eval_interval=1,
        early_stopping_patience=1,
        save_interval=1,
        output_dir=str(tmp_path / "early_stop_run"),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    trainer = Trainer(
        config,
        MemmapTokenDataset(train_path, training.seq_len),
        MemmapTokenDataset(val_path, training.seq_len),
    )
    trainer.best_val = 1.0
    trainer.fit()
    assert trainer.step == 1
    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == 1


def test_gradient_checkpointing_training(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 9)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        gradient_checkpointing=True,
        output_dir=str(tmp_path / "checkpointed_run"),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    trainer = Trainer(config, MemmapTokenDataset(train_path, training.seq_len))
    assert all(block.gradient_checkpointing for block in trainer.model.blocks)
    trainer.fit()
    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == 2


def test_training_with_weighted_mixture(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    paths = [tmp_path / "ru.bin", tmp_path / "en.bin"]
    for index, path in enumerate(paths):
        _write_tokens(path, 10 + index)
    dataset, sampler, _ = build_token_mixture(
        paths, [0.75, 0.25], seq_len=tiny_experiment_config.training.seq_len, seed=3
    )
    training = replace(
        tiny_experiment_config.training,
        train_path="",
        train_paths=tuple(str(path) for path in paths),
        train_weights=(0.75, 0.25),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        output_dir=str(tmp_path / "mixture_run"),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    Trainer(config, dataset, train_sampler=sampler).fit()
    checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert checkpoint["step"] == 2
