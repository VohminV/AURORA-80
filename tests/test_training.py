from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.mixture import build_token_mixture
from aurora.training.checkpoint import CHECKPOINT_FORMAT_VERSION, load_checkpoint
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 64, size=4096, dtype=np.uint16)
    tokens.tofile(path)


def _write_document_tokens(path: Path) -> None:
    pattern = np.array([10, 11, 12, 2, 20, 21, 22, 2], dtype=np.uint16)
    np.tile(pattern, 512).tofile(path)


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
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["step"] == training.max_steps
    assert checkpoint["qat_active"] is False
    run_metadata = checkpoint["run_metadata"]
    assert run_metadata["run_id"]
    assert run_metadata["started_utc"]
    assert "commit" in run_metadata["git"]
    assert run_metadata["environment"]["torch"]
    assert run_metadata["dataset"]["fingerprint"].startswith("sha256:")
    assert run_metadata["dataset"]["manifest_linked"] is False
    assert [source["role"] for source in run_metadata["dataset"]["sources"]] == ["train", "val"]
    assert checkpoint["metrics"]["last_train"] is not None
    assert checkpoint["metrics"]["last_validation"] is not None

    metrics_path = Path(training.output_dir) / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert {"train", "validation"} <= {record["type"] for record in records}
    assert {record["run_id"] for record in records} == {run_metadata["run_id"]}

    resumed_training = replace(training, max_steps=training.max_steps + 1, resume=str(latest))
    resumed_config = ExperimentConfig(model=config.model, training=resumed_training)
    resumed = Trainer(resumed_config, train_dataset, val_dataset)
    resumed.fit()
    resumed_checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert resumed_checkpoint["step"] == resumed_training.max_steps
    assert resumed_checkpoint["run_metadata"]["run_id"] == run_metadata["run_id"]
    assert resumed_checkpoint["run_metadata"]["resume_count"] == 1
    assert resumed_checkpoint["run_metadata"]["original_git"] == run_metadata["git"]
    assert resumed_checkpoint["run_metadata"]["original_environment"] == run_metadata["environment"]
    assert resumed_checkpoint["run_metadata"]["resumed_from"] == str(latest)

    second_training = replace(training, max_steps=training.max_steps + 2, resume=str(latest))
    second_config = ExperimentConfig(model=config.model, training=second_training)
    second = Trainer(second_config, train_dataset, val_dataset)
    second.fit()
    second_checkpoint = load_checkpoint(Path(training.output_dir) / "latest.pt")
    assert second_checkpoint["step"] == second_training.max_steps
    assert second_checkpoint["run_metadata"]["run_id"] == run_metadata["run_id"]
    assert second_checkpoint["run_metadata"]["resume_count"] == 2
    assert second_checkpoint["run_metadata"]["original_git"] == run_metadata["git"]
    assert second_checkpoint["run_metadata"]["original_environment"] == run_metadata["environment"]

    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert "resume" in {record["type"] for record in records}
    assert {record["run_id"] for record in records} == {run_metadata["run_id"]}


def test_resume_rejects_changed_dataset(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 31)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        output_dir=str(tmp_path / "provenance_run"),
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    Trainer(config, MemmapTokenDataset(train_path, training.seq_len)).fit()
    latest = Path(training.output_dir) / "latest.pt"

    _write_tokens(train_path, 32)
    resumed_training = replace(training, max_steps=3, resume=str(latest))
    resumed_config = ExperimentConfig(model=config.model, training=resumed_training)
    resumed = Trainer(
        resumed_config,
        MemmapTokenDataset(train_path, resumed_training.seq_len),
    )
    with pytest.raises(ValueError, match="Dataset provenance mismatch"):
        resumed.fit()


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


def test_training_passes_document_reset_masks(
    tmp_path: Path,
    tiny_experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_document_tokens(train_path)
    _write_document_tokens(val_path)
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path=str(val_path),
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        save_interval=2,
        output_dir=str(tmp_path / "document_reset_run"),
        document_reset=True,
        eos_token_id=2,
    )
    config = ExperimentConfig(model=tiny_experiment_config.model, training=training)
    trainer = Trainer(
        config,
        MemmapTokenDataset(train_path, training.seq_len),
        MemmapTokenDataset(val_path, training.seq_len),
    )
    seen: list[torch.Tensor | None] = []
    original_forward = trainer.model.forward

    def spy(
        input_ids: torch.Tensor,
        state: Any = None,
        reset_mask: torch.Tensor | None = None,
    ) -> Any:
        seen.append(reset_mask)
        return original_forward(input_ids, state=state, reset_mask=reset_mask)

    monkeypatch.setattr(trainer.model, "forward", spy)
    trainer.fit()

    assert seen
    assert all(mask is not None for mask in seen)
    assert all(
        mask.shape == (training.batch_size, training.seq_len) for mask in seen if mask is not None
    )
    assert any(bool(mask.any()) for mask in seen if mask is not None)


def test_document_reset_rejects_transformer(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_document_tokens(train_path)
    model = replace(
        tiny_experiment_config.model,
        architecture="transformer",
        attention_heads=2,
    )
    training = replace(
        tiny_experiment_config.training,
        train_path=str(train_path),
        val_path="",
        document_reset=True,
    )
    config = ExperimentConfig(model=model, training=training)
    with pytest.raises(ValueError, match="only supported for the aurora architecture"):
        Trainer(config, MemmapTokenDataset(train_path, training.seq_len))
