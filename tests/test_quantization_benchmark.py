from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aurora.benchmarks.quantization import build_variant, run_quantization_benchmark
from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int, size: int = 512) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=size, dtype=np.uint16).tofile(path)


def _make_checkpoint(tmp_path: Path, config: ExperimentConfig) -> Path:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 11)
    _write_tokens(val_path, 12)
    training = replace(
        config.training,
        train_path=str(train_path),
        val_path=str(val_path),
        max_steps=1,
        warmup_steps=0,
        eval_interval=0,
        save_interval=1,
        output_dir=str(tmp_path / "run"),
    )
    run_config = ExperimentConfig(model=config.model, training=training)
    trainer = Trainer(
        run_config,
        MemmapTokenDataset(train_path, training.seq_len),
        MemmapTokenDataset(val_path, training.seq_len),
    )
    trainer.fit()
    return Path(training.output_dir) / "latest.pt"


def test_quantization_benchmark_compares_variants(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint = _make_checkpoint(tmp_path, tiny_experiment_config)
    report = run_quantization_benchmark(
        checkpoint,
        tmp_path / "val.bin",
        variants=["fp32", "fp16", "ternary", "int8", "int4"],
        group_size=8,
        calibration_steps=2,
        min_features=16,
        seq_len=8,
        batch_size=2,
        max_batches=1,
        device="cpu",
        verify_data=False,
    )
    results = report["results"]
    assert list(results) == ["fp32", "fp16", "ternary", "int8", "int4"]
    assert math.isfinite(results["fp32"]["loss"])
    assert results["fp32"]["quantization"] is None
    assert results["fp32"]["perplexity"] > 0
    assert results["fp16"]["quantization"]["bits_per_weight"] == 16
    assert results["ternary"]["quantization"]["bits_per_weight"] <= 4.0
    assert (
        results["int4"]["quantization"]["estimated_bytes"]
        < results["int8"]["quantization"]["estimated_bytes"]
        < results["fp16"]["quantization"]["estimated_bytes"]
    )
    assert results["int4"]["quantization"]["weight_mse"] > 0
    assert report["checkpoint"]["total_parameters"] > 0
    assert report["checkpoint"]["step"] == 1
    assert report["dataset"]["sha256"]
    assert report["settings"]["group_size"] == 8
    assert report["evaluation"]["environment"]["torch"]


def test_benchmark_validation_and_skips(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint = _make_checkpoint(tmp_path, tiny_experiment_config)
    skipped = run_quantization_benchmark(
        checkpoint,
        tmp_path / "val.bin",
        variants=["fp32", "int8"],
        min_features=10_000,
        seq_len=8,
        batch_size=2,
        max_batches=1,
        device="cpu",
        verify_data=False,
    )
    assert skipped["results"]["int8"]["skipped"] == "no qualifying linear layers"
    assert "perplexity" not in skipped["results"]["int8"]

    with pytest.raises(ValueError, match="Unknown variant"):
        build_variant("fp8", group_size=8, calibration_steps=0)
    with pytest.raises(ValueError, match="At least one variant"):
        run_quantization_benchmark(checkpoint, tmp_path / "val.bin", variants=[])
    with pytest.raises(ValueError, match="group_size"):
        run_quantization_benchmark(checkpoint, tmp_path / "val.bin", group_size=0)
