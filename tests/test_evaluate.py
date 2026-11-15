from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aurora.benchmarks.evaluate import main as evaluate_main
from aurora.benchmarks.evaluate import parse_datasets
from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=1024, dtype=np.uint16).tofile(path)


def _train_tiny_checkpoint(
    tmp_path: Path, config: ExperimentConfig
) -> tuple[Path, ExperimentConfig]:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 11)
    training = replace(
        config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        output_dir=str(tmp_path / "eval_run"),
    )
    resolved = ExperimentConfig(model=config.model, training=training)
    Trainer(resolved, MemmapTokenDataset(train_path, training.seq_len)).fit()
    return Path(training.output_dir) / "latest.pt", resolved


def test_parse_datasets_validates_names_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    _write_tokens(path, 1)
    assert parse_datasets([f"ru={path}"]) == [("ru", path)]
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_datasets([str(path)])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_datasets([f"ru={path}", f"ru={path}"])
    with pytest.raises(ValueError, match="not found"):
        parse_datasets([f"ru={tmp_path / 'missing.bin'}"])


def test_evaluate_writes_multi_domain_report(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint_path, config = _train_tiny_checkpoint(tmp_path, tiny_experiment_config)
    first = tmp_path / "ru.bin"
    second = tmp_path / "code.bin"
    _write_tokens(first, 21)
    _write_tokens(second, 22)
    output = tmp_path / "reports" / "eval.json"

    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--data",
            f"ru={first}",
            "--data",
            f"code={second}",
            "--batch-size",
            "2",
            "--batches",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checkpoint"]["step"] == config.training.max_steps
    assert report["checkpoint"]["architecture"] == "aurora"
    assert report["checkpoint"]["total_parameters"] > 0
    assert report["settings"]["seq_len"] == config.training.seq_len
    assert set(report["results"]) == {"ru", "code"}
    for result in report["results"].values():
        assert result["tokens"] == 2 * config.training.seq_len
        assert result["loss"] > 0.0
        assert result["perplexity"] > 1.0
