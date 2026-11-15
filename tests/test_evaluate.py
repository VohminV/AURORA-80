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
from aurora.data.version import compute_dataset_version, sha256_file
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int, tokens: int = 1024) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=tokens, dtype=np.uint16).tofile(path)


def _write_manifest_dataset(tmp_path: Path, seed: int) -> tuple[Path, str]:
    output_dir = tmp_path / f"prepared_{seed}"
    output_dir.mkdir()
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    _write_tokens(train_path, seed)
    _write_tokens(val_path, seed + 1)
    outputs = {}
    for split, path in (("train", train_path), ("val", val_path)):
        outputs[split] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "tokens": path.stat().st_size // 2,
        }
    manifest = {
        "format_version": 1,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "tokenizer": {
            "kind": "byte",
            "model": None,
            "model_sha256": None,
            "vocab_size": 64,
        },
        "encoding": "utf-8",
        "dtype": "uint16",
        "split_mode": "token",
        "val_fraction": 0.5,
        "test_fraction": 0.0,
        "chunk_size_bytes": 1024,
        "inputs": [],
        "outputs": outputs,
    }
    version = compute_dataset_version(manifest)
    manifest["dataset_version"] = version
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return val_path, version


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
    assert report["checkpoint"]["run_id"] == report["run_metadata"]["run_id"]
    assert report["checkpoint"]["metrics"]["best_val"] is None
    assert report["evaluation"]["environment"]["device"] == "cpu"
    assert isinstance(report["evaluation"]["git"], dict)
    assert report["dataset_versions"] == []
    assert report["settings"]["seq_len"] == config.training.seq_len
    assert set(report["results"]) == {"ru", "code"}
    for result in report["results"].values():
        assert result["tokens"] == 2 * config.training.seq_len
        assert result["loss"] > 0.0
        assert result["perplexity"] > 1.0
        assert result["dataset"]["dataset_version"] is None


def test_evaluate_report_links_manifest_dataset_version(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint_path, config = _train_tiny_checkpoint(tmp_path, tiny_experiment_config)
    val_path, version = _write_manifest_dataset(tmp_path, 41)
    output = tmp_path / "reports" / "eval_manifest.json"

    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--data",
            f"val={val_path}",
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
    result = report["results"]["val"]
    assert report["dataset_versions"] == [version]
    assert result["dataset_version"] == version
    assert result["dataset"]["dataset_version"] == version
    assert result["dataset"]["split"] == "val"
    assert result["dataset"]["verified"] is True
    assert result["manifest"]["dataset_version"] == version
    assert result["manifest"]["split"] == "val"
    assert report["run_metadata"]["dataset"]["sources"][0]["dataset_version"] is None
    assert report["checkpoint"]["dataset_versions"] == list(
        report["run_metadata"]["dataset"]["versions"]
    )
