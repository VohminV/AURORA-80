from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig
from aurora.model_card import main, render_model_card
from aurora.training.checkpoint import save_checkpoint
from aurora.utils import parameter_counts


def _save_checkpoint(
    path: Path,
    config: ExperimentConfig,
    *,
    qat_active: bool = False,
) -> dict[str, object]:
    model = build_model(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    run_metadata = {
        "run_id": "model-card-run",
        "started_utc": "2026-01-01T00:00:00+00:00",
        "resume_count": 1,
        "git": {
            "available": True,
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
        },
        "environment": {
            "python": "3.12.0",
            "torch": str(torch.__version__),
            "device": "cpu",
        },
        "dataset": {
            "version": "sha256:dataset",
            "fingerprint": "sha256:fingerprint",
            "verified": True,
            "sources": [
                {
                    "role": "train",
                    "path": "data/train.bin",
                    "tokens": 1234,
                    "sha256": "deadbeef",
                    "dataset_version": "sha256:dataset",
                }
            ],
        },
    }
    metrics = {
        "best_val": 1.5,
        "tokens_seen": 4096,
        "last_train": {"lm_loss": 1.7},
        "last_validation": {"loss": 1.5, "perplexity": 4.48169},
    }
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        None,
        7,
        config,
        qat_active,
        best_val=1.5,
        run_metadata=run_metadata,
        metrics=metrics,
    )
    return {"run_metadata": run_metadata, "metrics": metrics}


def _evaluation() -> dict[str, object]:
    return {
        "created_utc": "2026-01-02T00:00:00+00:00",
        "evaluation": {"git": {"commit": "def456"}},
        "settings": {"seq_len": 8, "batch_size": 2, "device": "cpu"},
        "results": {
            "val": {
                "tokens": 512,
                "loss": 1.25,
                "perplexity": 3.49034,
                "dataset_version": "sha256:dataset",
            }
        },
    }


def test_model_card_from_checkpoint_and_evaluation(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    _save_checkpoint(checkpoint_path, tiny_experiment_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    card = render_model_card(
        tiny_experiment_config,
        name="Tiny AURORA",
        description="Tiny test model.",
        license_name="Apache-2.0",
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        evaluation=_evaluation(),
        limitations=["Test-only checkpoint."],
    )

    total, active = parameter_counts(build_model(tiny_experiment_config.model))
    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert "# Model card: Tiny AURORA" in card
    assert "Tiny test model." in card
    assert "Apache-2.0" in card
    assert f"{total:,}" in card
    assert f"{active:,}" in card
    assert sha256 in card
    assert "model-card-run" in card
    assert "abc123" in card
    assert "sha256:dataset" in card
    assert "sha256:fingerprint" in card
    assert "data/train.bin" in card
    assert "3.49034" in card
    assert "def456" in card
    assert "Test-only checkpoint." in card
    assert "No trained weights are associated with this card." not in card
    assert "No evaluation results are associated with this card." not in card

    output = tmp_path / "reports" / "MODEL_CARD.md"
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(_evaluation()), encoding="utf-8")
    main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--evaluation",
            str(evaluation_path),
            "--output",
            str(output),
            "--name",
            "CLI model",
            "--license",
            "MIT",
        ]
    )
    written = output.read_text(encoding="utf-8")
    assert "# Model card: CLI model" in written
    assert "MIT" in written
    assert sha256 in written


def test_model_card_from_config_only(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(tiny_experiment_config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "MODEL_CARD.md"
    main(["--config", str(config_path), "--output", str(output), "--name", "Config-only"])

    card = output.read_text(encoding="utf-8")
    assert "# Model card: Config-only" in card
    assert "Configuration only (no trained checkpoint)" in card
    assert "No trained weights are associated with this card." in card
    assert "No evaluation results are associated with this card." in card
    assert "Not determined" in card


def test_model_card_qat_warning(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    checkpoint_path = tmp_path / "qat.pt"
    _save_checkpoint(checkpoint_path, tiny_experiment_config, qat_active=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    card = render_model_card(
        tiny_experiment_config,
        name="QAT",
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    assert "Checkpoint with QAT active" in card
    assert "QAT results require an FP baseline" in card


def test_model_card_rejects_invalid_evaluation(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    evaluation_path = tmp_path / "bad.json"
    evaluation_path.write_text(json.dumps({"created_utc": "now"}), encoding="utf-8")
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(tiny_experiment_config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="results"):
        main(
            [
                "--config",
                str(config_path),
                "--evaluation",
                str(evaluation_path),
                "--output",
                str(tmp_path / "card.md"),
            ]
        )
