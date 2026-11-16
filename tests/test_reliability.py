from __future__ import annotations

import io
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aurora.config import ExperimentConfig, TrainingConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.training.alerts import ALERT_WEBHOOK_ENV, AlertNotifier
from aurora.training.checkpoint import load_checkpoint
from aurora.training.trainer import Trainer


class _FakeNotifier:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def send(self, event: str, payload: dict[str, Any]) -> bool:
        self.events.append((event, payload))
        return True


def _write_tokens(path: Path, tokens: int = 512) -> None:
    rng = np.random.default_rng(7)
    rng.integers(0, 64, size=tokens, dtype=np.uint16).tofile(path)


def _trainer(
    tmp_path: Path,
    config: ExperimentConfig,
    **training_overrides: Any,
) -> Trainer:
    training = replace(
        config.training,
        output_dir=str(tmp_path / "run"),
        eval_interval=0,
        save_interval=0,
        device="cpu",
        dtype_amp="fp32",
        **training_overrides,
    )
    resolved = ExperimentConfig(model=config.model, training=training)
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path)
    dataset = MemmapTokenDataset(train_path, training.seq_len, dtype=training.dtype)
    return Trainer(resolved, dataset)


def test_backups_rotate_and_load(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    trainer = _trainer(
        tmp_path,
        tiny_experiment_config,
        backup_interval=2,
        backup_count=2,
    )
    for step in (2, 4, 6):
        trainer.step = step
        trainer._save("latest.pt")

    backups = tmp_path / "run" / "backups"
    names = sorted(path.name for path in backups.glob("latest_step_*.pt"))
    assert names == ["latest_step_000000004.pt", "latest_step_000000006.pt"]
    checkpoint = load_checkpoint(backups / names[-1])
    assert checkpoint["step"] == 6


def test_backups_disabled_by_default(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    trainer = _trainer(tmp_path, tiny_experiment_config)
    trainer.step = 5
    trainer._save("latest.pt")

    assert not (tmp_path / "run" / "backups").exists()


def test_backups_respect_interval_and_custom_dir(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    custom_dir = tmp_path / "offsite"
    trainer = _trainer(
        tmp_path,
        tiny_experiment_config,
        backup_interval=3,
        backup_count=1,
        backup_dir=str(custom_dir),
    )
    trainer.step = 2
    trainer._save("latest.pt")
    assert list(custom_dir.glob("*.pt")) == []

    trainer.step = 3
    trainer._save("latest.pt")
    assert [path.name for path in custom_dir.glob("latest_step_*.pt")] == [
        "latest_step_000000003.pt"
    ]

    trainer.step = 6
    trainer._save("best.pt")
    names = sorted(path.name for path in custom_dir.glob("*_step_*.pt"))
    assert names == ["best_step_000000006.pt", "latest_step_000000003.pt"]


def test_alert_notifier_posts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> io.BytesIO:
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return io.BytesIO(b"ok")

    monkeypatch.setattr("aurora.training.alerts.urllib.request.urlopen", fake_urlopen)
    notifier = AlertNotifier("https://example.com/hook", logging.getLogger("test"), timeout=2.0)

    assert notifier.send("validation_degradation", {"step": 5, "best_val": 1.5}) is True
    assert captured["url"] == "https://example.com/hook"
    assert captured["timeout"] == 2.0
    assert captured["payload"]["event"] == "validation_degradation"
    assert captured["payload"]["step"] == 5
    assert captured["payload"]["best_val"] == 1.5
    assert captured["payload"]["created_utc"]


def test_alert_notifier_failures_are_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_urlopen(request: Any, timeout: float) -> io.BytesIO:
        raise OSError("connection refused")

    monkeypatch.setattr("aurora.training.alerts.urllib.request.urlopen", failing_urlopen)
    notifier = AlertNotifier("https://example.com/hook", logging.getLogger("test"))
    assert notifier.send("early_stopping", {}) is False
    assert AlertNotifier("", logging.getLogger("test")).send("early_stopping", {}) is False
    with pytest.raises(ValueError, match="http"):
        AlertNotifier("ftp://example.com/hook", logging.getLogger("test"))


def test_trainer_creates_notifier_from_environment(
    tmp_path: Path,
    tiny_experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ALERT_WEBHOOK_ENV, "https://example.invalid/hook")
    trainer = _trainer(tmp_path, tiny_experiment_config)
    assert trainer.notifier is not None
    assert trainer.notifier.enabled


def test_degradation_alerts_once_per_streak(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    trainer = _trainer(tmp_path, tiny_experiment_config, degradation_patience=1)
    notifier = _FakeNotifier()
    trainer.notifier = notifier  # type: ignore[assignment]
    trainer.best_val = 0.5
    trainer.step = 1
    trainer.evaluate = lambda: 1.0  # type: ignore[method-assign]

    assert trainer._evaluate_and_log() is False
    assert [event for event, _ in notifier.events] == ["validation_degradation"]
    payload = notifier.events[0][1]
    assert payload["step"] == 1
    assert payload["best_val"] == 0.5
    assert payload["evals_without_improvement"] == 1
    assert payload["run_id"] == trainer.run_metadata["run_id"]

    trainer.step = 2
    trainer._evaluate_and_log()
    assert len(notifier.events) == 1

    trainer.evaluate = lambda: 0.4  # type: ignore[method-assign]
    trainer.step = 3
    trainer._evaluate_and_log()
    assert len(notifier.events) == 1
    assert trainer._degradation_alerted is False

    trainer.evaluate = lambda: 0.6  # type: ignore[method-assign]
    trainer.step = 4
    trainer._evaluate_and_log()
    assert [event for event, _ in notifier.events] == [
        "validation_degradation",
        "validation_degradation",
    ]


def test_early_stopping_alert(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    trainer = _trainer(tmp_path, tiny_experiment_config, early_stopping_patience=1)
    notifier = _FakeNotifier()
    trainer.notifier = notifier  # type: ignore[assignment]
    trainer.best_val = 0.5
    trainer.step = 1
    trainer.evaluate = lambda: 1.0  # type: ignore[method-assign]

    assert trainer._evaluate_and_log() is True
    assert [event for event, _ in notifier.events] == ["early_stopping"]
    assert notifier.events[0][1]["patience"] == 1


def test_reliability_config_validation() -> None:
    with pytest.raises(ValueError, match="backup_interval"):
        TrainingConfig(backup_interval=-1)
    with pytest.raises(ValueError, match="backup_count"):
        TrainingConfig(backup_count=0)
    with pytest.raises(ValueError, match="degradation_patience"):
        TrainingConfig(degradation_patience=-1)
