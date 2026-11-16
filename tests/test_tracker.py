from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

from aurora.config import ExperimentConfig
from aurora.training.tracker import WandbTracker


def _wandb_config(
    config: ExperimentConfig,
    *,
    mode: str = "offline",
    enabled: bool = True,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=config.model,
        training=replace(
            config.training,
            wandb=enabled,
            wandb_project="aurora-test",
            wandb_run_name="test-run",
            wandb_mode=mode,
        ),
    )


def test_wandb_tracker_logs_and_finishes(
    monkeypatch: Any, tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    calls: dict[str, Any] = {"logs": []}
    module = types.ModuleType("wandb")

    def init(**kwargs: Any) -> object:
        calls["init"] = kwargs
        return object()

    def log(metrics: dict[str, float], *, step: int) -> None:
        calls["logs"].append((metrics, step))

    def finish() -> None:
        calls["finished"] = True

    module.init = init  # type: ignore[attr-defined]
    module.log = log  # type: ignore[attr-defined]
    module.finish = finish  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)

    config = _wandb_config(tiny_experiment_config)
    tracker = WandbTracker.start(config, tmp_path, {"run_id": "run-123"})
    assert tracker is not None
    assert calls["init"]["project"] == "aurora-test"
    assert calls["init"]["name"] == "test-run"
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["id"] == "run-123"
    assert calls["init"]["resume"] == "allow"
    assert calls["init"]["dir"] == str(tmp_path)

    tracker.log({"train/lm_loss": 1.0}, step=3)
    tracker.finish()
    assert calls["logs"] == [({"train/lm_loss": 1.0}, 3)]
    assert calls["finished"] is True


def test_wandb_tracker_disabled(
    monkeypatch: Any, tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    module = types.ModuleType("wandb")
    module.init = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)

    config = _wandb_config(tiny_experiment_config, enabled=False)
    assert WandbTracker.start(config, tmp_path, {"run_id": "run-123"}) is None


def test_wandb_tracker_missing_package(
    monkeypatch: Any, tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)

    config = _wandb_config(tiny_experiment_config)
    assert WandbTracker.start(config, tmp_path, {"run_id": "run-123"}) is None


def test_wandb_tracker_init_failure(
    monkeypatch: Any, tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    module = types.ModuleType("wandb")

    def init(**kwargs: Any) -> object:
        raise RuntimeError("offline failure")

    module.init = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", module)

    config = _wandb_config(tiny_experiment_config)
    assert WandbTracker.start(config, tmp_path, {"run_id": "run-123"}) is None
