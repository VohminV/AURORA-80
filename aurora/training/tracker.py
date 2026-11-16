from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aurora.config import ExperimentConfig

logger = logging.getLogger("aurora.tracker")


class WandbTracker:
    """Small optional wrapper around Weights & Biases scalar logging."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self._active = True

    @classmethod
    def start(
        cls,
        config: ExperimentConfig,
        output_dir: str | Path,
        run_metadata: Mapping[str, Any],
    ) -> WandbTracker | None:
        training = config.training
        if not training.wandb or training.wandb_mode == "disabled":
            return None
        try:
            module = importlib.import_module("wandb")
        except ImportError:
            logger.warning("wandb is not installed; disabling W&B logging")
            return None

        init_kwargs: dict[str, Any] = {
            "project": training.wandb_project,
            "config": config.to_dict(),
            "dir": str(output_dir),
            "mode": training.wandb_mode,
        }
        if training.wandb_run_name:
            init_kwargs["name"] = training.wandb_run_name
        run_id = run_metadata.get("run_id")
        if isinstance(run_id, str) and run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        try:
            module.init(**init_kwargs)
        except Exception as error:
            logger.warning("Failed to initialize W&B logging: %s", error)
            return None
        logger.info(
            "W&B logging enabled | project %s | mode %s",
            training.wandb_project,
            training.wandb_mode,
        )
        return cls(module)

    def log(self, metrics: Mapping[str, float | int], *, step: int) -> None:
        if not self._active:
            return
        try:
            self._module.log(dict(metrics), step=step)
        except Exception as error:
            logger.warning("W&B logging failed; disabling further W&B calls: %s", error)
            self._active = False

    def finish(self) -> None:
        if not self._active:
            return
        try:
            self._module.finish()
        except Exception as error:
            logger.warning("Failed to finish W&B run: %s", error)
        finally:
            self._active = False
