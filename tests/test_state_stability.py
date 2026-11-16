from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from aurora.architecture.factory import build_model
from aurora.benchmarks.state_stability import _load_tokens, measure_state_stability
from aurora.config import ExperimentConfig


def test_measure_state_stability(
    tiny_experiment_config: ExperimentConfig,
) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_experiment_config.model)
    tokens = torch.randint(0, tiny_experiment_config.model.vocab_size, (64,))
    results = measure_state_stability(
        model,
        tiny_experiment_config,
        [16, 32],
        chunk_size=8,
        tokens=tokens,
        device="cpu",
    )
    assert [result["length"] for result in results] == [16, 32]
    for result in results:
        assert result["chunks"] == result["length"] // 8
        assert result["finite"] is True
        assert math.isfinite(result["loss"])
        assert result["perplexity"] > 0
        assert result["state_norm_final"] > 0
        assert result["state_norm_max"] >= result["state_norm_final"]
        assert result["state_max_abs"] > 0
        assert result["state_growth"] is not None
        assert result["tokens_per_second"] > 0


def test_state_reset_interval(
    tiny_experiment_config: ExperimentConfig,
) -> None:
    torch.manual_seed(1)
    model = build_model(tiny_experiment_config.model)
    tokens = torch.randint(0, tiny_experiment_config.model.vocab_size, (32,))
    results = measure_state_stability(
        model,
        tiny_experiment_config,
        [32],
        chunk_size=8,
        reset_interval=8,
        tokens=tokens,
        device="cpu",
    )
    assert results[0]["reset_interval"] == 8
    assert results[0]["chunks"] == 4
    assert results[0]["finite"] is True

    with pytest.raises(ValueError, match="chunk_size"):
        measure_state_stability(model, tiny_experiment_config, [8], chunk_size=0)
    with pytest.raises(ValueError, match="reset_interval"):
        measure_state_stability(model, tiny_experiment_config, [8], reset_interval=-1)
    with pytest.raises(ValueError, match="lengths"):
        measure_state_stability(model, tiny_experiment_config, [])
    with pytest.raises(ValueError, match="need at least"):
        measure_state_stability(
            model,
            tiny_experiment_config,
            [64],
            chunk_size=8,
            tokens=tokens,
            device="cpu",
        )


def test_load_tokens(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.asarray([1, 2, 3, 4], dtype=np.uint16).tofile(path)
    tokens = _load_tokens(path, "uint16", 4)
    assert tokens.tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="need at least"):
        _load_tokens(path, "uint16", 5)
