from __future__ import annotations

from bisect import bisect_right
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from aurora.config import TrainingConfig
from aurora.data.mixture import build_token_mixture


def _write_tokens(path: Path, size: int) -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 64, size=size, dtype=np.uint16)
    tokens.tofile(path)


def test_mixture_sampler_matches_domain_weights(tmp_path: Path) -> None:
    paths = [tmp_path / f"domain_{index}.bin" for index in range(3)]
    for path in paths:
        _write_tokens(path, 8192)

    dataset, sampler, sources = build_token_mixture(paths, [0.6, 0.3, 0.1], seq_len=8, seed=7)
    assert len(sources) == 3
    assert sum(source.samples for source in sources) == len(dataset)
    assert len(sampler) == len(dataset)

    indices = torch.tensor(list(sampler))
    boundaries = [source.samples for source in sources]
    cumulative = list(np.cumsum(boundaries))
    counts = [0, 0, 0]
    for index in indices.tolist():
        counts[bisect_right(cumulative, index)] += 1

    shares = [count / len(dataset) for count in counts]
    for share, expected in zip(shares, [0.6, 0.3, 0.1], strict=True):
        assert abs(share - expected) < 0.05


def test_mixture_sampler_is_seeded(tmp_path: Path) -> None:
    paths = [tmp_path / f"domain_{index}.bin" for index in range(2)]
    for path in paths:
        _write_tokens(path, 4096)
    _, first, _ = build_token_mixture(paths, [0.5, 0.5], seq_len=8, seed=11)
    _, second, _ = build_token_mixture(paths, [0.5, 0.5], seq_len=8, seed=11)
    assert list(islice(first, 32)) == list(islice(second, 32))


def test_mixture_rejects_invalid_inputs(tmp_path: Path) -> None:
    path = tmp_path / "domain.bin"
    _write_tokens(path, 512)
    with pytest.raises(ValueError):
        build_token_mixture([], [], seq_len=8)
    with pytest.raises(ValueError):
        build_token_mixture([path], [], seq_len=8)
    with pytest.raises(ValueError):
        build_token_mixture([path], [0.0], seq_len=8)


def test_training_config_validates_mixture() -> None:
    config = TrainingConfig(train_paths=("a.bin", "b.bin"), train_weights=(0.5, 0.5))
    assert config.train_paths == ("a.bin", "b.bin")
    with pytest.raises(ValueError):
        TrainingConfig(train_paths=("a.bin", "b.bin"), train_weights=(1.0,))
    with pytest.raises(ValueError):
        TrainingConfig(train_path="train.bin", train_paths=("a.bin",), train_weights=(1.0,))
    with pytest.raises(ValueError):
        TrainingConfig(train_paths=("a.bin",), train_weights=(-1.0,))
    with pytest.raises(ValueError):
        TrainingConfig(train_weights=(1.0,))
