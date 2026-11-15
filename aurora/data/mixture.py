from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from aurora.data.dataset import MemmapTokenDataset


@dataclass(frozen=True)
class MixtureSource:
    path: str
    weight: float
    samples: int


def build_token_mixture(
    paths: Sequence[str | Path],
    weights: Sequence[float],
    seq_len: int,
    dtype: str = "uint16",
    seed: int = 42,
) -> tuple[Dataset, WeightedRandomSampler, list[MixtureSource]]:
    """Combine token files and sample them according to domain weights.

    The sampler draws ``len(ConcatDataset)`` indices with replacement, so the
    expected share of a domain is its normalized weight over a full epoch.
    """
    if not paths:
        raise ValueError("At least one mixture path is required")
    if len(paths) != len(weights):
        raise ValueError("paths and weights must have the same length")
    if any(weight <= 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be positive and finite")

    datasets = [MemmapTokenDataset(path, seq_len, dtype=dtype) for path in paths]
    total_samples = sum(len(dataset) for dataset in datasets)
    if total_samples == 0:
        raise ValueError("Mixture datasets are empty")

    sample_weights = torch.empty(total_samples, dtype=torch.double)
    sources: list[MixtureSource] = []
    offset = 0
    for path, weight, dataset in zip(paths, weights, datasets, strict=True):
        samples = len(dataset)
        sample_weights[offset : offset + samples] = weight / samples
        sources.append(MixtureSource(path=str(path), weight=weight, samples=samples))
        offset += samples

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        cast(Any, sample_weights),
        num_samples=total_samples,
        replacement=True,
        generator=generator,
    )
    return ConcatDataset(datasets), sampler, sources
