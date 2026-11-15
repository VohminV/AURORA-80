from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig, ModelConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.generation import generate, generate_batch, stream_generate
from aurora.inference import load_model_from_checkpoint
from aurora.quantization import TernaryLinear
from aurora.training.trainer import Trainer


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 64, size=4096, dtype=np.uint16)
    tokens.tofile(path)


def _train_tiny_model(
    tmp_path: Path, config: ExperimentConfig, *, qat: bool
) -> tuple[Path, ExperimentConfig]:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 5)
    training = replace(
        config.training,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        output_dir=str(tmp_path / "inference_run"),
        qat_enabled=qat,
        qat_start_step=1,
        qat_min_features=16,
    )
    resolved = ExperimentConfig(model=config.model, training=training)
    Trainer(resolved, MemmapTokenDataset(train_path, training.seq_len)).fit()
    return Path(training.output_dir) / "latest.pt", resolved


def test_checkpoint_inference_roundtrip(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint_path, config = _train_tiny_model(tmp_path, tiny_experiment_config, qat=False)
    model, loaded_config = load_model_from_checkpoint(checkpoint_path, device="cpu")

    assert loaded_config.model == config.model
    assert not any(isinstance(module, TernaryLinear) for module in model.modules())

    prompt = torch.tensor([[1, 2, 3, 4]])
    generated, state = generate(model, prompt, max_new_tokens=6, temperature=0.0)
    assert len(generated) == 6
    assert all(0 <= token < config.model.vocab_size for token in generated)
    assert len(state.blocks) == config.model.n_layers


def test_qat_checkpoint_inference_roundtrip(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    checkpoint_path, config = _train_tiny_model(tmp_path, tiny_experiment_config, qat=True)
    model, loaded_config = load_model_from_checkpoint(checkpoint_path, device="cpu")

    assert loaded_config.model == config.model
    assert any(isinstance(module, TernaryLinear) for module in model.modules())

    prompt = torch.tensor([[1, 2, 3, 4]])
    generated, _ = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    assert len(generated) == 4
    assert all(0 <= token < config.model.vocab_size for token in generated)


def test_batched_generation_matches_single(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batched, _ = generate_batch(model, prompts, max_new_tokens=5, temperature=0.0)
    singles = [
        generate(model, prompt.unsqueeze(0), max_new_tokens=5, temperature=0.0)[0]
        for prompt in prompts
    ]
    assert batched == singles


def test_stream_generation_matches_batched(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]])
    steps = list(stream_generate(model, prompts, max_new_tokens=4, temperature=0.0))
    assert all(len(step) == 2 for step in steps)
    rows = [[token for step in steps if (token := step[index]) is not None] for index in range(2)]
    batched, _ = generate_batch(model, prompts, max_new_tokens=4, temperature=0.0)
    assert rows == batched


def test_stream_generation_stops_on_eos(tiny_model_config: ModelConfig) -> None:
    model = build_model(tiny_model_config)
    prompt = torch.tensor([[1, 2, 3, 4]])
    first, _ = generate(model, prompt, max_new_tokens=1, temperature=0.0)
    steps = list(
        stream_generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=first[0])
    )
    assert steps == [[first[0]]]
