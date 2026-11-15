from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aurora.architecture.factory import build_model
from aurora.architecture.model import AuroraModel
from aurora.architecture.transformer import TransformerModel, TransformerState
from aurora.config import ModelConfig
from aurora.generation import generate


@pytest.fixture
def tiny_transformer_config(tiny_model_config: ModelConfig) -> ModelConfig:
    return replace(
        tiny_model_config,
        architecture="transformer",
        attention_heads=4,
        ffn_hidden=32,
    )


def test_factory_builds_both_architectures(
    tiny_model_config: ModelConfig, tiny_transformer_config: ModelConfig
) -> None:
    assert isinstance(build_model(tiny_model_config), AuroraModel)
    assert isinstance(build_model(tiny_transformer_config), TransformerModel)


def test_transformer_forward_and_backward(tiny_transformer_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_transformer_config)
    assert isinstance(model, TransformerModel)
    input_ids = torch.randint(0, tiny_transformer_config.vocab_size, (2, 8))
    output = model(input_ids)
    assert output.logits.shape == (2, 8, tiny_transformer_config.vocab_size)
    assert float(output.aux_loss) == 0.0
    assert float(output.memory_loss) == 0.0
    assert float(output.consistency_loss) == 0.0
    assert len(output.state.layers) == tiny_transformer_config.n_layers
    assert output.state.layers[0].keys.shape[-2] == 8

    targets = torch.randint(0, tiny_transformer_config.vocab_size, (2, 8))
    loss = F.cross_entropy(
        output.logits.reshape(-1, tiny_transformer_config.vocab_size),
        targets.reshape(-1),
    )
    loss.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_transformer_is_causal(tiny_transformer_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_transformer_config)
    model.eval()
    first = torch.randint(0, tiny_transformer_config.vocab_size, (1, 10))
    second = first.clone()
    second[:, 6:] = (second[:, 6:] + 1) % tiny_transformer_config.vocab_size
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6])
    assert not torch.allclose(first_logits[:, 6:], second_logits[:, 6:])


def test_transformer_state_continuation(tiny_transformer_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_transformer_config)
    model.eval()
    input_ids = torch.randint(0, tiny_transformer_config.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(input_ids)
        prefix = model(input_ids[:, :5])
        continuation = model(input_ids[:, 5:], state=prefix.state)
        single_step = model(input_ids[:, :5])
        pieces = [single_step.logits]
        for index in range(5, 12):
            single_step = model(input_ids[:, index : index + 1], state=single_step.state)
            pieces.append(single_step.logits)
    torch.testing.assert_close(full.logits[:, :5], prefix.logits)
    torch.testing.assert_close(full.logits[:, 5:], continuation.logits)
    torch.testing.assert_close(full.logits, torch.cat(pieces, dim=1))


def test_transformer_generation_with_cache(tiny_transformer_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_transformer_config)
    prompt = torch.randint(0, tiny_transformer_config.vocab_size, (1, 5))
    first, state = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    second, _ = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    assert len(first) == 4
    assert first == second
    assert len(state.layers) == tiny_transformer_config.n_layers
    assert state.layers[0].keys.shape[-2] == 5 + 4


def test_transformer_rejects_bad_state(tiny_transformer_config: ModelConfig) -> None:
    model = build_model(tiny_transformer_config)
    input_ids = torch.randint(0, tiny_transformer_config.vocab_size, (1, 4))
    with pytest.raises(ValueError):
        model(input_ids, state=TransformerState(layers=[]))


def test_transformer_parameter_count_is_in_target_range() -> None:
    from aurora.config import load_config
    from aurora.utils import parameter_counts

    config = load_config("configs/transformer_100m.yaml")
    with torch.device("meta"):
        model = build_model(config.model)
    total, active = parameter_counts(model)
    assert active == total
    assert 95_000_000 <= total <= 115_000_000


def test_transformer_checkpoint_roundtrip(
    tmp_path, tiny_transformer_config: ModelConfig, tiny_training_config
) -> None:
    import numpy as np

    from aurora.config import ExperimentConfig
    from aurora.data.dataset import MemmapTokenDataset
    from aurora.inference import load_model_from_checkpoint
    from aurora.training.trainer import Trainer

    train_path = tmp_path / "train.bin"
    np.random.default_rng(3).integers(0, 64, size=2048, dtype=np.uint16).tofile(train_path)
    training = replace(
        tiny_training_config,
        train_path=str(train_path),
        val_path="",
        max_steps=2,
        eval_interval=0,
        save_interval=2,
        output_dir=str(tmp_path / "transformer_run"),
    )
    config = ExperimentConfig(model=tiny_transformer_config, training=training)
    Trainer(config, MemmapTokenDataset(train_path, training.seq_len)).fit()

    model, loaded = load_model_from_checkpoint(
        Path(training.output_dir) / "latest.pt", device="cpu"
    )
    assert isinstance(model, TransformerModel)
    assert loaded.model.architecture == "transformer"
    prompt = torch.tensor([[1, 2, 3, 4]])
    generated, _ = generate(model, prompt, max_new_tokens=4, temperature=0.0)
    assert len(generated) == 4
