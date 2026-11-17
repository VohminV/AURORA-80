from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from aurora.architecture.model import AuroraModel
from aurora.config import ModelConfig
from aurora.generation import generate
from aurora.utils import parameter_counts


def test_model_forward_and_backward(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    input_ids = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    output = model(input_ids)
    assert output.logits.shape == (2, 8, tiny_model_config.vocab_size)
    assert output.aux_loss.ndim == 0
    assert len(output.state.blocks) == tiny_model_config.n_layers

    targets = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    lm_loss = F.cross_entropy(
        output.logits.reshape(-1, tiny_model_config.vocab_size), targets.reshape(-1)
    )
    (lm_loss + output.aux_loss).backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_model_auxiliary_memory_losses(tiny_model_config: ModelConfig) -> None:
    config = replace(
        tiny_model_config,
        memory_loss_coef=0.01,
        consistency_loss_coef=0.01,
    )
    torch.manual_seed(0)
    model = AuroraModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    output = model(input_ids)
    assert output.memory_loss.ndim == 0
    assert output.consistency_loss.ndim == 0
    assert float(output.memory_loss.detach()) > 0.0
    assert float(output.consistency_loss.detach()) > 0.0

    lm_loss = F.cross_entropy(
        output.logits.reshape(-1, config.vocab_size),
        input_ids.reshape(-1),
    )
    total = (
        lm_loss
        + output.aux_loss
        + config.memory_loss_coef * output.memory_loss
        + config.consistency_loss_coef * output.consistency_loss
    )
    total.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_model_is_causal(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    model.eval()
    first = torch.randint(0, tiny_model_config.vocab_size, (1, 10))
    second = first.clone()
    second[:, 6:] = (second[:, 6:] + 1) % tiny_model_config.vocab_size
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6])
    assert not torch.allclose(first_logits[:, 6:], second_logits[:, 6:])


def test_model_state_continuation(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    model.eval()
    input_ids = torch.randint(0, tiny_model_config.vocab_size, (1, 12))
    with torch.no_grad():
        full = model(input_ids)
        prefix = model(input_ids[:, :5])
        continuation = model(input_ids[:, 5:], state=prefix.state)
    torch.testing.assert_close(full.logits[:, :5], prefix.logits)
    torch.testing.assert_close(full.logits[:, 5:], continuation.logits)


def test_model_document_reset_matches_fresh_segment(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    model.eval()
    input_ids = torch.tensor([[5, 6, 7, 8, 2, 9, 10, 11, 12, 13]])
    reset_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    reset_mask[:, 5] = True
    with torch.no_grad():
        full = model(input_ids, reset_mask=reset_mask)
        segment = model(input_ids[:, 5:])
        without_reset = model(input_ids)
    torch.testing.assert_close(full.logits[:, :5], without_reset.logits[:, :5])
    torch.testing.assert_close(full.logits[:, 5:], segment.logits)


def test_model_parameter_counts(tiny_model_config: ModelConfig) -> None:
    model = AuroraModel(tiny_model_config)
    total, active = parameter_counts(model)
    assert total > 0
    assert 0 < active <= total


def test_generation_length_and_determinism(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    prompt = torch.randint(0, tiny_model_config.vocab_size, (1, 5))
    generated_first, _ = generate(model, prompt, max_new_tokens=4, temperature=0.0, eos_id=None)
    generated_second, _ = generate(model, prompt, max_new_tokens=4, temperature=0.0, eos_id=None)
    assert len(generated_first) == 4
    assert generated_first == generated_second


def test_generation_stops_at_eos(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    prompt = torch.randint(0, tiny_model_config.vocab_size, (1, 5))
    first, _ = generate(model, prompt, max_new_tokens=1, temperature=0.0)
    stopped, _ = generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=first[0])
    assert stopped == first


def test_generation_rejects_bad_input(tiny_model_config: ModelConfig) -> None:
    model = AuroraModel(tiny_model_config)
    with pytest.raises(ValueError):
        generate(model, torch.zeros(4, dtype=torch.long), max_new_tokens=2)
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 0, dtype=torch.long), max_new_tokens=2)
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 2, dtype=torch.long), max_new_tokens=-1)


def test_model_rejects_bad_state(tiny_model_config: ModelConfig) -> None:
    model = AuroraModel(tiny_model_config)
    input_ids = torch.randint(0, tiny_model_config.vocab_size, (1, 4))
    other = AuroraModel(tiny_model_config)
    with pytest.raises(ValueError):
        model(input_ids, state=other.init_state(2, input_ids.device))
