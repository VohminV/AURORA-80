from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from aurora.architecture.factory import build_model, set_gradient_checkpointing
from aurora.config import ModelConfig


def _config(architecture: str, tiny_model_config: ModelConfig) -> ModelConfig:
    base = replace(
        tiny_model_config,
        architecture=architecture,
        memory_loss_coef=0.01,
        consistency_loss_coef=0.01,
    )
    if architecture == "transformer":
        return replace(base, attention_heads=4, ffn_hidden=32)
    return base


def _loss(model: torch.nn.Module, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    output = model(input_ids)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    loss = F.cross_entropy(output.logits.reshape(-1, vocab_size), targets.reshape(-1))
    return loss + output.aux_loss + 0.01 * output.memory_loss + 0.01 * output.consistency_loss


@pytest.mark.parametrize("architecture", ["aurora", "transformer"])
def test_gradient_checkpointing_matches_full_backward(
    architecture: str, tiny_model_config: ModelConfig
) -> None:
    config = _config(architecture, tiny_model_config)
    torch.manual_seed(0)
    reference = build_model(config)
    torch.manual_seed(0)
    checkpointed = build_model(config)
    checkpointed.load_state_dict(reference.state_dict())
    assert set_gradient_checkpointing(checkpointed, True) == config.n_layers

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    reference.train()
    checkpointed.train()

    reference_loss = _loss(reference, input_ids, config.vocab_size)
    reference_loss.backward()
    checkpointed_loss = _loss(checkpointed, input_ids, config.vocab_size)
    checkpointed_loss.backward()

    torch.testing.assert_close(reference_loss, checkpointed_loss)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        expected = reference_parameters[name].grad
        if expected is None:
            assert parameter.grad is None
            continue
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-5)


def test_eval_forward_ignores_checkpointing(tiny_model_config: ModelConfig) -> None:
    config = _config("aurora", tiny_model_config)
    model = build_model(config)
    set_gradient_checkpointing(model, True)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        output = model(input_ids)
    assert torch.isfinite(output.logits).all()
