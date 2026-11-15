from __future__ import annotations

import math

import torch

from aurora.architecture.associative_memory import AssociativeMemory


def test_memory_shapes_and_state_change() -> None:
    torch.manual_seed(0)
    memory = AssociativeMemory(d_model=32, n_heads=2)
    x = torch.randn(2, 8, 32)
    output, state = memory(x)
    assert output.shape == (2, 8, 32)
    assert state.shape == (2, 2, 16, 16)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()
    assert state.abs().sum() > 0


def test_memory_is_causal_and_stateful() -> None:
    torch.manual_seed(0)
    memory = AssociativeMemory(d_model=32, n_heads=2)
    x = torch.randn(1, 9, 32)
    full_output, _ = memory(x)
    prefix_output, state = memory(x[:, :4])
    continuation_output, _ = memory(x[:, 4:], state)
    torch.testing.assert_close(prefix_output, full_output[:, :4])
    torch.testing.assert_close(continuation_output, full_output[:, 4:])


def test_memory_delta_rule_writes_and_reads() -> None:
    memory = AssociativeMemory(d_model=4, n_heads=1, epsilon=1e-8)
    with torch.no_grad():
        identity = torch.eye(4)
        memory.q_proj.weight.copy_(identity)
        memory.k_proj.weight.copy_(identity)
        memory.v_proj.weight.copy_(identity)
        memory.out_proj.weight.copy_(identity)
        memory.decay_proj.weight.zero_()
        memory.decay_proj.bias.fill_(20.0)
        memory.write_proj.weight.zero_()
        memory.write_proj.bias.fill_(20.0)

    token = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    with torch.no_grad():
        _, state = memory(token)
    assert state.shape == (1, 1, 4, 4)
    assert math.isclose(float(state[0, 0, 0, 0]), 1.0, rel_tol=1e-4, abs_tol=1e-4)

    query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    query = torch.nn.functional.normalize(query, dim=-1)
    read = (query.transpose(1, 2).squeeze(1).unsqueeze(-1) * state).sum(dim=-2)
    torch.testing.assert_close(read[0, 0], token[0, 0], rtol=1e-4, atol=1e-4)


def test_memory_backward() -> None:
    memory = AssociativeMemory(d_model=32, n_heads=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    output, _ = memory(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
