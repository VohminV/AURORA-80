from __future__ import annotations

import torch

from aurora.architecture.recurrent_core import RecurrentCore


def test_recurrent_core_shapes_and_state() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(d_model=32, d_state=16, n_levels=2)
    x = torch.randn(2, 8, 32)
    output, state = core(x)
    assert output.shape == (2, 8, 32)
    assert state.shape == (2, 2, 16)
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


def test_recurrent_core_is_causal_and_stateful() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(d_model=32, d_state=16, n_levels=2)
    x = torch.randn(1, 10, 32)
    full_output, _ = core(x)
    prefix_output, state = core(x[:, :4])
    continuation_output, _ = core(x[:, 4:], state)
    torch.testing.assert_close(prefix_output, full_output[:, :4])
    torch.testing.assert_close(continuation_output, full_output[:, 4:])


def test_recurrent_core_document_reset_matches_fresh_segment() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(d_model=32, d_state=16, n_levels=2)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full_output, _ = core(x)
        segment_output, segment_state = core(x[:, 5:])
        reset_mask = torch.zeros(1, 10, dtype=torch.bool)
        reset_mask[:, 5] = True
        reset_output, reset_state = core(x, reset_mask=reset_mask)
    torch.testing.assert_close(reset_output[:, :5], full_output[:, :5])
    torch.testing.assert_close(reset_output[:, 5:], segment_output)
    torch.testing.assert_close(reset_state, segment_state)


def test_recurrent_core_reset_is_deterministic() -> None:
    torch.manual_seed(0)
    core = RecurrentCore(d_model=32, d_state=16, n_levels=2)
    x = torch.randn(2, 6, 32)
    first, _ = core(x)
    second, _ = core(x)
    torch.testing.assert_close(first, second)


def test_recurrent_core_backward() -> None:
    core = RecurrentCore(d_model=32, d_state=16, n_levels=2)
    x = torch.randn(2, 5, 32, requires_grad=True)
    output, _ = core(x)
    output.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
