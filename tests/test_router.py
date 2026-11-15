from __future__ import annotations

import torch

from aurora.architecture.expert_router import MoE, TopKRouter


def test_topk_router_weights_and_losses() -> None:
    torch.manual_seed(0)
    router = TopKRouter(d_model=32, n_experts=4, top_k=2)
    x = torch.randn(10, 32, requires_grad=True)
    weights, indices, balance, z_loss = router(x)
    assert weights.shape == (10, 2)
    assert indices.shape == (10, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(10))
    assert balance.ndim == 0
    assert z_loss.ndim == 0
    assert float(balance.detach()) > 0.0
    (balance + z_loss).backward()
    assert x.grad is not None


def test_moe_output_and_parameter_stats() -> None:
    torch.manual_seed(0)
    moe = MoE(
        d_model=32,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
    )
    x = torch.randn(2, 5, 32, requires_grad=True)
    output, aux_loss = moe(x)
    assert output.shape == x.shape
    assert aux_loss.ndim == 0
    (output.square().mean() + aux_loss).backward()
    assert x.grad is not None
    total, active = moe.expert_parameter_stats()
    assert total > 0
    assert 0 < active < total


def test_moe_routes_tokens_without_overflow() -> None:
    torch.manual_seed(0)
    moe = MoE(d_model=16, n_experts=8, top_k=3, expert_hidden=16)
    x = torch.randn(64, 16)
    output, _ = moe(x)
    assert torch.isfinite(output).all()
