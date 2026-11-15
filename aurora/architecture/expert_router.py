from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, jitter: float = 0.0) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.jitter = jitter
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        if self.training and self.jitter > 0.0:
            logits = logits + torch.randn_like(logits) * self.jitter
        probabilities = F.softmax(logits, dim=-1)
        topk_probabilities, topk_indices = torch.topk(probabilities, self.top_k, dim=-1)
        topk_weights = topk_probabilities / topk_probabilities.sum(dim=-1, keepdim=True).clamp_min(
            1e-9
        )

        assignments = F.one_hot(topk_indices, self.n_experts).float().sum(dim=1)
        assignment_fraction = assignments.mean(dim=0) / self.top_k
        mean_probability = probabilities.mean(dim=0)
        balance_loss = self.n_experts * (assignment_fraction * mean_probability).sum()
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        return topk_weights, topk_indices, balance_loss, z_loss


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(hidden))


class MoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        router_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router = TopKRouter(d_model, n_experts, top_k, router_jitter)
        self.experts = nn.ModuleList(
            [Expert(d_model, expert_hidden, dropout) for _ in range(n_experts)]
        )

    def expert_parameter_stats(self) -> tuple[int, int]:
        total = sum(
            parameter.numel() for expert in self.experts for parameter in expert.parameters()
        )
        parameters_per_expert = total / self.n_experts
        return total, int(round(parameters_per_expert * self.top_k))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = x.shape
        flat = x.reshape(-1, input_shape[-1])
        n_tokens = flat.shape[0]

        topk_weights, topk_indices, balance_loss, z_loss = self.router(flat)

        assignments = topk_indices.reshape(-1)
        weights = topk_weights.reshape(-1)
        token_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)
        order = torch.argsort(assignments, stable=True)
        sorted_experts = assignments[order]
        sorted_tokens = token_indices[order]
        sorted_weights = weights[order]

        counts = torch.bincount(sorted_experts, minlength=self.n_experts)
        group_sizes = counts.tolist()
        token_groups = torch.split(sorted_tokens, group_sizes)
        weight_groups = torch.split(sorted_weights, group_sizes)

        output = torch.zeros_like(flat)
        for expert, tokens, weights in zip(self.experts, token_groups, weight_groups, strict=True):
            if tokens.numel() == 0:
                continue
            expert_output = expert(flat[tokens]) * weights.unsqueeze(-1)
            output.index_add_(0, tokens, expert_output)

        aux_loss = self.aux_loss_coef * balance_loss + self.router_z_loss_coef * z_loss
        return output.view(input_shape), aux_loss
