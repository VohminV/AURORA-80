from __future__ import annotations

import torch
from torch import nn

from aurora.architecture.layers import RMSNorm
from aurora.architecture.state import apply_reset, prepare_reset_mask


class RecurrentLevel(nn.Module):
    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.z_proj = nn.Linear(d_model, d_state)
        self.z_rec = nn.Linear(d_state, d_state, bias=False)
        self.g_proj = nn.Linear(d_model, d_state)
        self.g_rec = nn.Linear(d_state, d_state, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        candidate = torch.tanh(self.z_proj(x).float() + self.z_rec(state).float())
        gate = torch.sigmoid(self.g_proj(x).float() + self.g_rec(state).float())
        return gate * state + (1.0 - gate) * candidate


class HierarchicalInteraction(nn.Module):
    def __init__(self, n_levels: int, d_state: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.d_state = d_state
        self.norm = RMSNorm(n_levels * d_state, norm_eps)
        self.proj = nn.ModuleList(
            [nn.Linear(n_levels * d_state, d_state, bias=False) for _ in range(n_levels)]
        )
        self.scale = nn.Parameter(torch.full((n_levels,), 0.1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        context = self.norm(states.reshape(states.shape[0], -1))
        updates = []
        for level, projection in enumerate(self.proj):
            delta = projection(context)
            updates.append(states[:, level] + self.scale[level].float() * delta.float())
        return torch.stack(updates, dim=1)


class RecurrentCore(nn.Module):
    def __init__(self, d_model: int, d_state: int, n_levels: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_levels = n_levels
        self.levels = nn.ModuleList([RecurrentLevel(d_model, d_state) for _ in range(n_levels)])
        self.interaction = HierarchicalInteraction(n_levels, d_state, norm_eps)
        self.out_norm = RMSNorm(n_levels * d_state, norm_eps)
        self.out_proj = nn.Linear(n_levels * d_state, d_model, bias=False)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.n_levels,
            self.d_state,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("RecurrentCore expects input of shape (batch, time, d_model)")
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size, x.device)
        state = state.float()
        if state.shape != (batch_size, self.n_levels, self.d_state):
            raise ValueError(
                f"Invalid recurrent state shape {tuple(state.shape)}, "
                f"expected {(batch_size, self.n_levels, self.d_state)}"
            )
        reset_mask = prepare_reset_mask(reset_mask, batch_size, seq_len, x.device)

        outputs = []
        for token_index in range(seq_len):
            state = apply_reset(state, reset_mask, token_index)
            x_token = x[:, token_index]
            next_levels = [
                level(x_token, state[:, level_index])
                for level_index, level in enumerate(self.levels)
            ]
            state = self.interaction(torch.stack(next_levels, dim=1))
            outputs.append(state)

        stacked = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, -1)
        return self.out_proj(self.out_norm(stacked)), state
