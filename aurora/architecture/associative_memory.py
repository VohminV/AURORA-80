from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aurora.architecture.layers import RMSNorm


@dataclass
class MemoryLosses:
    """Auxiliary memory objectives.

    ``prediction`` is the one-step key-value prediction error before the write:
    the memory is rewarded for anticipating the value associated with a key.
    ``consistency`` is the write-read error after the write: the memory is
    rewarded for storing the association it just observed.
    """

    prediction: torch.Tensor
    consistency: torch.Tensor


def zero_memory_losses(device: torch.device) -> MemoryLosses:
    zero = torch.zeros((), device=device, dtype=torch.float32)
    return MemoryLosses(prediction=zero, consistency=zero)


class AssociativeMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        decay_init: float = 0.9,
        write_init: float = 0.5,
        epsilon: float = 1e-6,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.epsilon = epsilon

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = RMSNorm(d_model, norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.decay_proj = nn.Linear(d_model, n_heads, bias=True)
        self.write_proj = nn.Linear(d_model, n_heads, bias=True)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.decay_proj.bias.fill_(math.log(decay_init / (1.0 - decay_init)))
            self.write_proj.bias.fill_(math.log(write_init / (1.0 - write_init)))

    def reset_gate_biases(self, decay_init: float = 0.9, write_init: float = 0.5) -> None:
        with torch.no_grad():
            self.decay_proj.bias.fill_(math.log(decay_init / (1.0 - decay_init)))
            self.write_proj.bias.fill_(math.log(write_init / (1.0 - write_init)))

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_state, _ = self.forward_with_losses(x, state, track_losses=False)
        return output, next_state

    def forward_with_losses(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        track_losses: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, MemoryLosses]:
        if x.ndim != 3:
            raise ValueError("AssociativeMemory expects input of shape (batch, time, d_model)")
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size, x.device)
        state = state.float()
        expected_shape = (batch_size, self.n_heads, self.head_dim, self.head_dim)
        if state.shape != expected_shape:
            raise ValueError(
                f"Invalid memory state shape {tuple(state.shape)}, expected {expected_shape}"
            )

        query = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        key = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        query = F.normalize(query.float(), dim=-1).permute(0, 2, 1, 3)
        key = F.normalize(key.float(), dim=-1).permute(0, 2, 1, 3)
        value = value.float().permute(0, 2, 1, 3)

        decay = torch.sigmoid(self.decay_proj(x).float()).permute(0, 2, 1)
        write = torch.sigmoid(self.write_proj(x).float()).permute(0, 2, 1)

        losses = zero_memory_losses(x.device)
        reads = []
        for token_index in range(seq_len):
            query_token = query[:, :, token_index]
            key_token = key[:, :, token_index]
            value_token = value[:, :, token_index]

            read = (query_token.unsqueeze(-1) * state).sum(dim=-2)
            reads.append(read)

            prediction = (key_token.unsqueeze(-1) * state).sum(dim=-2)
            error = value_token - prediction
            if track_losses:
                losses.prediction = losses.prediction + error.square().mean()
            normalization = key_token.square().sum(dim=-1, keepdim=True) + self.epsilon
            rate = (write[:, :, token_index].unsqueeze(-1) / normalization).unsqueeze(-1)
            state = decay[:, :, token_index].unsqueeze(-1).unsqueeze(
                -1
            ) * state + rate * key_token.unsqueeze(-1) * error.unsqueeze(-2)
            if track_losses:
                stored = (key_token.unsqueeze(-1) * state).sum(dim=-2)
                losses.consistency = losses.consistency + (value_token - stored).square().mean()

        if track_losses:
            losses.prediction = losses.prediction / seq_len
            losses.consistency = losses.consistency / seq_len

        memory_output = torch.stack(reads, dim=2)
        memory_output = memory_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        memory_output = self.out_norm(memory_output.to(x.dtype))
        return self.dropout(self.out_proj(memory_output)), state, losses
