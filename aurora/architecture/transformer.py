from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from aurora.architecture.language_head import LanguageHead
from aurora.architecture.layers import RMSNorm, init_module_weights
from aurora.architecture.output import LanguageModelOutput
from aurora.config import ModelConfig


@dataclass
class TransformerLayerState:
    keys: torch.Tensor
    values: torch.Tensor

    def detach(self) -> TransformerLayerState:
        return TransformerLayerState(keys=self.keys.detach(), values=self.values.detach())


@dataclass
class TransformerState:
    layers: list[TransformerLayerState]

    def detach(self) -> TransformerState:
        return TransformerState(layers=[layer.detach() for layer in self.layers])


def _rope_frequencies(
    seq_len: int, head_dim: int, device: torch.device, theta: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inverse = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1))
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse)
    return frequencies.cos(), frequencies.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cos = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.attention_heads
        self.head_dim = config.d_model // config.attention_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, state: TransformerLayerState | None = None
    ) -> tuple[torch.Tensor, TransformerLayerState]:
        batch_size, seq_len, _ = x.shape
        if state is not None and state.keys.shape[0] != batch_size:
            raise ValueError("KV-cache batch size does not match the input")
        if state is not None and state.keys.shape[1] != self.n_heads:
            raise ValueError("KV-cache head count does not match the model")

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        past_len = 0 if state is None else state.keys.shape[-2]
        cos, sin = _rope_frequencies(past_len + seq_len, self.head_dim, x.device)
        cos, sin = cos[past_len:], sin[past_len:]
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)

        if state is not None:
            key = torch.cat([state.keys, key], dim=-2)
            value = torch.cat([state.values, value], dim=-2)

        attn_mask = None
        is_causal = state is None
        if state is not None and seq_len > 1:
            mask = torch.full(
                (seq_len, past_len + seq_len),
                torch.finfo(query.dtype).min,
                device=x.device,
                dtype=query.dtype,
            )
            attn_mask = torch.triu(mask, diagonal=past_len + 1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
        next_state = TransformerLayerState(keys=key, values=value)
        return self.out_proj(attended), next_state


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_ffn = RMSNorm(config.d_model, config.norm_eps)
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, state: TransformerLayerState | None = None
    ) -> tuple[torch.Tensor, TransformerLayerState]:
        attended, next_state = self.attn(self.norm_attn(x), state)
        x = x + self.dropout(attended)
        hidden = self.norm_ffn(x)
        hidden = F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        x = x + self.dropout(self.down_proj(hidden))
        return x, next_state


class TransformerModel(nn.Module):
    """Decoder-only Transformer baseline with RoPE and a KV-cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture != "transformer":
            raise ValueError("TransformerModel requires architecture='transformer'")
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = LanguageHead(
            config.d_model,
            config.vocab_size,
            norm_eps=config.norm_eps,
            embedding_weight=self.embed.weight if config.tie_embeddings else None,
        )

        self.apply(lambda module: init_module_weights(module, config.init_std))
        residual_scale = 1.0 / math.sqrt(2.0 * max(config.n_layers, 1))
        with torch.no_grad():
            for block in self.blocks:
                assert isinstance(block, TransformerBlock)
                block.attn.out_proj.weight.mul_(residual_scale)
                block.down_proj.weight.mul_(residual_scale)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> TransformerState:
        layers = [
            TransformerLayerState(
                keys=torch.zeros(
                    batch_size, self.config.attention_heads, 0, dtype=dtype, device=device
                ),
                values=torch.zeros(
                    batch_size, self.config.attention_heads, 0, dtype=dtype, device=device
                ),
            )
            for _ in range(self.config.n_layers)
        ]
        return TransformerState(layers=layers)

    def forward(
        self, input_ids: torch.Tensor, state: TransformerState | None = None
    ) -> LanguageModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        if state is not None and len(state.layers) != len(self.blocks):
            raise ValueError("State does not match model depth")

        x = self.dropout(self.embed(input_ids))
        layer_states = state.layers if state is not None else [None] * len(self.blocks)
        next_states: list[TransformerLayerState] = []
        for block, layer_state in zip(self.blocks, layer_states, strict=True):
            assert isinstance(block, TransformerBlock)
            x, next_state = block(x, layer_state)
            next_states.append(next_state)

        logits = self.lm_head(x)
        zeros = torch.zeros((), device=x.device, dtype=torch.float32)
        return LanguageModelOutput(
            logits=logits,
            aux_loss=zeros,
            memory_loss=zeros,
            consistency_loss=zeros,
            state=TransformerState(layers=next_states),
        )

    @torch.no_grad()
    def num_parameters(self) -> tuple[int, int]:
        from aurora.utils import parameter_counts

        return parameter_counts(self)
