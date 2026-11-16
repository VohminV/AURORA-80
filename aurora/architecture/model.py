from __future__ import annotations

import torch
from torch import nn

from aurora.architecture.block import AuroraBlock
from aurora.architecture.language_head import LanguageHead
from aurora.architecture.layers import init_module_weights, scale_residual_outputs
from aurora.architecture.output import LanguageModelOutput
from aurora.architecture.state import AuroraState
from aurora.config import ModelConfig

AuroraOutput = LanguageModelOutput


class AuroraModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([AuroraBlock(config) for _ in range(config.n_layers)])
        self.lm_head = LanguageHead(
            config.d_model,
            config.vocab_size,
            norm_eps=config.norm_eps,
            embedding_weight=self.embed.weight if config.tie_embeddings else None,
        )

        self.apply(lambda module: init_module_weights(module, config.init_std))
        for block in self.blocks:
            assert isinstance(block, AuroraBlock)
            if block.use_memory:
                block.memory.reset_gate_biases(config.memory_decay_init, config.memory_write_init)
        scale_residual_outputs(self, config.n_layers)

    def init_state(self, batch_size: int, device: torch.device) -> AuroraState:
        states = []
        for block in self.blocks:
            assert isinstance(block, AuroraBlock)
            states.append(block.init_state(batch_size, device))
        return AuroraState(blocks=states)

    def forward(self, input_ids: torch.Tensor, state: AuroraState | None = None) -> AuroraOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        if state is not None and len(state.blocks) != len(self.blocks):
            raise ValueError("State does not match model depth")

        x = self.dropout(self.embed(input_ids))
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        memory_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        consistency_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        block_states = state.blocks if state is not None else [None] * len(self.blocks)
        next_states = []

        for block, block_state in zip(self.blocks, block_states, strict=True):
            assert isinstance(block, AuroraBlock)
            x, next_state, block_aux_loss, block_memory_loss, block_consistency_loss = block(
                x, block_state
            )
            aux_loss = aux_loss + block_aux_loss.float()
            memory_loss = memory_loss + block_memory_loss.float()
            consistency_loss = consistency_loss + block_consistency_loss.float()
            next_states.append(next_state)

        logits = self.lm_head(x)
        return AuroraOutput(
            logits=logits,
            aux_loss=aux_loss,
            memory_loss=memory_loss,
            consistency_loss=consistency_loss,
            state=AuroraState(blocks=next_states),
        )

    @torch.no_grad()
    def num_parameters(self) -> tuple[int, int]:
        from aurora.utils import parameter_counts

        return parameter_counts(self)
