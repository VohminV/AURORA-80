from __future__ import annotations

import torch
from torch import nn

from aurora.architecture.associative_memory import AssociativeMemory
from aurora.architecture.expert_router import MoE
from aurora.architecture.integration import ResidualIntegration
from aurora.architecture.layers import RMSNorm
from aurora.architecture.recurrent_core import RecurrentCore
from aurora.architecture.state import BlockState
from aurora.config import ModelConfig


class AuroraBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_core = RMSNorm(config.d_model, config.norm_eps)
        self.core = RecurrentCore(config.d_model, config.d_state, config.n_levels, config.norm_eps)
        self.integrate_core = ResidualIntegration(
            config.d_model, config.layer_scale_init, config.dropout
        )

        self.norm_memory = RMSNorm(config.d_model, config.norm_eps)
        self.memory = AssociativeMemory(
            config.d_model,
            config.memory_heads,
            decay_init=config.memory_decay_init,
            write_init=config.memory_write_init,
            epsilon=config.memory_epsilon,
            dropout=config.dropout,
            norm_eps=config.norm_eps,
        )
        self.integrate_memory = ResidualIntegration(
            config.d_model, config.layer_scale_init, config.dropout
        )

        self.norm_moe = RMSNorm(config.d_model, config.norm_eps)
        self.moe = MoE(
            config.d_model,
            config.n_experts,
            config.top_k,
            config.expert_hidden,
            dropout=config.dropout,
            aux_loss_coef=config.aux_loss_coef,
            router_z_loss_coef=config.router_z_loss_coef,
            router_jitter=config.router_jitter,
        )
        self.integrate_moe = ResidualIntegration(
            config.d_model, config.layer_scale_init, config.dropout
        )

    def init_state(self, batch_size: int, device: torch.device) -> BlockState:
        return BlockState(
            core=self.core.init_state(batch_size, device),
            memory=self.memory.init_state(batch_size, device),
        )

    def forward(
        self, x: torch.Tensor, state: BlockState | None = None
    ) -> tuple[torch.Tensor, BlockState, torch.Tensor]:
        if state is None:
            state = self.init_state(x.shape[0], x.device)

        core_output, core_state = self.core(self.norm_core(x), state.core)
        x = self.integrate_core(x, core_output)

        memory_output, memory_state = self.memory(self.norm_memory(x), state.memory)
        x = self.integrate_memory(x, memory_output)

        moe_output, aux_loss = self.moe(self.norm_moe(x))
        x = self.integrate_moe(x, moe_output)

        return x, BlockState(core=core_state, memory=memory_state), aux_loss
