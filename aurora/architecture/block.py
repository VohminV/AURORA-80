from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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

        self.use_memory = config.use_memory
        if self.use_memory:
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

        self.use_moe = config.use_moe
        if self.use_moe:
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

        self.track_memory_losses = self.use_memory and (
            config.memory_loss_coef > 0.0 or config.consistency_loss_coef > 0.0
        )
        self.gradient_checkpointing = False

    def init_state(self, batch_size: int, device: torch.device) -> BlockState:
        return BlockState(
            core=self.core.init_state(batch_size, device),
            memory=(
                self.memory.init_state(batch_size, device)
                if self.use_memory
                else torch.zeros(0, device=device)
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: BlockState | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BlockState, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_state(x.shape[0], x.device)

        use_checkpoint = self.gradient_checkpointing and self.training

        core_input = self.norm_core(x)
        if use_checkpoint:
            core_output, core_state = checkpoint(
                self.core, core_input, state.core, reset_mask, use_reentrant=False
            )
        else:
            core_output, core_state = self.core(core_input, state.core, reset_mask)
        x = self.integrate_core(x, core_output)

        if self.use_memory:
            memory_input = self.norm_memory(x)
            track_memory_losses = self.track_memory_losses

            def memory_branch(
                hidden: torch.Tensor,
                memory_state: torch.Tensor,
                mask: torch.Tensor | None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                output, next_state, losses = self.memory.forward_with_losses(
                    hidden, memory_state, track_losses=track_memory_losses, reset_mask=mask
                )
                return output, next_state, losses.prediction, losses.consistency

            if use_checkpoint:
                memory_output, memory_state, prediction_loss, consistency_loss = checkpoint(
                    memory_branch, memory_input, state.memory, reset_mask, use_reentrant=False
                )
            else:
                memory_output, memory_state, prediction_loss, consistency_loss = memory_branch(
                    memory_input, state.memory, reset_mask
                )
            x = self.integrate_memory(x, memory_output)
        else:
            memory_state = state.memory
            prediction_loss = torch.zeros((), device=x.device, dtype=torch.float32)
            consistency_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        if self.use_moe:
            moe_input = self.norm_moe(x)
            if use_checkpoint:
                moe_output, aux_loss = checkpoint(self.moe, moe_input, use_reentrant=False)
            else:
                moe_output, aux_loss = self.moe(moe_input)
            x = self.integrate_moe(x, moe_output)
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        return (
            x,
            BlockState(core=core_state, memory=memory_state),
            aux_loss,
            prediction_loss,
            consistency_loss,
        )
