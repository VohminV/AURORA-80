from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockState:
    core: torch.Tensor
    memory: torch.Tensor

    def detach(self) -> BlockState:
        return BlockState(core=self.core.detach(), memory=self.memory.detach())


@dataclass
class AuroraState:
    blocks: list[BlockState]

    def detach(self) -> AuroraState:
        return AuroraState(blocks=[block.detach() for block in self.blocks])


def prepare_reset_mask(
    reset_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if reset_mask is None:
        return None
    mask = torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
    if mask.ndim != 2 or tuple(mask.shape) != (batch_size, seq_len):
        raise ValueError(
            f"reset_mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
        )
    return mask


def apply_reset(
    state: torch.Tensor, reset_mask: torch.Tensor | None, token_index: int
) -> torch.Tensor:
    if reset_mask is None:
        return state
    keep = (~reset_mask[:, token_index]).to(dtype=state.dtype)
    return state * keep.reshape((state.shape[0],) + (1,) * (state.ndim - 1))
