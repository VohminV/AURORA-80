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
