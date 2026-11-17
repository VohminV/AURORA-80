from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def document_reset_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Mark tokens that start a new document after an EOS token."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, time)")
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if input_ids.shape[1] > 1:
        mask[:, 1:] = input_ids[:, :-1] == eos_token_id
    return mask


class MemmapTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        dtype: str = "uint16",
        stride: int | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Token file not found: {self.path}")
        self.seq_len = seq_len
        self.stride = stride or seq_len
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.dtype = np.dtype(dtype)
        self._tokens: np.memmap | None = None
        tokens = self._open_tokens()
        available = len(tokens) - self.seq_len - 1
        self.n_samples = available // self.stride + 1 if available >= 0 else 0
        if self.n_samples == 0:
            raise ValueError(
                f"{self.path} contains {len(tokens)} tokens, which is too few for seq_len={seq_len}"
            )

    def _open_tokens(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._tokens

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._tokens = None

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.n_samples:
            raise IndexError(index)
        start = index * self.stride
        tokens = self._open_tokens()
        chunk = np.asarray(tokens[start : start + self.seq_len + 1], dtype=np.int64)
        tensor = torch.from_numpy(chunk)
        return tensor[:-1], tensor[1:]
