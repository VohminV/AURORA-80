from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path


class BaseTokenizer(ABC):
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError


class ByteTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = [byte + self.byte_offset for byte in text.encode("utf-8", errors="replace")]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        payload = bytes(
            token_id - self.byte_offset
            for token_id in token_ids
            if self.byte_offset <= token_id < self.vocab_size
        )
        return payload.decode("utf-8", errors="replace")


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: str | Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise ImportError("sentencepiece is required for SentencePieceTokenizer") from error

        self.processor = spm.SentencePieceProcessor()
        self.processor.load(str(model_path))
        self.vocab_size = int(self.processor.get_piece_size())
        self.pad_id = int(self.processor.pad_id())
        self.bos_id = int(self.processor.bos_id())
        self.eos_id = int(self.processor.eos_id())
        self.unk_id = int(self.processor.unk_id())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        filtered = [
            int(token_id)
            for token_id in token_ids
            if int(token_id) not in {self.pad_id, self.bos_id, self.eos_id}
        ]
        return str(self.processor.decode(filtered))


def load_tokenizer(kind: str, model_path: str | Path = "") -> BaseTokenizer:
    if kind == "byte":
        return ByteTokenizer()
    if kind == "sentencepiece":
        if not model_path:
            raise ValueError("model_path is required for sentencepiece tokenizer")
        return SentencePieceTokenizer(model_path)
    raise ValueError(f"Unknown tokenizer kind: {kind}")
