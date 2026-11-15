from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from aurora.data.dataset import MemmapTokenDataset
from aurora.data.prepare import main as prepare_main
from aurora.tokenizer import ByteTokenizer, SentencePieceTokenizer
from aurora.tokenizer.train_tokenizer import main as train_tokenizer_main


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "Привет, AURORA! def add(a, b):\n    return a + b"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_prepare_and_dataset(tmp_path: Path) -> None:
    text_path = tmp_path / "corpus.txt"
    text_path.write_text("AURORA learns language and code. " * 20, encoding="utf-8")
    output_dir = tmp_path / "processed"
    prepare_main(
        [
            "--input",
            str(text_path),
            "--output-dir",
            str(output_dir),
            "--tokenizer",
            "byte",
            "--val-fraction",
            "0.2",
        ]
    )
    assert (output_dir / "train.bin").is_file()
    assert (output_dir / "val.bin").is_file()
    dataset = MemmapTokenDataset(output_dir / "train.bin", seq_len=8)
    inputs, targets = dataset[0]
    assert inputs.shape == (8,)
    assert targets.shape == (8,)
    torch.testing.assert_close(inputs[1:], targets[:-1])
    assert inputs.dtype == torch.int64

    restored = pickle.loads(pickle.dumps(dataset))
    restored_inputs, restored_targets = restored[0]
    torch.testing.assert_close(restored_inputs, inputs)
    torch.testing.assert_close(restored_targets, targets)


def test_prepare_fallback_validation_split(tmp_path: Path) -> None:
    text = "AURORA learns a recurrent language model. " * 40
    text_path = tmp_path / "corpus.txt"
    text_path.write_text(text, encoding="utf-8")
    output_dir = tmp_path / "processed"
    prepare_main(
        [
            "--input",
            str(text_path),
            "--output-dir",
            str(output_dir),
            "--tokenizer",
            "byte",
            "--val-fraction",
            "0",
        ]
    )

    train = np.fromfile(output_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(output_dir / "val.bin", dtype=np.uint16)
    expected = np.asarray(ByteTokenizer().encode(text, add_eos=True), dtype=np.uint16)
    assert len(val) >= 1
    assert np.array_equal(np.concatenate([train, val]), expected)


def test_memmap_rejects_short_file(tmp_path: Path) -> None:
    path = tmp_path / "short.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        MemmapTokenDataset(path, seq_len=8)


def test_sentencepiece_tokenizer(tmp_path: Path) -> None:
    pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("AURORA learns Russian and English. " * 50, encoding="utf-8")
    prefix = tmp_path / "tokenizer"
    train_tokenizer_main(
        [
            "--input",
            str(corpus),
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "320",
            "--model-type",
            "bpe",
        ]
    )
    tokenizer = SentencePieceTokenizer(prefix.with_suffix(".model"))
    text = "AURORA learns Russian and English."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids).strip() == text
