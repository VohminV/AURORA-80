from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from aurora.data.dataset import MemmapTokenDataset
from aurora.data.prepare import FileFingerprint, iter_text_chunks
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


def test_streaming_chunks_decode_multibyte_characters(tmp_path: Path) -> None:
    text = "Первая строка\nВторая строка\nТретья строка\n"
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    fingerprint = FileFingerprint()
    chunks = list(iter_text_chunks(path, chunk_size=1, encoding="utf-8", fingerprint=fingerprint))
    assert "".join(chunks) == text
    assert fingerprint.bytes == len(text.encode("utf-8"))
    assert fingerprint.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_prepare_streaming_matches_reference_and_writes_manifest(tmp_path: Path) -> None:
    text = "AURORA learns language.\nОна изучает код.\n\ndef add(a, b):\n    return a + b\n" * 40
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
            "0.2",
            "--chunk-size",
            "17",
        ]
    )

    expected = np.asarray(ByteTokenizer().encode(text, add_eos=True), dtype=np.uint16)
    train = np.fromfile(output_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(output_dir / "val.bin", dtype=np.uint16)
    assert np.array_equal(np.concatenate([train, val]), expected)
    assert 0 < len(val) < len(expected)
    assert len(val) == pytest.approx(0.2 * len(expected), abs=len(expected) // 20)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["dtype"] == "uint16"
    assert manifest["tokenizer"]["kind"] == "byte"
    assert manifest["inputs"][0]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert manifest["inputs"][0]["tokens"] == len(expected)
    assert manifest["outputs"]["train"]["tokens"] == len(train)
    assert manifest["outputs"]["val"]["tokens"] == len(val)
    assert (
        manifest["outputs"]["train"]["sha256"]
        == hashlib.sha256((output_dir / "train.bin").read_bytes()).hexdigest()
    )
    assert (
        manifest["outputs"]["val"]["sha256"]
        == hashlib.sha256((output_dir / "val.bin").read_bytes()).hexdigest()
    )


def test_prepare_handles_multiple_files_and_cleans_temporary_files(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("First document.\n" * 10, encoding="utf-8")
    second.write_text("Second document.\n" * 10, encoding="utf-8")
    output_dir = tmp_path / "processed"
    prepare_main(
        [
            "--input",
            str(first),
            str(second),
            "--output-dir",
            str(output_dir),
            "--tokenizer",
            "byte",
            "--chunk-size",
            "23",
        ]
    )
    assert not (output_dir / "train.bin.tmp").exists()
    assert not (output_dir / "val.bin.tmp").exists()
    train = np.fromfile(output_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(output_dir / "val.bin", dtype=np.uint16)
    expected = np.asarray(
        ByteTokenizer().encode(first.read_text(), add_eos=True)
        + ByteTokenizer().encode(second.read_text(), add_eos=True),
        dtype=np.uint16,
    )
    assert len(train) + len(val) == len(expected)
    assert np.array_equal(
        np.sort(np.concatenate([train, val])),
        np.sort(expected),
    )


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


def test_token_stats_for_byte_tokenizer(tmp_path: Path) -> None:
    from aurora.tokenizer.stats import compute_token_stats

    text = "Hello world. Привет мир.\nprint('hi')\n"
    path = tmp_path / "ru_en_code.txt"
    path.write_text(text, encoding="utf-8")
    result = compute_token_stats(path, ByteTokenizer(), chunk_size=7)
    encoded = text.encode("utf-8")
    assert result.bytes == len(encoded)
    assert result.tokens == len(encoded)
    assert result.characters == len(text)
    assert result.bytes_per_token == pytest.approx(1.0)
    assert result.tokens_per_char > 1.0
    assert result.tokens_per_word > 1.0


def test_token_stats_cli_writes_report(tmp_path: Path) -> None:
    from aurora.tokenizer.stats import main as stats_main

    path = tmp_path / "corpus.txt"
    path.write_text("AURORA tests tokenizer statistics.\n" * 20, encoding="utf-8")
    output = tmp_path / "reports" / "tokenizer_stats.json"
    stats_main(
        [
            "--input",
            str(path),
            "--tokenizer",
            "byte",
            "--chunk-size",
            "13",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tokenizer"]["kind"] == "byte"
    assert report["tokenizer"]["vocab_size"] == 260
    assert report["chunk_size_bytes"] == 13
    assert report["files"][0]["tokens"] == len(path.read_bytes())
    assert report["files"][0]["bytes_per_token"] == pytest.approx(1.0)
