from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aurora.data.version import compute_dataset_version, sha256_file
from aurora.tokenizer import BaseTokenizer, load_tokenizer

DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass
class FileFingerprint:
    bytes: int = 0
    sha256: str = ""


class TokenWriter:
    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = path
        self.dtype = dtype
        self.count = 0
        self._handle = path.open("wb")

    def write(self, tokens: Sequence[int]) -> None:
        if not tokens:
            return
        limit = np.iinfo(self.dtype).max
        max_token = max(tokens)
        if max_token > limit:
            raise ValueError(
                f"Token id {max_token} does not fit into {self.dtype.name}; use --dtype uint32"
            )
        array = np.asarray(tokens, dtype=self.dtype)
        self._handle.write(array.tobytes())
        self.count += int(array.size)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> TokenWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize a text corpus for AURORA")
    parser.add_argument("--input", nargs="+", required=True, help="Input text files")
    parser.add_argument("--output-dir", required=True, help="Directory for train.bin and val.bin")
    parser.add_argument("--tokenizer", choices=["byte", "sentencepiece"], default="byte")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--split-mode",
        choices=["token", "document"],
        default="token",
        help="Split by token offset or by whole blank-line-separated documents",
    )
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of input bytes held in memory at once",
    )
    return parser


def iter_text_chunks(
    path: Path,
    chunk_size: int,
    encoding: str,
    fingerprint: FileFingerprint | None = None,
) -> Iterator[str]:
    """Yield newline-aligned text chunks while keeping memory usage bounded.

    Files are read in binary chunks and decoded incrementally so that multi-byte
    characters are never split. Only complete lines are yielded; the final
    partial line is emitted when the file ends.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    hasher = hashlib.sha256()
    buffer = ""
    total_bytes = 0

    with path.open("rb") as handle:
        while True:
            raw = handle.read(chunk_size)
            if not raw:
                break
            total_bytes += len(raw)
            hasher.update(raw)
            buffer += decoder.decode(raw)
            cut = buffer.rfind("\n")
            if cut != -1:
                yield buffer[: cut + 1]
                buffer = buffer[cut + 1 :]

    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer

    if fingerprint is not None:
        fingerprint.bytes = total_bytes
        fingerprint.sha256 = hasher.hexdigest()


def iter_text_documents(
    path: Path,
    chunk_size: int,
    encoding: str,
    fingerprint: FileFingerprint | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Yield blank-line-separated documents with byte offsets.

    Documents are streamed one at a time. Leading blank lines are attached to the
    next document, so the concatenation of all documents preserves the input.
    """
    document: list[str] = []
    has_content = False
    document_start = 0
    offset = 0
    for chunk in iter_text_chunks(path, chunk_size, encoding, fingerprint):
        for line in chunk.splitlines(keepends=True):
            if not document:
                document_start = offset
            document.append(line)
            offset += len(line.encode(encoding, errors="replace"))
            if line.strip():
                has_content = True
            elif has_content:
                yield "".join(document), document_start, offset
                document = []
                has_content = False
    if document:
        yield "".join(document), document_start, offset


def _iter_file_tokens(
    path: Path,
    tokenizer: BaseTokenizer,
    chunk_size: int,
    encoding: str,
    fingerprint: FileFingerprint,
) -> Iterator[tuple[list[int], int, int]]:
    """Yield (tokens, start_byte, end_byte) with EOS appended once per file."""
    chunks = iter_text_chunks(path, chunk_size, encoding, fingerprint)
    pending = next(chunks, None)
    if pending is None:
        if tokenizer.eos_id >= 0:
            yield [tokenizer.eos_id], 0, 0
        return
    offset = 0
    while pending is not None:
        text = pending
        pending = next(chunks, None)
        tokens = tokenizer.encode(text)
        end_offset = offset + len(text.encode(encoding, errors="replace"))
        if pending is None:
            if tokenizer.eos_id >= 0:
                tokens.append(tokenizer.eos_id)
            end_offset = fingerprint.bytes
        if tokens:
            yield tokens, offset, end_offset
        offset = end_offset


def _iter_file_document_tokens(
    path: Path,
    tokenizer: BaseTokenizer,
    chunk_size: int,
    encoding: str,
    fingerprint: FileFingerprint,
) -> Iterator[tuple[list[int], int, int]]:
    """Yield whole-document tokens with an EOS after every document."""
    yielded = False
    for text, start_offset, end_offset in iter_text_documents(
        path, chunk_size, encoding, fingerprint
    ):
        tokens = tokenizer.encode(text)
        if tokenizer.eos_id >= 0:
            tokens.append(tokenizer.eos_id)
        if tokens:
            yielded = True
            yield tokens, start_offset, end_offset
    if not yielded and tokenizer.eos_id >= 0:
        yield [tokenizer.eos_id], 0, 0


def _boundary_cut(tokens: list[int], start_offset: int, end_offset: int, boundary: int) -> int:
    if boundary <= start_offset:
        return 0
    if boundary >= end_offset:
        return len(tokens)
    span = max(end_offset - start_offset, 1)
    fraction = (boundary - start_offset) / span
    return min(max(int(len(tokens) * fraction), 0), len(tokens))


def _write_split(
    tokens: list[int],
    start_offset: int,
    end_offset: int,
    train_end_offset: int,
    val_end_offset: int,
    train_writer: TokenWriter,
    val_writer: TokenWriter,
    test_writer: TokenWriter | None,
) -> None:
    train_cut = _boundary_cut(tokens, start_offset, end_offset, train_end_offset)
    val_cut = _boundary_cut(tokens, start_offset, end_offset, val_end_offset)
    train_writer.write(tokens[:train_cut])
    val_writer.write(tokens[train_cut:val_cut])
    if test_writer is not None:
        test_writer.write(tokens[val_cut:])
    elif val_cut != len(tokens):
        raise ValueError("Test tokens were produced without a test writer")


def _move_tail_to_validation(
    train_path: Path, val_path: Path, dtype: np.dtype, fraction: float = 0.01
) -> int:
    itemsize = dtype.itemsize
    total_tokens = train_path.stat().st_size // itemsize
    if total_tokens < 2:
        raise SystemExit("Not enough tokens to create both train and validation splits")

    moved = max(int(total_tokens * fraction), 1)
    moved = min(moved, total_tokens - 1)
    boundary = (total_tokens - moved) * itemsize

    with train_path.open("r+b") as train_handle:
        train_handle.seek(boundary)
        tail = train_handle.read()
        train_handle.truncate(boundary)
    with val_path.open("ab") as val_handle:
        val_handle.write(tail)
    return moved


def _output_entry(path: Path, tokens: int) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tokens": tokens,
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0, 1)")
    if not 0.0 <= args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be in [0, 1)")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise SystemExit("--val-fraction + --test-fraction must be less than 1")
    if args.split_mode == "document" and args.val_fraction <= 0.0:
        raise SystemExit("--split-mode document requires --val-fraction > 0")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_model)
    if args.dtype == "uint16" and tokenizer.vocab_size > 65536:
        raise SystemExit("Tokenizer vocabulary is too large for uint16; use --dtype uint32")
    tokenizer_entry: dict[str, object] = {
        "kind": args.tokenizer,
        "model": args.tokenizer_model or None,
        "model_sha256": None,
        "vocab_size": tokenizer.vocab_size,
    }
    if args.tokenizer_model:
        tokenizer_model_path = Path(args.tokenizer_model)
        tokenizer_entry["model_bytes"] = tokenizer_model_path.stat().st_size
        tokenizer_entry["model_sha256"] = sha256_file(tokenizer_model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    test_path = output_dir / "test.bin"
    temporary_train = output_dir / "train.bin.tmp"
    temporary_val = output_dir / "val.bin.tmp"
    temporary_test = output_dir / "test.bin.tmp"
    np_dtype = np.dtype(args.dtype)

    input_entries: list[dict[str, object]] = []
    train_writer = TokenWriter(temporary_train, np_dtype)
    val_writer = TokenWriter(temporary_val, np_dtype)
    test_writer = TokenWriter(temporary_test, np_dtype) if args.test_fraction > 0.0 else None
    completed = False
    try:
        try:
            for input_path in args.input:
                path = Path(input_path)
                if not path.is_file():
                    raise SystemExit(f"Input file not found: {path}")

                file_size = path.stat().st_size
                train_end_offset = min(
                    int(file_size * (1.0 - args.val_fraction - args.test_fraction)), file_size
                )
                val_end_offset = min(int(file_size * (1.0 - args.test_fraction)), file_size)

                fingerprint = FileFingerprint()
                file_tokens = 0
                if args.split_mode == "document":
                    token_iterator = _iter_file_document_tokens(
                        path, tokenizer, args.chunk_size, args.encoding, fingerprint
                    )
                else:
                    token_iterator = _iter_file_tokens(
                        path, tokenizer, args.chunk_size, args.encoding, fingerprint
                    )

                for tokens, start_offset, end_offset in token_iterator:
                    file_tokens += len(tokens)
                    if args.split_mode == "document":
                        if start_offset < train_end_offset:
                            train_writer.write(tokens)
                        elif start_offset < val_end_offset:
                            val_writer.write(tokens)
                        elif test_writer is not None:
                            test_writer.write(tokens)
                        else:
                            raise SystemExit("Document split produced test tokens unexpectedly")
                    else:
                        _write_split(
                            tokens,
                            start_offset,
                            end_offset,
                            train_end_offset,
                            val_end_offset,
                            train_writer,
                            val_writer,
                            test_writer,
                        )
                input_entries.append(
                    {
                        "path": str(path),
                        "bytes": fingerprint.bytes,
                        "sha256": fingerprint.sha256,
                        "tokens": file_tokens,
                    }
                )
                print(f"{path}: {file_tokens:,} tokens")
        finally:
            train_writer.close()
            val_writer.close()
            if test_writer is not None:
                test_writer.close()

        if train_writer.count == 0:
            raise SystemExit("No training tokens were produced")
        if val_writer.count == 0:
            if args.split_mode == "document":
                raise SystemExit(
                    "Document split produced no validation tokens; "
                    "add more documents or use --split-mode token"
                )
            moved = _move_tail_to_validation(temporary_train, temporary_val, np_dtype)
            train_writer.count -= moved
            val_writer.count += moved
            print(f"validation split was empty; moved {moved:,} tokens from the training tail")
        if args.test_fraction > 0.0 and (test_writer is None or test_writer.count == 0):
            raise SystemExit("Test split produced no test tokens; adjust --test-fraction")

        os.replace(temporary_train, train_path)
        os.replace(temporary_val, val_path)
        if test_writer is not None:
            os.replace(temporary_test, test_path)

        outputs: dict[str, object] = {
            "train": _output_entry(train_path, train_writer.count),
            "val": _output_entry(val_path, val_writer.count),
        }
        if test_writer is not None:
            outputs["test"] = _output_entry(test_path, test_writer.count)

        manifest: dict[str, object] = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tokenizer": tokenizer_entry,
            "encoding": args.encoding,
            "dtype": args.dtype,
            "split_mode": args.split_mode,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "chunk_size_bytes": args.chunk_size,
            "inputs": input_entries,
            "outputs": outputs,
        }
        manifest["dataset_version"] = compute_dataset_version(manifest)
        _write_manifest(output_dir / "manifest.json", manifest)
        completed = True
    finally:
        if not completed:
            temporary_train.unlink(missing_ok=True)
            temporary_val.unlink(missing_ok=True)
            temporary_test.unlink(missing_ok=True)

    print(f"train: {train_writer.count:,} tokens -> {train_path}")
    print(f"val:   {val_writer.count:,} tokens -> {val_path}")
    if test_writer is not None:
        print(f"test:  {test_writer.count:,} tokens -> {test_path}")
    print(f"manifest -> {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
