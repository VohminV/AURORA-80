from __future__ import annotations

import argparse
import codecs
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from aurora.tokenizer.tokenizer import BaseTokenizer, load_tokenizer

DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass
class TokenStats:
    path: str
    bytes: int = 0
    characters: int = 0
    words: int = 0
    lines: int = 0
    tokens: int = 0

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / max(self.tokens, 1)

    @property
    def chars_per_token(self) -> float:
        return self.characters / max(self.tokens, 1)

    @property
    def tokens_per_char(self) -> float:
        return self.tokens / max(self.characters, 1)

    @property
    def tokens_per_word(self) -> float:
        return self.tokens / max(self.words, 1)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "bytes_per_token": self.bytes_per_token,
                "chars_per_token": self.chars_per_token,
                "tokens_per_char": self.tokens_per_char,
                "tokens_per_word": self.tokens_per_word,
            }
        )
        return result


def iter_text_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[str]:
    """Yield newline-aligned UTF-8 chunks without loading the whole file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    with path.open("rb") as handle:
        while True:
            raw = handle.read(chunk_size)
            if not raw:
                break
            buffer += decoder.decode(raw)
            cut = buffer.rfind("\n")
            if cut != -1:
                yield buffer[: cut + 1]
                buffer = buffer[cut + 1 :]
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer


def compute_token_stats(
    path: str | Path,
    tokenizer: BaseTokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> TokenStats:
    resolved = Path(path)
    stats = TokenStats(path=str(resolved), bytes=resolved.stat().st_size)
    for chunk in iter_text_chunks(resolved, chunk_size=chunk_size):
        stats.characters += len(chunk)
        stats.words += len(chunk.split())
        stats.lines += chunk.count("\n")
        stats.tokens += len(tokenizer.encode(chunk))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure tokenizer fertility and compression on text files"
    )
    parser.add_argument("--input", nargs="+", required=True, help="Text files to analyze")
    parser.add_argument("--tokenizer", choices=["byte", "sentencepiece"], default="byte")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--output", default="", help="Optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_model)
    results = []
    for input_path in args.input:
        path = Path(input_path)
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")
        stats = compute_token_stats(path, tokenizer, chunk_size=args.chunk_size)
        results.append(stats)
        print(
            f"{path}: tokens {stats.tokens:,} | bytes/token {stats.bytes_per_token:.3f} | "
            f"chars/token {stats.chars_per_token:.3f} | tokens/word {stats.tokens_per_word:.3f}"
        )

    if args.output:
        report = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tokenizer": {
                "kind": args.tokenizer,
                "model": args.tokenizer_model or None,
                "vocab_size": tokenizer.vocab_size,
            },
            "chunk_size_bytes": args.chunk_size,
            "files": [stats.to_dict() for stats in results],
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {output}")


if __name__ == "__main__":
    main()
