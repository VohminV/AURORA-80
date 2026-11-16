from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aurora.data.prepare import iter_text_chunks, iter_text_documents

FORMAT_VERSION = 1
_MASK = (1 << 64) - 1
_BASE = 0x100000001B3
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _Document:
    path: str
    index: int
    text: str
    offset: int | None = None
    line: int | None = None


class _NgramHasher:
    """Deterministic 64-bit word n-gram hashes with an O(1) rolling update."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.base_power = pow(_BASE, size - 1, 1 << 64)
        self.word_hashes: dict[str, int] = {}

    def _word_hash(self, word: str) -> int:
        value = self.word_hashes.get(word)
        if value is None:
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            self.word_hashes[word] = value
        return value

    def hashes(self, words: Sequence[str]) -> Iterator[tuple[int, int]]:
        size = self.size
        if len(words) < size:
            return
        window = [0] * size
        value = 0
        for index in range(size):
            word_hash = self._word_hash(words[index])
            window[index] = word_hash
            value = (value * _BASE + word_hash) & _MASK
        yield value, size - 1
        for index in range(size, len(words)):
            slot = index % size
            oldest = window[slot]
            newest = self._word_hash(words[index])
            window[slot] = newest
            value = ((value - oldest * self.base_power) * _BASE + newest) & _MASK
            yield value, index


def _normalized_words(text: str) -> list[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def _resolve_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


def _iter_documents(
    path: Path,
    input_format: str,
    json_field: str,
    chunk_size: int,
    encoding: str,
) -> Iterator[_Document]:
    resolved_format = _resolve_input_format(path, input_format)
    if resolved_format == "text":
        for index, (text, start, _) in enumerate(iter_text_documents(path, chunk_size, encoding)):
            yield _Document(path=str(path), index=index, text=text, offset=start)
        return

    line_number = 0
    document_index = 0
    for chunk in iter_text_chunks(path, chunk_size, encoding):
        for line in chunk.splitlines():
            line_number += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict) or not isinstance(record.get(json_field), str):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object with string field {json_field!r}"
                )
            yield _Document(
                path=str(path),
                index=document_index,
                text=record[json_field],
                line=line_number,
            )
            document_index += 1


def check_contamination(
    corpus_paths: Sequence[str | Path],
    benchmark_paths: Sequence[str | Path],
    *,
    ngram_size: int = 13,
    input_format: str = "auto",
    json_field: str = "text",
    chunk_size: int = 1024 * 1024,
    encoding: str = "utf-8",
    max_examples: int = 20,
) -> dict[str, Any]:
    """Scan a corpus for word n-grams that also occur in benchmark documents."""
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not json_field:
        raise ValueError("json_field must not be empty")
    if input_format not in {"auto", "text", "jsonl"}:
        raise ValueError("format must be one of auto, text, jsonl")
    if not corpus_paths:
        raise ValueError("at least one corpus path is required")
    if not benchmark_paths:
        raise ValueError("at least one benchmark path is required")

    hasher = _NgramHasher(ngram_size)
    benchmark_index: dict[int, set[int]] = {}
    benchmark_documents: list[tuple[int, int]] = []
    benchmark_stats: list[dict[str, Any]] = []

    for file_index, raw_path in enumerate(benchmark_paths):
        path = Path(raw_path)
        resolved_format = _resolve_input_format(path, input_format)
        documents = 0
        indexed_documents = 0
        ngrams = 0
        for document in _iter_documents(path, resolved_format, json_field, chunk_size, encoding):
            documents += 1
            words = _normalized_words(document.text)
            if len(words) < ngram_size:
                continue
            unique_hashes: set[int] = set()
            document_ngrams = 0
            for value, _ in hasher.hashes(words):
                unique_hashes.add(value)
                document_ngrams += 1
            document_id = len(benchmark_documents)
            benchmark_documents.append((file_index, document.index))
            for value in unique_hashes:
                benchmark_index.setdefault(value, set()).add(document_id)
            indexed_documents += 1
            ngrams += document_ngrams
        benchmark_stats.append(
            {
                "path": str(path),
                "format": resolved_format,
                "documents": documents,
                "indexed_documents": indexed_documents,
                "ngrams": ngrams,
                "matched_documents": 0,
                "contamination_fraction": 0.0,
            }
        )

    matched_document_ids: set[int] = set()
    matched_hashes: set[int] = set()
    matched_ngram_occurrences = 0
    contaminated_corpus_documents = 0
    corpus_documents = 0
    corpus_words = 0
    corpus_ngrams = 0
    corpus_stats: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    example_hashes: set[int] = set()

    for raw_path in corpus_paths:
        path = Path(raw_path)
        resolved_format = _resolve_input_format(path, input_format)
        documents = 0
        words_seen = 0
        ngrams = 0
        file_matched_documents = 0
        file_matched_ngrams = 0
        for document in _iter_documents(path, resolved_format, json_field, chunk_size, encoding):
            documents += 1
            words = _normalized_words(document.text)
            words_seen += len(words)
            if len(words) < ngram_size:
                continue
            document_matched_hashes: set[int] = set()
            for value, end in hasher.hashes(words):
                ngrams += 1
                if value not in benchmark_index:
                    continue
                matched_ngram_occurrences += 1
                file_matched_ngrams += 1
                matched_hashes.add(value)
                document_matched_hashes.add(value)
                matched_document_ids.update(benchmark_index[value])
                if len(examples) < max_examples and value not in example_hashes:
                    example_hashes.add(value)
                    ngram_text = " ".join(words[end - ngram_size + 1 : end + 1])
                    references = []
                    for document_id in sorted(benchmark_index[value])[:5]:
                        file_index, document_index = benchmark_documents[document_id]
                        references.append(
                            {
                                "path": benchmark_stats[file_index]["path"],
                                "document_index": document_index,
                            }
                        )
                    examples.append(
                        {
                            "corpus": {
                                "path": str(path),
                                "document_index": document.index,
                                "offset": document.offset,
                                "line": document.line,
                            },
                            "ngram": ngram_text[:500],
                            "benchmark_documents": references,
                        }
                    )
            if document_matched_hashes:
                contaminated_corpus_documents += 1
                file_matched_documents += 1
        corpus_documents += documents
        corpus_words += words_seen
        corpus_ngrams += ngrams
        corpus_stats.append(
            {
                "path": str(path),
                "format": resolved_format,
                "documents": documents,
                "words": words_seen,
                "ngrams": ngrams,
                "documents_with_overlap": file_matched_documents,
                "matched_ngrams": file_matched_ngrams,
            }
        )

    for document_id in matched_document_ids:
        file_index, _ = benchmark_documents[document_id]
        benchmark_stats[file_index]["matched_documents"] += 1
    for stats in benchmark_stats:
        documents = stats["documents"]
        if documents:
            stats["contamination_fraction"] = stats["matched_documents"] / documents

    benchmark_documents_total = sum(stats["documents"] for stats in benchmark_stats)
    contamination_fraction = (
        len(matched_document_ids) / benchmark_documents_total if benchmark_documents_total else 0.0
    )
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "ngram_size": ngram_size,
            "format": input_format,
            "json_field": json_field,
            "chunk_size": chunk_size,
            "encoding": encoding,
            "max_examples": max_examples,
        },
        "benchmarks": benchmark_stats,
        "corpus": corpus_stats,
        "summary": {
            "benchmark_documents": benchmark_documents_total,
            "benchmark_indexed_documents": len(benchmark_documents),
            "benchmark_unique_ngrams": len(benchmark_index),
            "benchmark_documents_with_overlap": len(matched_document_ids),
            "benchmark_contamination_fraction": contamination_fraction,
            "corpus_documents": corpus_documents,
            "corpus_words": corpus_words,
            "corpus_ngrams": corpus_ngrams,
            "corpus_documents_with_overlap": contaminated_corpus_documents,
            "matched_ngram_occurrences": matched_ngram_occurrences,
            "unique_matched_ngrams": len(matched_hashes),
        },
        "examples": examples,
        "limitations": [
            "Word n-gram overlap is a heuristic: translation, paraphrasing, changed "
            "tokenization or obfuscation can evade it.",
            "N-gram hashes are 64-bit; collisions are unlikely but not impossible.",
            "Text without spaces (for example CJK) is treated as one word per run of "
            "word characters; use a tokenizer-aware checker for such benchmarks.",
            "A match is not proof of intentional contamination; inspect the examples.",
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect benchmark contamination with word n-gram overlap"
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="PATH",
        help="Corpus text/JSONL file; repeat for multiple files",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        dest="benchmarks",
        metavar="PATH",
        help="Benchmark text/JSONL file; repeat for multiple files",
    )
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument(
        "--format",
        choices=["auto", "text", "jsonl"],
        default="auto",
        dest="input_format",
        help="Input format for all files (default: detect by extension)",
    )
    parser.add_argument("--json-field", default="text")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument(
        "--fail-on-match",
        action="store_true",
        help="Exit with status 1 when any benchmark overlap is found",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = check_contamination(
            args.corpus,
            args.benchmarks,
            ngram_size=args.ngram_size,
            input_format=args.input_format,
            json_field=args.json_field,
            chunk_size=args.chunk_size,
            encoding=args.encoding,
            max_examples=args.max_examples,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"contamination report -> {output} | "
        f"benchmark docs with overlap: {summary['benchmark_documents_with_overlap']}/"
        f"{summary['benchmark_documents']} | "
        f"matched n-grams: {summary['matched_ngram_occurrences']}"
    )
    if args.fail_on_match and summary["matched_ngram_occurrences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
