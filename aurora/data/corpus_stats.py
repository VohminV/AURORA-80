from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aurora.data.prepare import iter_text_chunks, iter_text_documents

FORMAT_VERSION = 1
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_CODE_LINE_RE = re.compile(
    r"(^\s*(def|class|import|from|return|function|const|let|var|fn|pub|package|"
    r"#include|using|select|insert|update|delete|create)\b|[{}();=<>])",
    re.IGNORECASE,
)


@dataclass
class _DomainStats:
    documents: int = 0
    characters: int = 0
    words: int = 0
    lines: int = 0
    code_lines: int = 0
    scripts: Counter[str] = field(default_factory=Counter)

    def add(self, text: str) -> None:
        self.documents += 1
        self.characters += len(text)
        normalized = unicodedata.normalize("NFKC", text)
        self.words += len(_WORD_RE.findall(normalized.casefold()))
        lines = normalized.splitlines()
        self.lines += len(lines)
        self.code_lines += sum(1 for line in lines if _CODE_LINE_RE.search(line))
        for char in normalized:
            if char.isalpha():
                self.scripts[_script_of(char)] += 1


def _script_of(char: str) -> str:
    name = unicodedata.name(char, "")
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
        return "cjk"
    if name.startswith("LATIN"):
        return "latin"
    return "other"


def _language_hint(scripts: Counter[str], code_line_fraction: float, threshold: float) -> str:
    total = sum(scripts.values())
    if total == 0:
        return "unknown"
    fractions = {name: count / total for name, count in scripts.items()}
    if code_line_fraction >= threshold and fractions.get("latin", 0.0) >= 0.5:
        return "code"
    if fractions.get("cyrillic", 0.0) >= 0.5:
        return "cyrillic"
    if fractions.get("cjk", 0.0) >= 0.5:
        return "cjk"
    if fractions.get("latin", 0.0) >= 0.5:
        return "latin"
    return "mixed"


def _resolve_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


def _iter_text(
    path: Path,
    input_format: str,
    json_field: str,
    domain_field: str,
    chunk_size: int,
    encoding: str,
    max_documents: int,
    fallback_domain: str,
) -> Iterator[tuple[str, str]]:
    resolved = _resolve_format(path, input_format)
    count = 0
    if resolved == "text":
        for text, _, _ in iter_text_documents(path, chunk_size, encoding):
            if max_documents and count >= max_documents:
                return
            count += 1
            yield text, fallback_domain
        return

    line_number = 0
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
            if max_documents and count >= max_documents:
                return
            domain = fallback_domain
            if domain_field:
                candidate = record.get(domain_field)
                if isinstance(candidate, str) and candidate.strip():
                    domain = candidate.strip()
            count += 1
            yield record[json_field], domain


def analyze_corpus(
    input_paths: Sequence[str | Path],
    *,
    input_format: str = "auto",
    json_field: str = "text",
    domain_field: str = "",
    chunk_size: int = 1024 * 1024,
    encoding: str = "utf-8",
    max_documents: int = 0,
    code_line_threshold: float = 0.15,
) -> dict[str, Any]:
    """Summarize script and domain distribution without loading the corpus into RAM."""
    if not input_paths:
        raise ValueError("at least one input path is required")
    if input_format not in {"auto", "text", "jsonl"}:
        raise ValueError("format must be one of auto, text, jsonl")
    if not json_field:
        raise ValueError("json_field must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_documents < 0:
        raise ValueError("max_documents must be non-negative")
    if not 0.0 <= code_line_threshold <= 1.0:
        raise ValueError("code_line_threshold must be in [0, 1]")

    domains: dict[str, _DomainStats] = {}
    inputs: list[dict[str, Any]] = []
    seen_documents = 0

    for raw_path in input_paths:
        remaining = max_documents - seen_documents if max_documents else 0
        if max_documents and remaining <= 0:
            break
        path = Path(raw_path)
        resolved = _resolve_format(path, input_format)
        fallback_domain = path.stem or "input"
        file_documents = 0
        for text, domain in _iter_text(
            path,
            resolved,
            json_field,
            domain_field,
            chunk_size,
            encoding,
            remaining,
            fallback_domain,
        ):
            stats = domains.setdefault(domain, _DomainStats())
            stats.add(text)
            file_documents += 1
            seen_documents += 1
        inputs.append(
            {
                "path": str(path),
                "format": resolved,
                "documents": file_documents,
            }
        )

    total_scripts: Counter[str] = Counter()
    for stats in domains.values():
        total_scripts.update(stats.scripts)
    domain_report: dict[str, dict[str, Any]] = {}
    for name, stats in sorted(domains.items()):
        script_total = sum(stats.scripts.values())
        code_fraction = stats.code_lines / stats.lines if stats.lines else 0.0
        domain_report[name] = {
            "documents": stats.documents,
            "characters": stats.characters,
            "words": stats.words,
            "lines": stats.lines,
            "code_line_fraction": code_fraction,
            "scripts": dict(sorted(stats.scripts.items())),
            "script_fractions": {
                script: count / script_total if script_total else 0.0
                for script, count in sorted(stats.scripts.items())
            },
            "language_hint": _language_hint(stats.scripts, code_fraction, code_line_threshold),
        }

    total_characters = sum(stats.characters for stats in domains.values())
    total_words = sum(stats.words for stats in domains.values())
    total_lines = sum(stats.lines for stats in domains.values())
    total_code_lines = sum(stats.code_lines for stats in domains.values())
    overall_code_fraction = total_code_lines / total_lines if total_lines else 0.0
    script_total = sum(total_scripts.values())
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "format": input_format,
            "json_field": json_field,
            "domain_field": domain_field,
            "chunk_size": chunk_size,
            "encoding": encoding,
            "max_documents": max_documents,
            "code_line_threshold": code_line_threshold,
        },
        "inputs": inputs,
        "domains": domain_report,
        "summary": {
            "documents": sum(stats.documents for stats in domains.values()),
            "characters": total_characters,
            "words": total_words,
            "lines": total_lines,
            "code_line_fraction": overall_code_fraction,
            "scripts": dict(sorted(total_scripts.items())),
            "script_fractions": {
                script: count / script_total if script_total else 0.0
                for script, count in sorted(total_scripts.items())
            },
            "language_hint": _language_hint(
                total_scripts, overall_code_fraction, code_line_threshold
            ),
        },
        "limitations": [
            "Script ratios and language hints are heuristic; they are not a language "
            "identification model.",
            "The code heuristic reacts to punctuation and common keywords and can "
            "misclassify prose or markup.",
            "Token counts are not reported because tokenization is corpus-specific; "
            "use aurora-tokenizer-stats with the selected tokenizer.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report script, language-hint and domain distribution for a corpus"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="PATH",
        help="Corpus text/JSONL file; repeat for multiple files",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "text", "jsonl"],
        default="auto",
        dest="input_format",
        help="Input format for all files (default: detect by extension)",
    )
    parser.add_argument("--json-field", default="text")
    parser.add_argument(
        "--domain-field",
        default="",
        help="Optional JSONL field used as the domain label (default: file name)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--code-line-threshold", type=float, default=0.15)
    parser.add_argument("--output", required=True, help="JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_corpus(
            args.input,
            input_format=args.input_format,
            json_field=args.json_field,
            domain_field=args.domain_field,
            chunk_size=args.chunk_size,
            encoding=args.encoding,
            max_documents=args.max_documents,
            code_line_threshold=args.code_line_threshold,
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
        f"corpus stats -> {output} | documents: {summary['documents']:,} | "
        f"words: {summary['words']:,} | hint: {summary['language_hint']} | "
        f"domains: {len(report['domains'])}"
    )


if __name__ == "__main__":
    main()
