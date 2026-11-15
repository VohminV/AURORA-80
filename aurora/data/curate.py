from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aurora.data.prepare import FileFingerprint, iter_text_chunks, iter_text_documents
from aurora.data.version import sha256_file
from aurora.provenance import build_run_metadata, utc_now

CURATION_FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 1024 * 1024
NEAR_DUPLICATE_BANDS = 8
NEAR_DUPLICATE_BAND_BITS = 8

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ipv4", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("card", re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")),
)


@dataclass(frozen=True)
class CurationConfig:
    min_chars: int = 200
    max_chars: int = 1_000_000
    min_words: int = 30
    min_letter_ratio: float = 0.5
    max_digit_ratio: float = 0.25
    max_symbol_ratio: float = 0.3
    max_duplicate_line_ratio: float = 0.3
    max_repeated_char_run: int = 50
    near_dedup: bool = True
    near_dedup_threshold: int = 3
    pii_action: str = "drop"
    normalize_unicode: bool = True

    def __post_init__(self) -> None:
        if self.min_chars < 0 or self.max_chars <= self.min_chars:
            raise ValueError("max_chars must be greater than min_chars")
        if self.min_words < 0:
            raise ValueError("min_words must be non-negative")
        for name in (
            "min_letter_ratio",
            "max_digit_ratio",
            "max_symbol_ratio",
            "max_duplicate_line_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_repeated_char_run < 1:
            raise ValueError("max_repeated_char_run must be positive")
        if not 0 <= self.near_dedup_threshold < NEAR_DUPLICATE_BANDS:
            raise ValueError("near_dedup_threshold must be in [0, 7]")
        if self.pii_action not in {"drop", "redact", "keep"}:
            raise ValueError("pii_action must be one of drop, redact, keep")


@dataclass
class CurationCounts:
    input_records: int = 0
    invalid_records: int = 0
    kept_documents: int = 0
    kept_characters: int = 0
    dropped_empty: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_too_few_words: int = 0
    dropped_low_letter_ratio: int = 0
    dropped_high_digit_ratio: int = 0
    dropped_high_symbol_ratio: int = 0
    dropped_control_characters: int = 0
    dropped_repeated_characters: int = 0
    dropped_duplicate_lines: int = 0
    dropped_exact_duplicate: int = 0
    dropped_near_duplicate: int = 0
    dropped_pii: int = 0
    redacted_documents: int = 0
    scripts: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        setattr(self, f"dropped_{reason}", getattr(self, f"dropped_{reason}") + 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        reasons = {
            key.removeprefix("dropped_"): value
            for key, value in payload.items()
            if key.startswith("dropped_") and value
        }
        payload["dropped_total"] = sum(reasons.values())
        payload["drop_reasons"] = reasons
        return payload


@dataclass(frozen=True)
class CurationDocument:
    text: str
    source: str
    line: int | None = None


def _normalize_text(text: str, normalize_unicode: bool) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _has_control_characters(text: str) -> bool:
    return any(unicodedata.category(char) == "Cc" and char not in {"\n", "\t"} for char in text)


def _max_repeated_run(text: str) -> int:
    longest = 1
    current = 1
    for index in range(1, len(text)):
        if text[index] == text[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _duplicate_line_ratio(text: str) -> tuple[float, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0, len(lines)
    return 1.0 - len(set(lines)) / len(lines), len(lines)


def _quality_reason(text: str, config: CurationConfig) -> str | None:
    if not text:
        return "empty"
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if len(text.split()) < config.min_words:
        return "too_few_words"
    characters = len(text)
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    symbols = sum(not char.isalnum() and not char.isspace() for char in text)
    if letters / characters < config.min_letter_ratio:
        return "low_letter_ratio"
    if digits / characters > config.max_digit_ratio:
        return "high_digit_ratio"
    if symbols / characters > config.max_symbol_ratio:
        return "high_symbol_ratio"
    if _has_control_characters(text):
        return "control_characters"
    if _max_repeated_run(text) > config.max_repeated_char_run:
        return "repeated_characters"
    duplicate_ratio, line_count = _duplicate_line_ratio(text)
    if line_count >= 4 and duplicate_ratio > config.max_duplicate_line_ratio:
        return "duplicate_lines"
    return None


def _find_pii(text: str) -> list[str]:
    return [name for name, pattern in PII_PATTERNS if pattern.search(text)]


def _redact_pii(text: str) -> str:
    for name, pattern in PII_PATTERNS:
        text = pattern.sub(f"[{name.upper()}]", text)
    return text


def _script_of(char: str) -> str:
    name = unicodedata.name(char, "")
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
        return "cjk"
    if name.startswith("LATIN"):
        return "latin"
    return "other"


def _script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        if char.isalpha():
            counts[_script_of(char)] += 1
    return counts


def _dedup_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingles(text: str) -> list[str]:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 3:
        return []
    return [" ".join(words[index : index + 3]) for index in range(len(words) - 2)]


def _simhash(text: str) -> int | None:
    shingles = _shingles(text)
    if not shingles:
        return None
    vector = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            signature |= 1 << bit
    return signature


class NearDuplicateIndex:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.signatures: list[int] = []
        self.buckets: list[dict[int, list[int]]] = [{} for _ in range(NEAR_DUPLICATE_BANDS)]

    @staticmethod
    def _bands(signature: int) -> list[int]:
        mask = (1 << NEAR_DUPLICATE_BAND_BITS) - 1
        return [
            (signature >> (band * NEAR_DUPLICATE_BAND_BITS)) & mask
            for band in range(NEAR_DUPLICATE_BANDS)
        ]

    def is_duplicate(self, signature: int) -> bool:
        candidates: set[int] = set()
        for band, value in enumerate(self._bands(signature)):
            candidates.update(self.buckets[band].get(value, ()))
        return any(
            (self.signatures[index] ^ signature).bit_count() <= self.threshold
            for index in candidates
        )

    def add(self, signature: int) -> None:
        index = len(self.signatures)
        self.signatures.append(signature)
        for band, value in enumerate(self._bands(signature)):
            self.buckets[band].setdefault(value, []).append(index)


def _iter_text_documents(
    path: Path,
    chunk_size: int,
    encoding: str,
    fingerprint: FileFingerprint,
) -> Iterator[CurationDocument]:
    for text, _, _ in iter_text_documents(path, chunk_size, encoding, fingerprint):
        yield CurationDocument(text=text, source=str(path))


def _iter_jsonl_documents(
    path: Path,
    chunk_size: int,
    encoding: str,
    json_field: str,
    fingerprint: FileFingerprint,
) -> Iterator[CurationDocument | None]:
    line_number = 0
    for chunk in iter_text_chunks(path, chunk_size, encoding, fingerprint):
        for line in chunk.splitlines():
            line_number += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            if not isinstance(record, dict) or not isinstance(record.get(json_field), str):
                yield None
                continue
            yield CurationDocument(
                text=record[json_field],
                source=str(path),
                line=line_number,
            )


def _resolve_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


def _preview(text: str, config: CurationConfig) -> str | None:
    if config.pii_action == "drop":
        return None
    preview = text[:200].replace("\n", " ")
    if config.pii_action == "redact":
        preview = _redact_pii(preview)
    return preview


def _curation_version(
    config: CurationConfig,
    inputs: Sequence[dict[str, Any]],
    output: dict[str, Any],
) -> str:
    payload = {
        "format_version": CURATION_FORMAT_VERSION,
        "config": asdict(config),
        "inputs": [{"bytes": entry["bytes"], "sha256": entry["sha256"]} for entry in inputs],
        "output": {"bytes": output["bytes"], "sha256": output["sha256"]},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def curate(
    inputs: Sequence[str | Path],
    output_path: str | Path,
    *,
    config: CurationConfig | None = None,
    report_path: str | Path | None = None,
    rejects_path: str | Path | None = None,
    input_format: str = "auto",
    json_field: str = "text",
    encoding: str = "utf-8",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_documents: int = 0,
) -> dict[str, Any]:
    """Curate text/JSONL documents and write a deterministic plain-text corpus."""
    config = config or CurationConfig()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not json_field:
        raise ValueError("json_field must not be empty")
    if max_documents < 0:
        raise ValueError("max_documents must be non-negative")
    if not inputs:
        raise ValueError("At least one input file is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    rejects = Path(rejects_path) if rejects_path is not None else None
    temporary_rejects = rejects.with_name(rejects.name + ".tmp") if rejects is not None else None
    if temporary_rejects is not None:
        temporary_rejects.parent.mkdir(parents=True, exist_ok=True)

    counts = CurationCounts()
    scripts: Counter[str] = Counter()
    exact_seen: set[str] = set()
    near_index = NearDuplicateIndex(config.near_dedup_threshold) if config.near_dedup else None
    input_entries: list[dict[str, Any]] = []
    output_hasher = hashlib.sha256()
    output_bytes = 0
    reject_handle = (
        temporary_rejects.open("w", encoding="utf-8", newline="\n")
        if temporary_rejects is not None
        else None
    )
    completed = False
    try:
        with temporary_output.open("w", encoding="utf-8", newline="\n") as output_handle:
            for raw_path in inputs:
                path = Path(raw_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Input file not found: {path}")
                fingerprint = FileFingerprint()
                resolved_format = _resolve_input_format(path, input_format)
                if resolved_format == "text":
                    documents: Iterator[CurationDocument | None] = _iter_text_documents(
                        path, chunk_size, encoding, fingerprint
                    )
                elif resolved_format == "jsonl":
                    documents = _iter_jsonl_documents(
                        path, chunk_size, encoding, json_field, fingerprint
                    )
                else:
                    raise ValueError(f"Unsupported input format: {resolved_format}")

                file_records = 0
                file_invalid = 0
                truncated = False
                for document in documents:
                    if max_documents and counts.kept_documents >= max_documents:
                        truncated = True
                        break
                    if document is None:
                        counts.input_records += 1
                        counts.invalid_records += 1
                        file_records += 1
                        file_invalid += 1
                        continue
                    counts.input_records += 1
                    file_records += 1
                    text = _normalize_text(document.text, config.normalize_unicode)
                    pii = _find_pii(text)
                    redacted = False
                    if pii and config.pii_action == "drop":
                        counts.record_drop("pii")
                        _write_reject(
                            reject_handle,
                            document,
                            counts.input_records,
                            "pii",
                            config,
                            text,
                            extra={"pii": pii},
                        )
                        continue
                    if pii and config.pii_action == "redact":
                        text = _redact_pii(text)
                        counts.redacted_documents += 1
                        redacted = True

                    reason = _quality_reason(text, config)
                    if reason is not None:
                        counts.record_drop(reason)
                        _write_reject(
                            reject_handle,
                            document,
                            counts.input_records,
                            reason,
                            config,
                            text,
                            extra={"pii": pii, "redacted": redacted},
                        )
                        continue

                    dedup_key = _dedup_key(text)
                    if dedup_key in exact_seen:
                        counts.record_drop("exact_duplicate")
                        _write_reject(
                            reject_handle,
                            document,
                            counts.input_records,
                            "exact_duplicate",
                            config,
                            text,
                            extra={"pii": pii, "redacted": redacted},
                        )
                        continue
                    signature = _simhash(text) if near_index is not None else None
                    if (
                        signature is not None
                        and near_index is not None
                        and near_index.is_duplicate(signature)
                    ):
                        counts.record_drop("near_duplicate")
                        _write_reject(
                            reject_handle,
                            document,
                            counts.input_records,
                            "near_duplicate",
                            config,
                            text,
                            extra={"pii": pii, "redacted": redacted},
                        )
                        continue

                    exact_seen.add(dedup_key)
                    if signature is not None and near_index is not None:
                        near_index.add(signature)

                    separator = "" if counts.kept_documents == 0 else "\n\n"
                    payload = separator + text + "\n"
                    encoded = payload.encode("utf-8")
                    output_handle.write(payload)
                    output_hasher.update(encoded)
                    output_bytes += len(encoded)
                    counts.kept_documents += 1
                    counts.kept_characters += len(text)
                    scripts.update(_script_counts(text))

                if truncated:
                    fingerprint.bytes = path.stat().st_size
                    fingerprint.sha256 = sha256_file(path)
                input_entries.append(
                    {
                        "path": str(path),
                        "format": resolved_format,
                        "bytes": fingerprint.bytes,
                        "sha256": fingerprint.sha256,
                        "records": file_records,
                        "invalid_records": file_invalid,
                        "truncated": truncated,
                    }
                )
                if max_documents and counts.kept_documents >= max_documents:
                    break
        if counts.kept_documents == 0:
            raise ValueError("No documents survived curation; relax the filters")
        os.replace(temporary_output, output)
        if reject_handle is not None and temporary_rejects is not None and rejects is not None:
            reject_handle.close()
            reject_handle = None
            os.replace(temporary_rejects, rejects)
        completed = True
    finally:
        if reject_handle is not None:
            reject_handle.close()
        if not completed:
            temporary_output.unlink(missing_ok=True)
            if temporary_rejects is not None:
                temporary_rejects.unlink(missing_ok=True)

    counts.scripts = dict(sorted(scripts.items()))
    output_entry = {
        "path": str(output),
        "bytes": output_bytes,
        "sha256": "sha256:" + output_hasher.hexdigest(),
        "documents": counts.kept_documents,
    }
    report: dict[str, Any] = {
        "format_version": CURATION_FORMAT_VERSION,
        "created_utc": utc_now(),
        "config": asdict(config),
        "inputs": input_entries,
        "output": output_entry,
        "counts": counts.to_dict(),
        "dataset_version": _curation_version(config, input_entries, output_entry),
        "run_metadata": build_run_metadata(),
    }
    if rejects_path is not None:
        report["rejects"] = str(rejects_path)
    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_destination.with_name(report_destination.name + ".tmp")
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, report_destination)
    return report


def _write_reject(
    handle: Any,
    document: CurationDocument,
    record_number: int,
    reason: str,
    config: CurationConfig,
    text: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    if handle is None:
        return
    entry: dict[str, Any] = {
        "record": record_number,
        "source": document.source,
        "line": document.line,
        "reason": reason,
        "characters": len(text),
        "preview": _preview(text, config),
    }
    if extra:
        entry.update(extra)
    handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate and filter a text corpus before tokenization"
    )
    parser.add_argument("--input", nargs="+", required=True, help="Input text/JSONL files")
    parser.add_argument("--output", required=True, help="Curated plain-text output")
    parser.add_argument("--report", default="", help="Optional JSON curation report")
    parser.add_argument("--rejects", default="", help="Optional JSONL of rejected records")
    parser.add_argument(
        "--input-format",
        choices=["auto", "text", "jsonl"],
        default="auto",
        help="Input format; auto detects .jsonl/.ndjson by extension",
    )
    parser.add_argument("--json-field", default="text", help="Text field for JSONL input")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--min-letter-ratio", type=float, default=0.5)
    parser.add_argument("--max-digit-ratio", type=float, default=0.25)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.3)
    parser.add_argument("--max-duplicate-line-ratio", type=float, default=0.3)
    parser.add_argument("--max-repeated-char-run", type=int, default=50)
    parser.add_argument("--no-near-dedup", action="store_true")
    parser.add_argument("--near-dedup-threshold", type=int, default=3)
    parser.add_argument(
        "--pii-action",
        choices=["drop", "redact", "keep"],
        default="drop",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-documents", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        config = CurationConfig(
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_words=args.min_words,
            min_letter_ratio=args.min_letter_ratio,
            max_digit_ratio=args.max_digit_ratio,
            max_symbol_ratio=args.max_symbol_ratio,
            max_duplicate_line_ratio=args.max_duplicate_line_ratio,
            max_repeated_char_run=args.max_repeated_char_run,
            near_dedup=not args.no_near_dedup,
            near_dedup_threshold=args.near_dedup_threshold,
            pii_action=args.pii_action,
            normalize_unicode=not args.no_normalize,
        )
        report = curate(
            args.input,
            args.output,
            config=config,
            report_path=args.report or None,
            rejects_path=args.rejects or None,
            input_format=args.input_format,
            json_field=args.json_field,
            encoding=args.encoding,
            chunk_size=args.chunk_size,
            max_documents=args.max_documents,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    counts = report["counts"]
    print(
        f"kept {counts['kept_documents']:,}/{counts['input_records']:,} documents "
        f"({counts['kept_characters']:,} chars) -> {report['output']['path']}"
    )
    print(f"dataset_version {report['dataset_version']}")
    if args.report:
        print(f"report -> {args.report}")
    if args.rejects:
        print(f"rejects -> {args.rejects}")


if __name__ == "__main__":
    main()
