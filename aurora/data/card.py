from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aurora.data.version import read_manifest, verify_manifest_files


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _languages(languages: Sequence[str]) -> str:
    return ", ".join(languages) if languages else "Not specified"


def render_data_card(
    manifest: Mapping[str, Any],
    *,
    name: str = "AURORA dataset",
    description: str = "",
    languages: Sequence[str] = (),
    license_name: str = "",
) -> str:
    version = str(manifest["dataset_version"])
    tokenizer = manifest["tokenizer"]
    inputs = manifest["inputs"]
    outputs = manifest["outputs"]
    if (
        not isinstance(tokenizer, dict)
        or not isinstance(inputs, list)
        or not isinstance(outputs, dict)
    ):
        raise ValueError("Manifest has an unsupported structure")

    lines = [
        f"# Data card: {name}",
        "",
        "## Overview",
        "",
        f"- Dataset version: `{version}`",
        f"- Manifest format version: `{manifest.get('format_version')}`",
        f"- Created (UTC): `{manifest.get('created_utc')}`",
        f"- Languages: {_languages(languages)}",
        f"- License: {license_name or 'Not specified'}",
        "",
        description or "Description not provided.",
        "",
        "## Splits",
        "",
        "| Split | Tokens | Bytes | SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for split_name in sorted(outputs):
        entry = outputs[split_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid output entry: {split_name}")
        lines.append(
            f"| `{split_name}` | {int(entry['tokens']):,} | {int(entry['bytes']):,} | "
            f"`{_cell(entry['sha256'])}` |"
        )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Path | Tokens | Bytes | SHA-256 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for source in inputs:
        if not isinstance(source, dict):
            raise ValueError("Invalid input entry")
        lines.append(
            f"| `{_cell(source['path'])}` | {int(source['tokens']):,} | "
            f"{int(source['bytes']):,} | `{_cell(source['sha256'])}` |"
        )

    lines.extend(
        [
            "",
            "## Preparation",
            "",
            f"- Split mode: `{manifest.get('split_mode')}`",
            f"- Validation fraction: `{manifest.get('val_fraction')}`",
            f"- Test fraction: `{manifest.get('test_fraction')}`",
            f"- Encoding: `{manifest.get('encoding')}`",
            f"- Token dtype: `{manifest.get('dtype')}`",
            f"- Chunk size: `{manifest.get('chunk_size_bytes')}` bytes",
            "",
            "| Tokenizer field | Value |",
            "| --- | --- |",
            f"| Kind | `{_cell(tokenizer.get('kind'))}` |",
            f"| Vocabulary size | {_cell(tokenizer.get('vocab_size'))} |",
            f"| Model | `{_cell(tokenizer.get('model') or 'none')}` |",
            f"| Model SHA-256 | `{_cell(tokenizer.get('model_sha256') or 'none')}` |",
            "",
            "## Integrity and limitations",
            "",
            "- The dataset version is a deterministic SHA-256 over source hashes, split hashes,",
            "  preparation settings and tokenizer identity.",
            "- The manifest does not prove corpus quality, licensing, deduplication, PII removal",
            "  or benchmark contamination checks. Those checks must be documented separately.",
            "- Re-run `prepare.py` after changing inputs, tokenizer, split settings or dtype; the",
            "  dataset version changes when the produced token content changes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a data card from a manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--output", default="", help="Output Markdown path (default: stdout)")
    parser.add_argument("--name", default="AURORA dataset")
    parser.add_argument("--description", default="")
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--license", dest="license_name", default="")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify split file sizes and checksums before writing the card",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
        if args.verify:
            verify_manifest_files(manifest, Path(args.manifest).parent)
        card = render_data_card(
            manifest,
            name=args.name,
            description=args.description,
            languages=args.language,
            license_name=args.license_name,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card, encoding="utf-8")
        print(f"data card -> {output}")
    else:
        print(card, end="")


if __name__ == "__main__":
    main()
