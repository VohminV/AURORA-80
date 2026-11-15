from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.data.card import main as card_main
from aurora.data.card import render_data_card
from aurora.data.prepare import main as prepare_main
from aurora.data.version import (
    compute_dataset_version,
    read_manifest,
    verify_manifest_files,
)


def _prepare(tmp_path: Path, text: str, name: str) -> Path:
    source = tmp_path / f"{name}.txt"
    source.write_text(text, encoding="utf-8")
    output_dir = tmp_path / name
    prepare_main(
        [
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--tokenizer",
            "byte",
            "--val-fraction",
            "0.2",
            "--test-fraction",
            "0.1",
        ]
    )
    return output_dir / "manifest.json"


def test_manifest_dataset_version_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    first = read_manifest(_prepare(tmp_path, "AURORA dataset version test. " * 30, "first"))
    second = read_manifest(_prepare(tmp_path, "AURORA dataset version test. " * 30, "second"))
    changed = read_manifest(_prepare(tmp_path, "Different corpus content. " * 30, "changed"))

    assert first["dataset_version"].startswith("sha256:")
    assert len(first["dataset_version"]) == len("sha256:") + 64
    assert first["dataset_version"] == second["dataset_version"]
    assert first["dataset_version"] != changed["dataset_version"]
    assert compute_dataset_version(first) == first["dataset_version"]
    assert first["tokenizer"]["model_sha256"] is None
    assert set(first["outputs"]) == {"train", "val", "test"}
    verify_manifest_files(first, tmp_path / "first")


def test_data_card_renders_and_writes_markdown(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "AURORA data card test. " * 40, "processed")
    manifest = read_manifest(manifest_path)
    card = render_data_card(
        manifest,
        name="Demo corpus",
        description="Small deterministic corpus.",
        languages=("ru", "en"),
        license_name="CC-BY-4.0",
    )
    assert "# Data card: Demo corpus" in card
    assert manifest["dataset_version"] in card
    assert "Small deterministic corpus." in card
    assert "ru, en" in card
    assert "CC-BY-4.0" in card
    assert "| `train` |" in card
    assert manifest["outputs"]["train"]["sha256"] in card

    output = tmp_path / "reports" / "DATA_CARD.md"
    card_main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--name",
            "CLI corpus",
            "--language",
            "ru",
            "--license",
            "Apache-2.0",
            "--verify",
        ]
    )
    written = output.read_text(encoding="utf-8")
    assert "# Data card: CLI corpus" in written
    assert "Apache-2.0" in written
    assert manifest["dataset_version"] in written


def test_manifest_rejects_tampered_metadata(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Tamper detection corpus. " * 30, "tampered")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["train"]["tokens"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="version mismatch"):
        read_manifest(manifest_path)


def test_manifest_file_verification_detects_corruption(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Checksum verification corpus. " * 30, "verified")
    manifest = read_manifest(manifest_path)
    val_path = tmp_path / "verified" / "val.bin"
    data = bytearray(val_path.read_bytes())
    data[0] ^= 0xFF
    val_path.write_bytes(data)

    with pytest.raises(ValueError, match="Checksum mismatch for val"):
        verify_manifest_files(manifest, tmp_path / "verified")
