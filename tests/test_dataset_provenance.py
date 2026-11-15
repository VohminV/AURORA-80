from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aurora.data.prepare import main as prepare_main
from aurora.data.provenance import (
    collect_dataset_sources,
    compute_dataset_fingerprint,
    describe_dataset_sources,
    resolve_dataset_source,
)
from aurora.data.version import read_manifest


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


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=512, dtype=np.uint16).tofile(path)


def test_sources_are_linked_to_manifest_versions(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Dataset provenance corpus. " * 40, "linked")
    manifest = read_manifest(manifest_path)
    output_dir = manifest_path.parent
    sources = collect_dataset_sources(
        [output_dir / "train.bin"],
        output_dir / "val.bin",
        expected_dtype="uint16",
    )

    assert [source.role for source in sources] == ["train", "val"]
    assert [source.split for source in sources] == ["train", "val"]
    assert all(source.dataset_version == manifest["dataset_version"] for source in sources)
    assert all(source.verified for source in sources)
    assert all(source.manifest_path == str(manifest_path.resolve()) for source in sources)

    description = describe_dataset_sources(sources)
    assert description["version"] == manifest["dataset_version"]
    assert description["versions"] == [manifest["dataset_version"]]
    assert description["manifest_linked"] is True
    assert description["fingerprint"] == compute_dataset_fingerprint(sources)
    assert description["fingerprint"].startswith("sha256:")


def test_mixture_fingerprint_tracks_versions_and_weights(tmp_path: Path) -> None:
    ru = _prepare(tmp_path, "Русский корпус для проверки версий. " * 40, "ru")
    en = _prepare(tmp_path, "English corpus for version checks. " * 40, "en")
    sources = collect_dataset_sources(
        [ru.parent / "train.bin", en.parent / "train.bin"],
        ru.parent / "val.bin",
        train_weights=(0.75, 0.25),
    )

    assert [source.weight for source in sources[:2]] == [0.75, 0.25]
    description = describe_dataset_sources(sources)
    assert description["version"] is None
    assert len(description["versions"]) == 2
    assert description["manifest_linked"] is True

    reweighted = collect_dataset_sources(
        [ru.parent / "train.bin", en.parent / "train.bin"],
        ru.parent / "val.bin",
        train_weights=(0.5, 0.5),
    )
    assert compute_dataset_fingerprint(sources) != compute_dataset_fingerprint(reweighted)


def test_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Manifest tamper corpus. " * 40, "tampered")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["train"]["tokens"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="version mismatch"):
        collect_dataset_sources([manifest_path.parent / "train.bin"])


def test_corrupted_token_file_is_rejected(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Corrupted token corpus. " * 40, "corrupted")
    data_path = manifest_path.parent / "train.bin"
    data = bytearray(data_path.read_bytes())
    data[0] ^= 0xFF
    data_path.write_bytes(data)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        collect_dataset_sources([data_path])


def test_unverified_mode_uses_manifest_checksums_without_rehashing(
    tmp_path: Path,
) -> None:
    manifest_path = _prepare(tmp_path, "No rehash corpus. " * 40, "nohash")
    data_path = manifest_path.parent / "train.bin"
    data = bytearray(data_path.read_bytes())
    data[0] ^= 0xFF
    data_path.write_bytes(data)

    source = resolve_dataset_source(data_path, verify_files=False, expected_dtype="uint16")
    assert source.verified is False
    assert source.dataset_version is not None


def test_explicit_manifest_and_dtype_checks(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, "Explicit manifest corpus. " * 40, "explicit")
    data_path = manifest_path.parent / "train.bin"

    source = resolve_dataset_source(
        data_path,
        manifest_paths=[manifest_path],
        expected_dtype="uint16",
    )
    assert source.dataset_version == read_manifest(manifest_path)["dataset_version"]

    with pytest.raises(ValueError, match="dtype mismatch"):
        resolve_dataset_source(data_path, expected_dtype="uint32")

    other = _prepare(tmp_path, "Another manifest corpus. " * 40, "other")
    with pytest.raises(ValueError, match="None of the configured"):
        resolve_dataset_source(data_path, manifest_paths=[other])


def test_unversioned_raw_file_is_still_content_addressed(tmp_path: Path) -> None:
    path = tmp_path / "raw.bin"
    _write_tokens(path, 1)
    source = resolve_dataset_source(path)

    assert source.dataset_version is None
    assert source.manifest_path is None
    assert source.sha256 is not None
    assert source.verified is True
    description = describe_dataset_sources([source])
    assert description["version"] is None
    assert description["versions"] == []
    assert description["manifest_linked"] is False
