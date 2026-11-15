from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataset_version(manifest: Mapping[str, Any]) -> str:
    """Return a deterministic content hash for a prepared dataset manifest."""
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(inputs, list)
        or not isinstance(outputs, dict)
        or not isinstance(tokenizer, dict)
    ):
        raise ValueError("Manifest must contain inputs, outputs and tokenizer")
    if not all(isinstance(item, dict) for item in inputs):
        raise ValueError("Manifest inputs must be objects")
    if not all(isinstance(entry, dict) for entry in outputs.values()):
        raise ValueError("Manifest outputs must be objects")

    payload: dict[str, Any] = {
        "format_version": manifest.get("format_version"),
        "tokenizer": {
            "kind": tokenizer.get("kind"),
            "vocab_size": tokenizer.get("vocab_size"),
            "model_sha256": tokenizer.get("model_sha256"),
        },
        "encoding": manifest.get("encoding"),
        "dtype": manifest.get("dtype"),
        "split_mode": manifest.get("split_mode"),
        "val_fraction": manifest.get("val_fraction"),
        "test_fraction": manifest.get("test_fraction"),
        "inputs": [
            {
                "sha256": item.get("sha256"),
                "bytes": item.get("bytes"),
                "tokens": item.get("tokens"),
            }
            for item in inputs
        ],
        "outputs": {
            str(name): {
                "sha256": entry.get("sha256"),
                "bytes": entry.get("bytes"),
                "tokens": entry.get("tokens"),
            }
            for name, entry in outputs.items()
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be an object: {source}")
    version = manifest.get("dataset_version")
    if not isinstance(version, str) or not version.startswith("sha256:"):
        raise ValueError(f"Manifest has no dataset_version: {source}")
    expected = compute_dataset_version(manifest)
    if version != expected:
        raise ValueError(
            f"Dataset version mismatch for {source}: manifest has {version}, computed {expected}"
        )
    return manifest


def verify_manifest_files(manifest: Mapping[str, Any], root: str | Path) -> list[Path]:
    """Verify prepared split files against the manifest's sizes and hashes."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Manifest has no outputs")
    base = Path(root)
    verified: list[Path] = []
    for name, entry in outputs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid output entry: {name}")
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"Output {name} has no path")
        path = base / relative
        if not path.is_file():
            raise ValueError(f"Missing split file for {name}: {path}")
        expected_bytes = entry.get("bytes")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Size mismatch for {name}: expected {expected_bytes}, got {actual_bytes}"
            )
        expected_sha = entry.get("sha256")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Checksum mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        verified.append(path)
    return verified
