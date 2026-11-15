from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aurora.data.version import read_manifest, sha256_file


@dataclass(frozen=True)
class DatasetSource:
    role: str
    path: str
    bytes: int
    sha256: str | None
    dataset_version: str | None
    manifest_path: str | None
    split: str | None
    tokens: int | None
    dtype: str | None
    weight: float | None
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "dataset_version": self.dataset_version,
            "manifest_path": self.manifest_path,
            "split": self.split,
            "tokens": self.tokens,
            "dtype": self.dtype,
            "weight": self.weight,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class _ManifestMatch:
    manifest_path: Path
    manifest: dict[str, Any]
    split: str
    entry: dict[str, Any]


def _output_path(manifest_path: Path, entry: Mapping[str, Any]) -> Path | None:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        return None
    return (manifest_path.parent / relative).resolve()


def _match_manifest(manifest_path: Path, target: Path) -> _ManifestMatch | None:
    manifest = read_manifest(manifest_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Manifest has no outputs: {manifest_path}")
    for split, entry in outputs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid output entry {split!r} in {manifest_path}")
        if _output_path(manifest_path, entry) != target:
            continue
        return _ManifestMatch(
            manifest_path=manifest_path,
            manifest=manifest,
            split=str(split),
            entry=dict(entry),
        )
    return None


def _find_manifest(
    target: Path,
    manifest_paths: Sequence[str | Path],
) -> tuple[_ManifestMatch | None, bool]:
    """Return a matching manifest and whether explicit manifests were requested."""
    if manifest_paths:
        matches: list[_ManifestMatch] = []
        for raw_manifest in manifest_paths:
            candidate = Path(raw_manifest)
            if not candidate.is_file():
                raise FileNotFoundError(f"Dataset manifest not found: {candidate}")
            match = _match_manifest(candidate, target)
            if match is not None:
                matches.append(match)
        if not matches:
            raise ValueError(f"None of the configured dataset manifests describes {target}")
        if len(matches) > 1:
            paths = ", ".join(str(match.manifest_path) for match in matches)
            raise ValueError(f"Dataset file {target} is described by multiple manifests: {paths}")
        return matches[0], True

    candidate = target.parent / "manifest.json"
    if not candidate.is_file():
        return None, False
    match = _match_manifest(candidate, target)
    if match is None:
        raise ValueError(f"Manifest {candidate} does not describe dataset file {target}")
    return match, False


def resolve_dataset_source(
    path: str | Path,
    *,
    role: str = "dataset",
    weight: float | None = None,
    manifest_paths: Sequence[str | Path] = (),
    verify_files: bool = True,
    expected_dtype: str | None = None,
) -> DatasetSource:
    """Resolve a token file to its manifest, version and checksum."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Token file not found: {source}")
    resolved = source.resolve()
    actual_bytes = resolved.stat().st_size
    match, _explicit = _find_manifest(resolved, manifest_paths)

    if match is not None:
        manifest_dtype = match.manifest.get("dtype")
        if (
            expected_dtype is not None
            and isinstance(manifest_dtype, str)
            and manifest_dtype != expected_dtype
        ):
            raise ValueError(
                f"Dataset dtype mismatch for {resolved}: manifest has {manifest_dtype}, "
                f"config expects {expected_dtype}"
            )
        expected_bytes = match.entry.get("bytes")
        if expected_bytes != actual_bytes:
            raise ValueError(
                f"Size mismatch for {resolved}: manifest has {expected_bytes}, "
                f"file has {actual_bytes}"
            )
        expected_sha = match.entry.get("sha256")
        if not isinstance(expected_sha, str):
            raise ValueError(f"Manifest output for {resolved} has no sha256")
        verified = False
        if verify_files:
            actual_sha = sha256_file(resolved)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {resolved}: manifest has {expected_sha}, "
                    f"file has {actual_sha}"
                )
            verified = True
        tokens = match.entry.get("tokens")
        version = match.manifest.get("dataset_version")
        if not isinstance(version, str):
            raise ValueError(f"Manifest for {resolved} has no dataset_version")
        return DatasetSource(
            role=role,
            path=str(resolved),
            bytes=actual_bytes,
            sha256=expected_sha,
            dataset_version=version,
            manifest_path=str(match.manifest_path.resolve()),
            split=match.split,
            tokens=int(tokens) if isinstance(tokens, int) else None,
            dtype=str(manifest_dtype) if isinstance(manifest_dtype, str) else None,
            weight=weight,
            verified=verified,
        )

    return DatasetSource(
        role=role,
        path=str(resolved),
        bytes=actual_bytes,
        sha256=sha256_file(resolved),
        dataset_version=None,
        manifest_path=None,
        split=None,
        tokens=None,
        dtype=None,
        weight=weight,
        verified=True,
    )


def collect_dataset_sources(
    train_paths: Sequence[str | Path],
    val_path: str | Path | None = None,
    *,
    train_weights: Sequence[float] = (),
    manifest_paths: Sequence[str | Path] = (),
    verify_files: bool = True,
    expected_dtype: str | None = None,
) -> list[DatasetSource]:
    """Describe train/val/mixture files and verify their manifests."""
    paths = [str(path) for path in train_paths if str(path)]
    if train_weights and len(train_weights) != len(paths):
        raise ValueError("train_paths and train_weights must have the same length")

    sources: list[DatasetSource] = []
    for index, path in enumerate(paths):
        weight = float(train_weights[index]) if train_weights else None
        sources.append(
            resolve_dataset_source(
                path,
                role="train",
                weight=weight,
                manifest_paths=manifest_paths,
                verify_files=verify_files,
                expected_dtype=expected_dtype,
            )
        )
    if val_path:
        sources.append(
            resolve_dataset_source(
                val_path,
                role="val",
                manifest_paths=manifest_paths,
                verify_files=verify_files,
                expected_dtype=expected_dtype,
            )
        )
    return sources


def compute_dataset_fingerprint(sources: Sequence[DatasetSource]) -> str | None:
    if not sources:
        return None
    payload = [
        {
            "role": source.role,
            "weight": source.weight,
            "sha256": source.sha256,
            "bytes": source.bytes,
            "dataset_version": source.dataset_version,
            "split": source.split,
        }
        for source in sources
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def describe_dataset_sources(sources: Sequence[DatasetSource]) -> dict[str, Any]:
    versions = sorted(
        {source.dataset_version for source in sources if source.dataset_version is not None}
    )
    return {
        "version": versions[0] if len(versions) == 1 else None,
        "versions": versions,
        "fingerprint": compute_dataset_fingerprint(sources),
        "verified": all(source.verified for source in sources) if sources else None,
        "manifest_linked": all(source.manifest_path is not None for source in sources)
        if sources
        else None,
        "sources": [source.to_dict() for source in sources],
    }
