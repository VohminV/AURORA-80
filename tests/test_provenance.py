from __future__ import annotations

from pathlib import Path

import torch

from aurora.provenance import (
    build_run_metadata,
    environment_metadata,
    git_metadata,
    utc_now,
)


def test_git_metadata_outside_repository_is_safe(tmp_path: Path) -> None:
    metadata = git_metadata(tmp_path)
    assert metadata["available"] is False
    assert metadata["commit"] is None


def test_environment_metadata_records_cpu_runtime() -> None:
    metadata = environment_metadata(torch.device("cpu"))
    assert metadata["device"] == "cpu"
    assert metadata["torch"] == torch.__version__
    assert metadata["cuda_available"] == torch.cuda.is_available()
    assert metadata["python"]


def test_build_run_metadata_has_identity_and_provenance() -> None:
    metadata = build_run_metadata("cpu")
    assert isinstance(metadata["run_id"], str)
    assert metadata["run_id"]
    assert metadata["started_utc"]
    assert isinstance(metadata["git"], dict)
    assert metadata["environment"]["device"] == "cpu"
    assert utc_now() >= metadata["started_utc"]
