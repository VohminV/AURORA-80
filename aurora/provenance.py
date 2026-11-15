from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata(root: str | Path | None = None) -> dict[str, Any]:
    """Return git revision details without failing outside a repository."""
    requested = Path(root) if root is not None else Path.cwd()
    metadata: dict[str, Any] = {
        "available": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "dirty_files": None,
        "root": None,
    }
    repository = _git(requested, ["rev-parse", "--show-toplevel"])
    if repository is None:
        return metadata

    status = _git(requested, ["status", "--porcelain", "--untracked-files=no"])
    metadata.update(
        {
            "available": True,
            "commit": _git(requested, ["rev-parse", "HEAD"]),
            "branch": _git(requested, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(status) if status is not None else None,
            "dirty_files": len(status.splitlines()) if status is not None else None,
            "root": repository,
        }
    )
    return metadata


def environment_metadata(device: torch.device | str | None = None) -> dict[str, Any]:
    """Capture the runtime environment needed to reproduce a run."""
    resolved_device = str(device) if device is not None else None
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "numpy": str(np.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": resolved_device,
        "gpu": None,
    }
    if resolved_device is not None:
        parsed = torch.device(resolved_device)
        if parsed.type == "cuda" and torch.cuda.is_available():
            index = parsed.index if parsed.index is not None else 0
            properties = torch.cuda.get_device_properties(index)
            metadata["gpu"] = {
                "name": properties.name,
                "total_memory_mb": properties.total_memory / (1024.0 * 1024.0),
                "capability": f"{properties.major}.{properties.minor}",
            }
    return metadata


def build_run_metadata(device: torch.device | str | None = None) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "started_utc": utc_now(),
        "git": git_metadata(),
        "environment": environment_metadata(device),
    }
