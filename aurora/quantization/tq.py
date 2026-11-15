from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig
from aurora.quantization.qat import DEFAULT_SKIP_PREFIXES, TernaryLinear
from aurora.quantization.ternary import (
    TernaryTensor,
    dequantize_ternary,
    pack_ternary,
    quantize_ternary,
    unpack_ternary,
)
from aurora.utils import resolve_device

TQ_FORMAT = "AURORA-TQ"
TQ_VERSION = 1
MAX_METADATA_BYTES = 4 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_state_names(model: nn.Module) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, TernaryLinear):
            continue
        child_prefix = f"{module_name}.linear" if module_name else "linear"
        canonical_prefix = module_name
        for suffix in ("weight", "bias"):
            source = f"{child_prefix}.{suffix}"
            canonical = f"{canonical_prefix}.{suffix}" if canonical_prefix else suffix
            mapping[source] = canonical
    return mapping


def _dense_bytes(tensor: torch.Tensor) -> tuple[bytes, str, list[int]]:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C"), str(array.dtype), list(array.shape)


def _read_entry(archive: zipfile.ZipFile, name: str, expected_sha256: str) -> bytes:
    parts = Path(name).parts
    if not name.startswith("tensors/") or ".." in parts:
        raise ValueError(f"Unsafe tensor entry: {name}")
    try:
        data = archive.read(name)
    except KeyError as error:
        raise ValueError(f"Missing tensor entry: {name}") from error
    if _sha256(data) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {name}")
    return data


def save_tq(
    path: str | Path,
    model: nn.Module,
    config: ExperimentConfig,
    *,
    group_size: int = 128,
    min_features: int = 256,
    skip_prefixes: Iterable[str] = DEFAULT_SKIP_PREFIXES,
    scales_dtype: str = "float16",
    calibrate: bool = False,
    calibration_steps: int = 4,
) -> dict[str, Any]:
    """Write an AURORA-TQ archive with packed ternary and dense tensors."""
    if group_size <= 0 or min_features <= 0:
        raise ValueError("group_size and min_features must be positive")
    if scales_dtype not in {"float16", "float32"}:
        raise ValueError("scales_dtype must be float16 or float32")
    if calibration_steps < 0:
        raise ValueError("calibration_steps must be non-negative")
    effective_steps = calibration_steps if calibrate else 0

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    skip = tuple(skip_prefixes)
    qat_state = any(isinstance(module, TernaryLinear) for module in model.modules())
    canonical_names = _canonical_state_names(model)
    scale_dtype = torch.float16 if scales_dtype == "float16" else torch.float32

    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    total_parameters = 0
    ternary_parameters = 0
    dense_parameters = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, (source_name, tensor) in enumerate(model.state_dict().items()):
                if not torch.is_tensor(tensor):
                    raise TypeError(f"State entry {source_name} is not a tensor")
                name = canonical_names.get(source_name, source_name)
                if name in seen_names:
                    raise ValueError(f"Duplicate canonical tensor name: {name}")
                seen_names.add(name)
                total_parameters += tensor.numel()
                quantizable = (
                    tensor.ndim >= 2
                    and tensor.is_floating_point()
                    and min(tensor.shape[-2], tensor.shape[-1]) >= min_features
                    and not any(token in source_name for token in skip)
                )
                if quantizable:
                    quantized = quantize_ternary(
                        tensor,
                        group_size,
                        calibration_steps=effective_steps,
                    )
                    packed, numel = pack_ternary(quantized.levels)
                    scales = quantized.scales.to(scale_dtype).cpu().contiguous()
                    levels_data = packed.cpu().numpy().tobytes()
                    scales_data = scales.numpy().tobytes()
                    levels_file = f"tensors/{index:05d}.levels"
                    scales_file = f"tensors/{index:05d}.scales"
                    archive.writestr(levels_file, levels_data)
                    archive.writestr(scales_file, scales_data)
                    entries.append(
                        {
                            "name": name,
                            "source_name": source_name,
                            "kind": "ternary",
                            "shape": list(quantized.shape),
                            "numel": numel,
                            "group_size": quantized.group_size,
                            "levels_file": levels_file,
                            "levels_sha256": _sha256(levels_data),
                            "scales_file": scales_file,
                            "scales_sha256": _sha256(scales_data),
                            "scales_dtype": scales_dtype,
                        }
                    )
                    ternary_parameters += numel
                else:
                    data, dtype_name, shape = _dense_bytes(tensor)
                    data_file = f"tensors/{index:05d}.data"
                    archive.writestr(data_file, data)
                    entries.append(
                        {
                            "name": name,
                            "source_name": source_name,
                            "kind": "dense",
                            "shape": shape,
                            "dtype": dtype_name,
                            "data_file": data_file,
                            "sha256": _sha256(data),
                        }
                    )
                    dense_parameters += tensor.numel()

            metadata: dict[str, Any] = {
                "format": TQ_FORMAT,
                "format_version": TQ_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_config": config.to_dict(),
                "quantization": {
                    "qat_state": qat_state,
                    "group_size": group_size,
                    "min_features": min_features,
                    "skip_prefixes": list(skip),
                    "scales_dtype": scales_dtype,
                    "calibrated": calibrate,
                    "calibration_steps": effective_steps,
                },
                "total_parameters": total_parameters,
                "ternary_parameters": ternary_parameters,
                "dense_parameters": dense_parameters,
                "tensors": entries,
            }
            archive.writestr(
                "metadata.json",
                json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return metadata


def load_tq(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | TernaryTensor]]:
    """Load and validate an AURORA-TQ archive."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"AURORA-TQ file not found: {source}")
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Not an AURORA-TQ archive: {source}")

    with zipfile.ZipFile(source, "r") as archive:
        try:
            metadata_info = archive.getinfo("metadata.json")
        except KeyError as error:
            raise ValueError("AURORA-TQ metadata.json is missing") from error
        if metadata_info.file_size > MAX_METADATA_BYTES:
            raise ValueError("AURORA-TQ metadata is too large")
        try:
            metadata = json.loads(archive.read("metadata.json"))
        except json.JSONDecodeError as error:
            raise ValueError("AURORA-TQ metadata is not valid JSON") from error
        if not isinstance(metadata, dict):
            raise ValueError("AURORA-TQ metadata must be an object")
        if metadata.get("format") != TQ_FORMAT:
            raise ValueError("Unknown AURORA-TQ format")
        if metadata.get("format_version") != TQ_VERSION:
            raise ValueError(f"Unsupported AURORA-TQ version: {metadata.get('format_version')}")

        entries = metadata.get("tensors")
        if not isinstance(entries, list):
            raise ValueError("AURORA-TQ metadata has no tensor list")

        tensors: dict[str, torch.Tensor | TernaryTensor] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError("Invalid AURORA-TQ tensor entry")
            name = entry["name"]
            if name in tensors:
                raise ValueError(f"Duplicate tensor name: {name}")
            kind = entry.get("kind")
            if kind == "ternary":
                tensors[name] = _load_ternary_entry(archive, entry)
            elif kind == "dense":
                tensors[name] = _load_dense_entry(archive, entry)
            else:
                raise ValueError(f"Unknown tensor kind for {name}: {kind}")
        return metadata, tensors


def _parse_shape(entry: dict[str, Any], name: str) -> tuple[int, ...]:
    raw_shape = entry.get("shape")
    if not isinstance(raw_shape, list):
        raise ValueError(f"Invalid shape for {name}")
    try:
        shape = tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid shape for {name}") from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"Invalid shape for {name}")
    return shape


def _load_ternary_entry(archive: zipfile.ZipFile, entry: dict[str, Any]) -> TernaryTensor:
    name = entry["name"]
    shape = _parse_shape(entry, name)
    try:
        numel = int(entry["numel"])
        group_size = int(entry["group_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid ternary metadata for {name}") from error
    scales_dtype = entry.get("scales_dtype")
    if len(shape) < 2 or any(dimension == 0 for dimension in shape) or math.prod(shape) != numel:
        raise ValueError(f"Invalid ternary shape for {name}")
    if group_size <= 0:
        raise ValueError(f"Invalid group size for {name}")
    if scales_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported scales dtype for {name}: {scales_dtype}")
    levels_file = entry.get("levels_file")
    scales_file = entry.get("scales_file")
    if not isinstance(levels_file, str) or not isinstance(scales_file, str):
        raise ValueError(f"Invalid tensor entries for {name}")

    levels_data = _read_entry(archive, levels_file, entry["levels_sha256"])
    scales_data = _read_entry(archive, scales_file, entry["scales_sha256"])
    if len(levels_data) != math.ceil(numel / 4):
        raise ValueError(f"Invalid packed level size for {name}")

    columns = shape[-1]
    rows = numel // columns
    groups_per_row = math.ceil(columns / group_size)
    numpy_dtype = np.float16 if scales_dtype == "float16" else np.float32
    scales_array = np.frombuffer(scales_data, dtype=numpy_dtype)
    if scales_array.size != rows * groups_per_row:
        raise ValueError(f"Invalid scales size for {name}")

    packed = torch.frombuffer(bytearray(levels_data), dtype=torch.uint8).clone()
    levels = unpack_ternary(packed, numel).reshape(shape)
    scales = torch.from_numpy(scales_array.copy()).to(torch.float32)
    return TernaryTensor(
        levels=levels,
        scales=scales,
        shape=shape,
        group_size=group_size,
    )


def _load_dense_entry(archive: zipfile.ZipFile, entry: dict[str, Any]) -> torch.Tensor:
    name = entry["name"]
    shape = _parse_shape(entry, name)
    dtype = getattr(torch, str(entry.get("dtype")), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype for {name}: {entry.get('dtype')}")
    data_file = entry.get("data_file")
    if not isinstance(data_file, str):
        raise ValueError(f"Invalid tensor entry for {name}")
    data = _read_entry(archive, data_file, entry["sha256"])
    expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected:
        raise ValueError(f"Invalid dense tensor size for {name}")
    tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).clone()
    return tensor


def load_tq_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, ExperimentConfig, dict[str, Any]]:
    """Rebuild a model from an AURORA-TQ archive.

    Ternary tensors are dequantized into a regular dense model. This path is for
    compatibility and validation, not for the future memory-mapped runtime.
    """
    metadata, tensors = load_tq(path)
    config = ExperimentConfig.from_dict(metadata["experiment_config"])
    model = build_model(config.model)
    state = {
        name: dequantize_ternary(tensor) if isinstance(tensor, TernaryTensor) else tensor
        for name, tensor in tensors.items()
    }
    model.load_state_dict(state, strict=True)
    resolved = resolve_device(device) if isinstance(device, str) else device
    return model.to(resolved), config, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export or inspect AURORA-TQ files")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export a checkpoint")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--group-size", type=int, default=128)
    export_parser.add_argument("--min-features", type=int, default=256)
    export_parser.add_argument("--scales-dtype", choices=["float16", "float32"], default="float16")
    export_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Refine group scales by minimizing weight reconstruction error",
    )
    export_parser.add_argument("--calibration-steps", type=int, default=4)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect an archive")
    inspect_parser.add_argument("path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        metadata, _ = load_tq(args.path)
        summary = {
            key: metadata[key]
            for key in (
                "format",
                "format_version",
                "created_utc",
                "quantization",
                "total_parameters",
                "ternary_parameters",
                "dense_parameters",
            )
        }
        summary["tensors"] = len(metadata["tensors"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    from aurora.inference import load_model_from_checkpoint

    model, config = load_model_from_checkpoint(args.checkpoint, device="cpu")
    metadata = save_tq(
        args.output,
        model,
        config,
        group_size=args.group_size,
        min_features=args.min_features,
        scales_dtype=args.scales_dtype,
        calibrate=args.calibrate,
        calibration_steps=args.calibration_steps,
    )
    print(
        f"{args.output}: {metadata['total_parameters']:,} parameters, "
        f"{metadata['ternary_parameters']:,} ternary, "
        f"{len(metadata['tensors'])} tensors"
    )


if __name__ == "__main__":
    main()
