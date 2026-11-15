from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig
from aurora.quantization import convert_to_ternary, dequantize_ternary
from aurora.quantization.ternary import TernaryTensor
from aurora.quantization.tq import load_tq, load_tq_model, main, save_tq


def _rewrite_archive(path: Path, mutate: Any) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    mutate(entries)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_tq_roundtrip_qat_model(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    torch.manual_seed(0)
    model = build_model(tiny_experiment_config.model)
    convert_to_ternary(model, group_size=16, min_features=16)
    model.eval()
    path = tmp_path / "model.aurora-tq"
    metadata = save_tq(
        path,
        model,
        tiny_experiment_config,
        group_size=16,
        min_features=16,
        scales_dtype="float32",
    )
    assert metadata["format"] == "AURORA-TQ"
    assert metadata["format_version"] == 1
    assert metadata["quantization"]["qat_state"] is True
    assert metadata["ternary_parameters"] > 0
    assert path.is_file()

    loaded_model, loaded_config, loaded_metadata = load_tq_model(path)
    loaded_model.eval()
    assert loaded_metadata["format_version"] == 1
    assert loaded_config.model == tiny_experiment_config.model
    assert not any(
        name.endswith(".linear.weight") for name in dict(loaded_model.named_parameters())
    )

    input_ids = torch.randint(0, tiny_experiment_config.model.vocab_size, (2, 6))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = loaded_model(input_ids).logits
    torch.testing.assert_close(actual, expected)


def test_tq_default_fp16_scales_roundtrip(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    torch.manual_seed(2)
    model = build_model(tiny_experiment_config.model)
    convert_to_ternary(model, group_size=16, min_features=16)
    model.eval()
    path = tmp_path / "fp16.aurora-tq"
    metadata = save_tq(path, model, tiny_experiment_config, group_size=16, min_features=16)
    assert metadata["quantization"]["scales_dtype"] == "float16"
    loaded_model, _, _ = load_tq_model(path)
    loaded_model.eval()
    input_ids = torch.randint(0, tiny_experiment_config.model.vocab_size, (2, 6))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = loaded_model(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def test_tq_roundtrip_regular_model(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    torch.manual_seed(1)
    model = build_model(tiny_experiment_config.model)
    model.eval()
    path = tmp_path / "regular.aurora-tq"
    metadata = save_tq(
        path,
        model,
        tiny_experiment_config,
        group_size=16,
        min_features=16,
        scales_dtype="float32",
    )
    assert metadata["quantization"]["qat_state"] is False

    loaded_model, _, _ = load_tq_model(path)
    metadata, tensors = load_tq(path)
    assert any(isinstance(tensor, TernaryTensor) for tensor in tensors.values())
    expected_state = {
        name: dequantize_ternary(tensor) if isinstance(tensor, TernaryTensor) else tensor
        for name, tensor in tensors.items()
    }
    for name, tensor in loaded_model.state_dict().items():
        torch.testing.assert_close(tensor, expected_state[name])


def test_tq_rejects_corrupted_tensor(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    model = build_model(tiny_experiment_config.model)
    path = tmp_path / "corrupt.aurora-tq"
    save_tq(path, model, tiny_experiment_config, group_size=16, min_features=16)
    metadata, _ = load_tq(path)
    entry = next(entry for entry in metadata["tensors"] if entry["kind"] == "ternary")

    def corrupt(entries: dict[str, bytes]) -> None:
        data = entries[entry["levels_file"]]
        entries[entry["levels_file"]] = bytes([data[0] ^ 0xFF]) + data[1:]

    _rewrite_archive(path, corrupt)
    with pytest.raises(ValueError, match="Checksum"):
        load_tq(path)


def test_tq_rejects_unsupported_version(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    model = build_model(tiny_experiment_config.model)
    path = tmp_path / "version.aurora-tq"
    save_tq(path, model, tiny_experiment_config, group_size=16, min_features=16)

    def bump_version(entries: dict[str, bytes]) -> None:
        metadata = json.loads(entries["metadata.json"])
        metadata["format_version"] = 999
        entries["metadata.json"] = json.dumps(metadata).encode()

    _rewrite_archive(path, bump_version)
    with pytest.raises(ValueError, match="Unsupported"):
        load_tq(path)


def test_tq_rejects_non_archive(tmp_path: Path) -> None:
    path = tmp_path / "bad.aurora-tq"
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Not an AURORA-TQ"):
        load_tq(path)
    with pytest.raises(FileNotFoundError):
        load_tq(tmp_path / "missing.aurora-tq")


def test_tq_inspect_cli(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    model = build_model(tiny_experiment_config.model)
    path = tmp_path / "inspect.aurora-tq"
    save_tq(path, model, tiny_experiment_config, group_size=16, min_features=16)
    main(["inspect", str(path)])
    output = json.loads(capsys.readouterr().out)
    assert output["format"] == "AURORA-TQ"
    assert output["tensors"] > 0
