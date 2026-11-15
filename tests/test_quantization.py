from __future__ import annotations

import pytest
import torch

from aurora.architecture.model import AuroraModel
from aurora.config import ModelConfig
from aurora.quantization import (
    TernaryLinear,
    apply_weight_quantization,
    calibrate_group_scales,
    convert_to_ternary,
    dequantize_ternary,
    pack_ternary,
    quantize_symmetric,
    quantize_ternary,
    quantized_tensor_bytes,
    round_weights_fp16,
    ternary_ste,
    unpack_ternary,
)
from aurora.quantization.ternary import theoretical_size_bytes
from aurora.utils import parameter_counts


def test_quantize_and_dequantize() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    quantized = quantize_ternary(weight, group_size=16)
    assert quantized.levels.shape == weight.shape
    assert set(quantized.levels.unique().tolist()) <= {-1, 0, 1}
    dequantized = dequantize_ternary(quantized)
    assert dequantized.shape == weight.shape
    assert torch.isfinite(dequantized).all()


def test_calibration_reduces_reconstruction_error() -> None:
    torch.manual_seed(3)
    weight = torch.randn(32, 64)
    baseline = quantize_ternary(weight, group_size=16)
    calibrated = quantize_ternary(weight, group_size=16, calibration_steps=4)
    baseline_error = (weight - dequantize_ternary(baseline)).pow(2).mean()
    calibrated_error = (weight - dequantize_ternary(calibrated)).pow(2).mean()
    assert calibrated_error <= baseline_error
    assert calibrated_error < baseline_error


def test_calibration_is_deterministic() -> None:
    torch.manual_seed(4)
    weight = torch.randn(16, 32)
    first = quantize_ternary(weight, group_size=8, calibration_steps=3)
    second = quantize_ternary(weight, group_size=8, calibration_steps=3)
    torch.testing.assert_close(first.levels, second.levels)
    torch.testing.assert_close(first.scales, second.scales)

    groups = weight.reshape(-1, 8)
    initial = groups.abs().mean(dim=-1).clamp_min(1e-8)
    capped = calibrate_group_scales(groups, initial, steps=0)
    torch.testing.assert_close(capped, initial)
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_group_scales(groups, initial, steps=-1)


def test_integer_quantization_error_ordering() -> None:
    torch.manual_seed(6)
    weight = torch.randn(16, 64)
    fp16 = round_weights_fp16(weight)
    int8 = quantize_symmetric(weight, bits=8, group_size=16)
    int4 = quantize_symmetric(weight, bits=4, group_size=16)
    fp16_error = (weight - fp16).pow(2).mean()
    int8_error = (weight - int8).pow(2).mean()
    int4_error = (weight - int4).pow(2).mean()
    assert fp16_error < int8_error < int4_error
    assert quantized_tensor_bytes((16, 64), 2, 16) == 16 * 64 * 2 // 8 + 16 * 4 * 2
    with pytest.raises(ValueError, match="bits"):
        quantize_symmetric(weight, bits=1)
    with pytest.raises(ValueError, match="bits"):
        quantized_tensor_bytes((16, 64), 17, 16)


def test_apply_weight_quantization_stats(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(7)
    model = AuroraModel(tiny_model_config)
    stats = apply_weight_quantization(
        model,
        lambda weight: quantize_symmetric(weight, bits=8, group_size=8),
        bits=8,
        group_size=8,
        min_features=16,
    )
    assert stats.modules > 0
    assert stats.parameters > 0
    assert stats.weight_mse > 0
    assert stats.dense_bytes > stats.estimated_bytes
    input_ids = torch.randint(0, tiny_model_config.vocab_size, (1, 4))
    with torch.no_grad():
        logits = model(input_ids).logits
    assert torch.isfinite(logits).all()


def test_ste_has_gradient() -> None:
    weight = torch.randn(8, 64, requires_grad=True)
    quantized = ternary_ste(weight, group_size=16)
    quantized.sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()


def test_pack_unpack_roundtrip_with_padding() -> None:
    levels = torch.randint(-1, 2, (101,))
    packed, numel = pack_ternary(levels)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 26
    restored = unpack_ternary(packed, numel)
    torch.testing.assert_close(restored, levels.to(torch.int8))


def test_theoretical_size() -> None:
    size = theoretical_size_bytes(80_000_000_000, group_size=128)
    assert 20_000_000_000 <= size <= 22_000_000_000


def test_ternary_linear() -> None:
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    ternary = TernaryLinear(linear, group_size=16)
    x = torch.randn(4, 64, requires_grad=True)
    output = ternary(x)
    assert output.shape == (4, 32)
    output.sum().backward()
    assert linear.weight.grad is not None
    assert x.grad is not None


def test_convert_model_to_ternary(tiny_model_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = AuroraModel(tiny_model_config)
    total_before, active_before = parameter_counts(model)
    converted = convert_to_ternary(model, group_size=16, min_features=16)
    assert converted > 0
    assert convert_to_ternary(model, group_size=16, min_features=16) == 0
    total_after, active_after = parameter_counts(model)
    assert (total_after, active_after) == (total_before, active_before)

    input_ids = torch.randint(0, tiny_model_config.vocab_size, (1, 4))
    output = model(input_ids)
    assert output.logits.shape == (1, 4, tiny_model_config.vocab_size)
    assert torch.isfinite(output.logits).all()
