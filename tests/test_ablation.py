from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from aurora.architecture.factory import build_model
from aurora.benchmarks.ablation import build_variant, run_ablation
from aurora.config import ExperimentConfig
from aurora.utils import parameter_counts


def _write_tokens(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=4096, dtype=np.uint16).tofile(path)


def test_build_variant_flags() -> None:
    core = build_variant("core")
    assert core.use_memory is False and core.use_moe is False
    core_memory = build_variant("core_memory")
    assert core_memory.use_memory is True and core_memory.use_moe is False
    full = build_variant("full")
    assert full.use_memory is True and full.use_moe is True
    with pytest.raises(ValueError, match="Unknown variant"):
        build_variant("core_moe")


def test_disabled_components_remove_parameters_and_state(
    tiny_experiment_config: ExperimentConfig,
) -> None:
    base = tiny_experiment_config.model
    variants = {
        "core": replace(base, use_memory=False, use_moe=False),
        "core_memory": replace(base, use_memory=True, use_moe=False),
        "full": base,
    }
    counts = {}
    for name, config in variants.items():
        torch.manual_seed(0)
        model = build_model(config)
        counts[name] = parameter_counts(model)
        output = model(torch.randint(0, config.vocab_size, (2, 4)))
        assert output.logits.shape == (2, 4, config.vocab_size)
        assert bool(torch.isfinite(output.logits).all())
        keys = set(model.state_dict())
        if name == "core":
            assert not any("memory" in key or ".moe." in key for key in keys)
            assert output.state.blocks[0].memory.numel() == 0
            assert float(output.aux_loss) == 0.0
            assert float(output.memory_loss) == 0.0
            assert float(output.consistency_loss) == 0.0
        elif name == "core_memory":
            assert any("memory" in key for key in keys)
            assert not any(".moe." in key for key in keys)
            assert output.state.blocks[0].memory.numel() > 0
            assert float(output.aux_loss) == 0.0
        else:
            assert any("memory" in key for key in keys)
            assert any(".moe." in key for key in keys)
            assert output.state.blocks[0].memory.numel() > 0
            assert float(output.aux_loss.detach()) != 0.0

    assert counts["core"][0] < counts["core_memory"][0] < counts["full"][0]
    assert counts["core"][0] == counts["core"][1]
    assert counts["core_memory"][0] == counts["core_memory"][1]
    assert counts["full"][1] < counts["full"][0]


def test_run_ablation_end_to_end(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 51)
    _write_tokens(val_path, 52)
    output_root = tmp_path / "ablation"
    report = run_ablation(
        tiny_experiment_config,
        variants=("core", "core_memory", "full"),
        max_steps=2,
        train_path=train_path,
        val_path=val_path,
        output_root=output_root,
        device="cpu",
        seed=11,
    )

    settings = report["settings"]
    assert settings["variants"] == ["core", "core_memory", "full"]
    assert settings["max_steps"] == 2
    assert settings["tokens_per_step"] == 2 * 2 * tiny_experiment_config.training.seq_len
    assert settings["max_tokens"] == 2 * settings["tokens_per_step"]
    assert settings["device"] == "cpu"
    assert settings["seed"] == 11
    assert report["evaluation"]["environment"]["device"] == "cpu"
    assert {source["role"] for source in report["dataset"]["sources"]} == {"train", "val"}

    results = report["results"]
    assert list(results) == ["core", "core_memory", "full"]
    for name, result in results.items():
        assert result["use_memory"] == (name != "core")
        assert result["use_moe"] == (name == "full")
        assert result["run_id"]
        assert result["training"]["steps"] == 2
        assert result["training"]["tokens_seen"] == settings["max_tokens"]
        assert result["training"]["tokens_per_second"] > 0
        assert result["parameters"]["total"] > 0
        assert result["parameters"]["active"] > 0
        assert result["validation"]["final_loss"] is not None
        assert result["validation"]["final_perplexity"] is not None
        assert Path(result["checkpoint"]).is_file()
        assert Path(result["output_dir"]) == output_root / name

    assert (
        results["core"]["parameters"]["total"]
        < results["core_memory"]["parameters"]["total"]
        < results["full"]["parameters"]["total"]
    )
    assert results["full"]["parameters"]["active"] < results["full"]["parameters"]["total"]
    assert "loss_delta_vs_core" not in results["core"]["validation"]
    for name in ("core_memory", "full"):
        assert "loss_delta_vs_core" in results[name]["validation"]
        assert "perplexity_ratio_vs_core" in results[name]["validation"]


def test_run_ablation_validates_inputs(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path, 61)
    with pytest.raises(ValueError, match="validation token file"):
        run_ablation(tiny_experiment_config, train_path=train_path, val_path="")
    with pytest.raises(ValueError, match="At least one variant"):
        run_ablation(tiny_experiment_config, variants=(), train_path=train_path)
    with pytest.raises(ValueError, match="Unknown variant"):
        run_ablation(tiny_experiment_config, variants=("core_moe",), train_path=train_path)
    with pytest.raises(ValueError, match="aurora architecture"):
        transformer = replace(
            tiny_experiment_config.model,
            architecture="transformer",
            attention_heads=2,
            ffn_hidden=64,
        )
        run_ablation(
            ExperimentConfig(model=transformer, training=tiny_experiment_config.training),
            train_path=train_path,
        )


def test_ablation_report_serializes(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 71)
    _write_tokens(val_path, 72)
    report = run_ablation(
        tiny_experiment_config,
        variants=("core",),
        max_steps=1,
        train_path=train_path,
        val_path=val_path,
        output_root=tmp_path / "serialized",
        device="cpu",
    )
    payload = json.loads(json.dumps(report))
    assert payload["results"]["core"]["training"]["steps"] == 1
