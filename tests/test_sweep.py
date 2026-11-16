from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from aurora.config import ExperimentConfig
from aurora.sweep import (
    SweepSpec,
    apply_overrides,
    build_trials,
    load_sweep_spec,
    main,
    parse_sweep_spec,
    render_markdown,
    run_sweep,
)


def _write_tokens(path: Path, seed: int, tokens: int = 4096) -> None:
    rng = np.random.default_rng(seed)
    rng.integers(0, 64, size=tokens, dtype=np.uint16).tofile(path)


def _write_config(path: Path, config: ExperimentConfig) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


def _write_spec(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _sweep_base(tmp_path: Path, config: ExperimentConfig) -> Path:
    training = replace(
        config.training,
        train_path="",
        val_path="",
        eval_interval=0,
        save_interval=1,
        log_interval=1,
    )
    base = ExperimentConfig(model=config.model, training=training)
    path = tmp_path / "base.yaml"
    _write_config(path, base)
    return path


def test_parse_and_load_sweep_spec(tmp_path: Path) -> None:
    path = tmp_path / "sweep.yaml"
    _write_spec(
        path,
        {
            "base": "configs/smoke.yaml",
            "max_steps": 10,
            "seeds": [1, 2],
            "output_dir": "runs/sweeps/test",
            "parameters": {
                "training.learning_rate": [0.0001, 0.0003],
                "model.n_levels": [2, 3],
            },
        },
    )

    spec = load_sweep_spec(path)

    assert spec.base == "configs/smoke.yaml"
    assert spec.max_steps == 10
    assert spec.seeds == (1, 2)
    assert spec.output_dir == "runs/sweeps/test"
    assert spec.parameters == {
        "model.n_levels": (2, 3),
        "training.learning_rate": (0.0001, 0.0003),
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "non-empty 'base'"),
        ({"base": "x.yaml", "unknown": 1}, "Unknown sweep spec fields"),
        ({"base": "x.yaml", "parameters": {"training.seed": [1]}}, "not tunable"),
        ({"base": "x.yaml", "parameters": {"d_model": [32]}}, "must start with"),
        (
            {"base": "x.yaml", "parameters": {"training.learning_rate": 0.1}},
            "non-empty list",
        ),
        ({"base": "x.yaml", "seeds": [1, 1]}, "Duplicate sweep seed"),
        ({"base": "x.yaml", "seeds": [-1]}, "non-negative integers"),
        ({"base": "x.yaml", "max_steps": 0}, "positive integer"),
        ({"base": "x.yaml", "output_dir": ""}, "non-empty path"),
    ],
)
def test_parse_sweep_spec_rejects_invalid_input(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_sweep_spec(payload)


def test_apply_overrides_coerces_and_revalidates(
    tiny_experiment_config: ExperimentConfig,
) -> None:
    updated = apply_overrides(
        tiny_experiment_config,
        {"training.learning_rate": 1, "model.n_layers": 3},
    )

    assert updated.training.learning_rate == 1.0
    assert isinstance(updated.training.learning_rate, float)
    assert updated.model.n_layers == 3
    assert tiny_experiment_config.training.learning_rate == 1e-3

    with pytest.raises(ValueError, match="not tunable"):
        apply_overrides(tiny_experiment_config, {"training.seed": 1})
    with pytest.raises(ValueError, match="Unknown parameter"):
        apply_overrides(tiny_experiment_config, {"model.unknown": 1})
    with pytest.raises(ValueError, match="expects an integer"):
        apply_overrides(tiny_experiment_config, {"model.n_layers": 2.5})
    with pytest.raises(ValueError, match="expects a boolean"):
        apply_overrides(tiny_experiment_config, {"model.tie_embeddings": 1})
    with pytest.raises(ValueError, match="n_levels and d_state"):
        apply_overrides(tiny_experiment_config, {"model.n_levels": 0})


def test_build_trials_is_deterministic_and_capped() -> None:
    spec = SweepSpec(
        base="base.yaml",
        parameters={
            "training.learning_rate": (1e-4, 3e-4),
            "training.weight_decay": (0.0, 0.1),
        },
        seeds=(0, 1),
    )

    trials = build_trials(spec, base_seed=42, max_trials=8)

    assert len(trials) == 8
    assert trials[0].name == "trial_000_seed0"
    assert trials[1].name == "trial_000_seed1"
    assert trials[2].name == "trial_001_seed0"
    assert trials[0].overrides == {
        "training.learning_rate": 1e-4,
        "training.weight_decay": 0.0,
    }
    assert trials[-1].overrides == {
        "training.learning_rate": 3e-4,
        "training.weight_decay": 0.1,
    }
    assert [trial.seed for trial in trials] == [0, 1, 0, 1, 0, 1, 0, 1]

    with pytest.raises(ValueError, match="exceeds --max-trials"):
        build_trials(spec, base_seed=42, max_trials=7)


def test_run_sweep_end_to_end(tmp_path: Path, tiny_experiment_config: ExperimentConfig) -> None:
    base = _sweep_base(tmp_path, tiny_experiment_config)
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 71)
    _write_tokens(val_path, 72)
    spec = SweepSpec(
        base=str(base),
        parameters={"training.learning_rate": (1e-3, 5e-4)},
        seeds=(0, 1),
        max_steps=1,
    )

    report = run_sweep(
        spec,
        spec_path=tmp_path / "sweep.yaml",
        train_path=train_path,
        val_path=val_path,
        output_root=tmp_path / "sweep",
        device="cpu",
    )

    assert report["summary"] == {
        "trials": 4,
        "completed": 4,
        "failed": 0,
        "planned": 0,
    }
    assert report["settings"]["tokens_per_step"] == (
        tiny_experiment_config.training.batch_size
        * tiny_experiment_config.training.grad_accum_steps
        * tiny_experiment_config.training.seq_len
    )
    assert report["settings"]["max_tokens"] == report["settings"]["tokens_per_step"]
    assert report["evaluation"]["environment"]["device"] == "cpu"
    assert {source["role"] for source in report["dataset"]["sources"]} == {
        "train",
        "val",
    }

    assert len(report["trials"]) == 4
    for entry in report["trials"]:
        assert entry["status"] == "completed"
        assert entry["validation"]["final_loss"] is not None
        assert entry["validation"]["final_perplexity"] is not None
        assert entry["training"]["tokens_seen"] == report["settings"]["tokens_per_step"]
        assert Path(entry["config_path"]).is_file()
        assert Path(entry["checkpoint"]).is_file()
        saved = yaml.safe_load(Path(entry["config_path"]).read_text(encoding="utf-8"))
        assert saved["training"]["seed"] == entry["seed"]
        assert saved["training"]["learning_rate"] == entry["overrides"]["training.learning_rate"]
        assert saved["training"]["max_steps"] == 1

    assert len(report["configurations"]) == 2
    assert all(configuration["runs"] == 2 for configuration in report["configurations"])
    assert all(
        configuration["std_final_loss"] is not None for configuration in report["configurations"]
    )
    assert report["best"] is not None
    assert report["best"]["runs"] == 2
    assert report["best"]["overrides"] in [
        configuration["overrides"] for configuration in report["configurations"]
    ]


def test_run_sweep_dry_run_generates_configs(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    base = _sweep_base(tmp_path, tiny_experiment_config)
    spec = SweepSpec(
        base=str(base),
        parameters={"training.weight_decay": (0.0, 0.1)},
        max_steps=2,
    )

    report = run_sweep(
        spec,
        output_root=tmp_path / "generated",
        device="cpu",
        dry_run=True,
    )

    assert report["settings"]["dry_run"] is True
    assert report["summary"] == {
        "trials": 2,
        "completed": 0,
        "failed": 0,
        "planned": 2,
    }
    assert report["configurations"] == []
    assert report["best"] is None
    assert "dataset" not in report
    for entry in report["trials"]:
        assert entry["status"] == "planned"
        assert "validation" not in entry
        config_path = Path(entry["config_path"])
        assert config_path.is_file()
        assert Path(entry["output_dir"]).is_dir()


def test_run_sweep_records_failed_trials(
    tmp_path: Path,
    tiny_experiment_config: ExperimentConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _sweep_base(tmp_path, tiny_experiment_config)
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 81)
    _write_tokens(val_path, 82)
    spec = SweepSpec(base=str(base), parameters={}, max_steps=1)

    def fail_fit(self: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("aurora.sweep.Trainer.fit", fail_fit)

    report = run_sweep(
        spec,
        train_path=train_path,
        val_path=val_path,
        output_root=tmp_path / "sweep",
        device="cpu",
    )

    assert report["summary"] == {
        "trials": 1,
        "completed": 0,
        "failed": 1,
        "planned": 0,
    }
    assert report["trials"][0]["status"] == "failed"
    assert report["trials"][0]["error"] == "RuntimeError: boom"
    assert report["best"] is None
    assert report["configurations"] == []


def test_run_sweep_rejects_mixture(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    training = replace(
        tiny_experiment_config.training,
        train_path="",
        train_paths=(str(tmp_path / "a.bin"), str(tmp_path / "b.bin")),
        train_weights=(0.5, 0.5),
    )
    base = tmp_path / "base.yaml"
    _write_config(
        base,
        replace(tiny_experiment_config, training=training),
    )

    with pytest.raises(ValueError, match="single training file"):
        run_sweep(
            SweepSpec(base=str(base), parameters={}, output_dir=str(tmp_path / "out")),
            val_path=tmp_path / "val.bin",
            dry_run=True,
        )


def test_sweep_cli_writes_json_and_markdown(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    base = _sweep_base(tmp_path, tiny_experiment_config)
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, 91)
    _write_tokens(val_path, 92)
    spec_path = tmp_path / "sweep.yaml"
    _write_spec(
        spec_path,
        {
            "base": str(base),
            "max_steps": 1,
            "parameters": {"training.learning_rate": [0.001, 0.002]},
        },
    )
    output = tmp_path / "out" / "sweep.json"
    markdown_path = tmp_path / "out" / "sweep.md"

    main(
        [
            "--spec",
            str(spec_path),
            "--train-path",
            str(train_path),
            "--val-path",
            str(val_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
            "--output",
            str(output),
            "--markdown",
            str(markdown_path),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["completed"] == 2
    assert report["settings"]["seeds"] == [tiny_experiment_config.training.seed]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# AURORA hyperparameter sweep")
    assert "## Configurations" in markdown
    assert "## Trials" in markdown
    assert render_markdown(report) == markdown


def test_sweep_cli_rejects_missing_spec(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No such file"):
        main(
            [
                "--spec",
                str(tmp_path / "missing.yaml"),
                "--output",
                str(tmp_path / "sweep.json"),
            ]
        )
