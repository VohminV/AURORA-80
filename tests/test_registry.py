from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aurora.config import ExperimentConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.data.version import sha256_file
from aurora.registry import _report_kind, build_registry, main, render_markdown


def _write_tokens(path: Path, tokens: int = 512) -> None:
    rng = np.random.default_rng(3)
    rng.integers(0, 64, size=tokens, dtype=np.uint16).tofile(path)


def _save_checkpoints(tmp_path: Path, config: ExperimentConfig) -> tuple[Path, Path, str]:
    output_dir = tmp_path / "runs" / "run_a"
    training = replace(
        config.training,
        output_dir=str(output_dir),
        eval_interval=0,
        save_interval=0,
        device="cpu",
        dtype_amp="fp32",
    )
    resolved = ExperimentConfig(model=config.model, training=training)
    train_path = tmp_path / "train.bin"
    _write_tokens(train_path)
    dataset = MemmapTokenDataset(train_path, training.seq_len, dtype=training.dtype)
    from aurora.training.trainer import Trainer

    trainer = Trainer(resolved, dataset)
    trainer.step = 3
    trainer._save("best.pt")
    trainer._save("latest.pt")
    return output_dir / "best.pt", output_dir / "latest.pt", str(trainer.run_metadata["run_id"])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_kind_detects_known_reports() -> None:
    cases = [
        ({"results": {"ru": {"perplexity": 1.0}}}, "eval"),
        ({"results": {"fp32": {"perplexity": 1.0, "quantization": {}}}}, "quantization"),
        ({"results": {"core": {"flags": {}, "validation": {}}}}, "ablation"),
        ({"results": [], "peak_rss_mb": 1.0}, "state_stability"),
        ({"tasks": {"mmlu": {"accuracy": 0.5}}}, "zero_shot"),
        ({"metrics": [], "reports": []}, "summary"),
        ({"trials": [], "configurations": []}, "sweep"),
        ({"points": [], "fits": {}}, "scaling"),
        (
            {"tokenizer": {}, "outputs": {}, "dataset_version": "sha256:data"},
            "dataset_manifest",
        ),
        ({"something": 1}, "unknown"),
    ]
    for report, expected in cases:
        assert _report_kind(report) == expected


def test_registry_groups_checkpoints_and_reports(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    best, latest, run_id = _save_checkpoints(tmp_path, tiny_experiment_config)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    eval_path = reports_dir / "eval.json"
    zero_path = reports_dir / "zero_shot.json"
    _write_json(
        eval_path,
        {
            "run_metadata": {"run_id": run_id, "git": {"commit": "abc123"}},
            "checkpoint": {"step": 3},
            "dataset_versions": ["sha256:data"],
            "results": {
                "ru": {"perplexity": 2.5},
                "en": {"perplexity": 3.0},
            },
        },
    )
    _write_json(
        zero_path,
        {
            "run_metadata": {"run_id": run_id, "git": {"commit": "abc123"}},
            "checkpoint": {"step": 3},
            "tasks": {"mmlu": {"accuracy": 0.5, "correct": 2, "evaluated": 4}},
        },
    )

    registry = build_registry([tmp_path / "runs"], [reports_dir])

    assert registry["errors"] == []
    assert len(registry["artifacts"]) == 4
    assert len(registry["runs"]) == 1
    run = registry["runs"][0]
    assert run["run_id"] == run_id
    assert run["git_commit"]
    assert run["dataset_versions"] == ["sha256:data"]
    assert sorted(run["checkpoints"]) == sorted([str(best), str(latest)])
    assert sorted(run["reports"]) == sorted([str(eval_path), str(zero_path)])

    artifacts = {Path(artifact["path"]).name: artifact for artifact in registry["artifacts"]}
    assert artifacts["best.pt"]["kind"] == "checkpoint"
    assert artifacts["best.pt"]["step"] == 3
    assert artifacts["best.pt"]["architecture"] == "aurora"
    assert artifacts["best.pt"]["dataset_versions"] == []
    assert artifacts["best.pt"]["sha256"] is None
    assert artifacts["best.pt"]["git_commit"] == run["git_commit"]
    assert artifacts["eval.json"]["kind"] == "eval"
    assert artifacts["eval.json"]["git_commit"] == "abc123"
    assert artifacts["eval.json"]["summary"]["metrics"]["ru"]["perplexity"] == 2.5
    assert artifacts["zero_shot.json"]["kind"] == "zero_shot"
    assert artifacts["zero_shot.json"]["summary"]["metrics"]["mmlu"]["accuracy"] == 0.5


def test_registry_verify_hashes_artifacts(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    best, latest, _ = _save_checkpoints(tmp_path, tiny_experiment_config)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = reports_dir / "eval.json"
    _write_json(report_path, {"results": {"ru": {"perplexity": 2.0}}})

    registry = build_registry([tmp_path / "runs"], [reports_dir], verify=True)
    digests = {
        Path(artifact["path"]).name: artifact["sha256"] for artifact in registry["artifacts"]
    }

    assert digests["best.pt"] == sha256_file(best)
    assert digests["latest.pt"] == sha256_file(latest)
    assert digests["eval.json"] == sha256_file(report_path)


def test_registry_records_invalid_artifacts_and_unknown_reports(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "broken.pt").write_bytes(b"not a checkpoint")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_json(reports_dir / "notes.json", {"foo": 1})

    registry = build_registry([runs_dir], [reports_dir])

    assert len(registry["errors"]) == 1
    assert registry["errors"][0]["path"].endswith("broken.pt")
    assert [artifact["kind"] for artifact in registry["artifacts"]] == ["unknown"]
    assert registry["artifacts"][0]["summary"] == {}


def test_registry_markdown_and_cli(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    _save_checkpoints(tmp_path, tiny_experiment_config)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    _write_json(reports_dir / "eval.json", {"results": {"ru": {"perplexity": 2.0}}})
    output = tmp_path / "out" / "registry.json"
    markdown_path = tmp_path / "out" / "registry.md"

    main(
        [
            "--runs",
            str(tmp_path / "runs"),
            "--reports",
            str(reports_dir),
            "--verify",
            "--output",
            str(output),
            "--markdown",
            str(markdown_path),
        ]
    )

    registry = json.loads(output.read_text(encoding="utf-8"))
    assert registry["settings"]["verify"] is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# AURORA experiment registry")
    assert "## Runs" in markdown
    assert "## Artifacts" in markdown
    assert "best.pt" in markdown
    assert render_markdown(registry) == markdown


def test_registry_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="Directory not found"):
        main(
            [
                "--runs",
                str(tmp_path / "missing"),
                "--reports",
                str(tmp_path),
                "--output",
                str(tmp_path / "registry.json"),
            ]
        )
