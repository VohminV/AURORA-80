from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig, load_config
from aurora.scaling import (
    ScalingPoint,
    build_scaling_report,
    estimate_compute_optimal,
    fit_bivariate_power_law,
    fit_power_law,
    load_points_json,
    main,
    point_from_checkpoint,
    render_markdown,
)
from aurora.utils import parameter_counts

CONFIG_COUNTS = {
    "aurora_10m.yaml": (10_237_408, 8_762_848),
    "aurora_30m.yaml": (30_710_704, 20_093_872),
    "aurora_100m.yaml": (105_727_560, 49_104_456),
    "aurora_300m.yaml": (300_620_428, 94_182_028),
}


def _single_axis_points() -> list[ScalingPoint]:
    parameters = [1e8, 3e8, 1e9, 3e9, 1e10]
    points = []
    for index, total in enumerate(parameters):
        points.append(
            ScalingPoint(
                name=f"p{index}",
                parameters=total,
                active_parameters=0.8 * total,
                tokens=20.0 * total,
                loss=1.2 + 2.5e9 * total**-0.5,
            )
        )
    return points


def _bivariate_points() -> list[ScalingPoint]:
    parameters = [1e8, 3e8, 1e9, 3e9, 1e10, 3e10]
    tokens = [2e9, 3e10, 2e10, 6e10, 6e11, 6e11]
    amplitude = 2.0 * 1e9**0.3
    token_amplitude = 1.5 * 1e10**0.3
    return [
        ScalingPoint(
            name=f"b{index}",
            parameters=n,
            tokens=d,
            loss=1.0 + amplitude * n**-0.3 + token_amplitude * d**-0.3,
        )
        for index, (n, d) in enumerate(zip(parameters, tokens, strict=True))
    ]


def _checkpoint_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "format_version": 2,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": None,
        "step": 3,
        "config": config.to_dict(),
        "qat_active": False,
        "best_val": 9.0,
        "rng_state": None,
        "run_metadata": {"run_id": "scaling-test"},
        "metrics": {
            "tokens_seen": 1234,
            "last_validation": {"loss": 2.5},
        },
    }


def test_fit_power_law_recovers_synthetic_parameters() -> None:
    x_values = [1e8, 3e8, 1e9, 3e9, 1e10]
    losses = [1.2 + 2.5e9 * value**-0.5 for value in x_values]

    fit = fit_power_law(x_values, losses)

    assert fit["alpha"] == pytest.approx(0.5, abs=2e-3)
    assert fit["A"] == pytest.approx(2.5e9, rel=1e-2)
    assert fit["E"] == pytest.approx(1.2, rel=1e-5)
    assert fit["r2"] > 0.999999
    assert fit["n"] == 5

    with pytest.raises(ValueError, match="At least four"):
        fit_power_law(x_values[:3], losses[:3])


def test_fit_bivariate_power_law_and_compute_optimal() -> None:
    points = _bivariate_points()

    fit = fit_bivariate_power_law(points)

    assert fit is not None
    assert fit["alpha"] == pytest.approx(0.3, abs=2e-3)
    assert fit["beta"] == pytest.approx(0.3, abs=2e-3)
    assert fit["A"] == pytest.approx(2.0 * 1e9**0.3, rel=1e-2)
    assert fit["B"] == pytest.approx(1.5 * 1e10**0.3, rel=1e-2)
    assert fit["E"] == pytest.approx(1.0, rel=1e-5)
    assert fit["reliable"] is True
    assert fit_bivariate_power_law(points[:4]) is None

    estimates = estimate_compute_optimal(fit, [1e22])

    assert len(estimates) == 1
    estimate = estimates[0]
    assert 1e9 < estimate["parameters"] < 1e11
    assert estimate["tokens"] > estimate["parameters"]
    assert estimate["predicted_loss"] > fit["E"]
    assert estimate["extrapolated"] is False

    noisy = dict(fit)
    noisy["reliable"] = False
    assert estimate_compute_optimal(noisy, [1e22]) == []


def test_load_points_json_aliases_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "points.json"
    path.write_text(
        json.dumps(
            {
                "points": [
                    {
                        "name": "small",
                        "total_parameters": 1e8,
                        "tokens_seen": 2e9,
                        "loss": 2.1,
                        "metadata": {"seed": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    points = load_points_json(path)

    assert points[0].name == "small"
    assert points[0].parameters == 1e8
    assert points[0].tokens == 2e9
    assert points[0].compute == 6e8 * 2e9
    assert points[0].metadata["seed"] == 1
    assert points[0].metadata["source"] == str(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            [
                {"name": "same", "parameters": 1, "tokens": 1, "loss": 1},
                {"name": "same", "parameters": 2, "tokens": 2, "loss": 2},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate point name"):
        load_points_json(duplicate)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps([{"name": "x", "parameters": 1, "tokens": 1, "loss": 1, "bad": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown point fields"):
        load_points_json(unknown)


def test_point_from_checkpoint_extracts_metrics(
    tmp_path: Path, tiny_experiment_config: ExperimentConfig
) -> None:
    path = tmp_path / "step_3.pt"
    torch.save(_checkpoint_payload(tiny_experiment_config), path)
    model = build_model(tiny_experiment_config.model)
    total, active = parameter_counts(model)

    point = point_from_checkpoint("tiny", path)

    assert point.parameters == total
    assert point.active_parameters == active
    assert point.tokens == 1234
    assert point.loss == 2.5
    assert point.metadata["run_id"] == "scaling-test"
    assert point.metadata["step"] == 3

    fallback = _checkpoint_payload(tiny_experiment_config)
    fallback["metrics"] = {}
    fallback["best_val"] = 3.25
    fallback_path = tmp_path / "best.pt"
    torch.save(fallback, fallback_path)
    fallback_point = point_from_checkpoint("fallback", fallback_path)
    assert fallback_point.tokens == (
        3
        * tiny_experiment_config.training.batch_size
        * tiny_experiment_config.training.grad_accum_steps
        * tiny_experiment_config.training.seq_len
    )
    assert fallback_point.loss == 3.25


def test_build_scaling_report_holdout_and_skipped() -> None:
    points = _single_axis_points()

    report = build_scaling_report(points)

    assert set(report["fits"]) == {
        "parameters",
        "active_parameters",
        "tokens",
        "compute",
        "active_compute",
    }
    assert report["holdout"]["name"] == "p4"
    assert report["holdout"]["status"] == "evaluated"
    for detail in report["holdout"]["fits"].values():
        assert detail["relative_error"] < 0.01

    skipped = build_scaling_report(points[:4], holdout="p3")

    assert skipped["holdout"]["status"] == "skipped"

    with pytest.raises(ValueError, match="Holdout point not found"):
        build_scaling_report(points, holdout="missing")
    with pytest.raises(ValueError, match="At least four"):
        build_scaling_report(points[:3])


def test_scaling_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps({"points": [point.to_dict() for point in _bivariate_points()]}),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "scaling.json"
    markdown_path = tmp_path / "out" / "scaling.md"

    main(
        [
            "--points",
            str(points_path),
            "--compute-budget",
            "1e22",
            "--output",
            str(output),
            "--markdown",
            str(markdown_path),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["format_version"] == 1
    assert report["bivariate"]["reliable"] is True
    assert len(report["compute_optimal"]) == 1
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# AURORA scaling law report")
    assert "## Compute-optimal estimates" in markdown
    assert render_markdown(report) == markdown

    too_small = tmp_path / "too_small.json"
    too_small.write_text(
        json.dumps({"points": [point.to_dict() for point in _bivariate_points()[:3]]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="At least four"):
        main(
            [
                "--points",
                str(too_small),
                "--output",
                str(tmp_path / "too_small_report.json"),
            ]
        )


@pytest.mark.parametrize("config_name, counts", CONFIG_COUNTS.items())
def test_scaling_configs_have_expected_parameter_counts(
    config_name: str, counts: tuple[int, int]
) -> None:
    config = load_config(Path("configs") / config_name)

    with torch.device("meta"):
        model = build_model(config.model)

    assert parameter_counts(model) == counts
