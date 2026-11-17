from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig
from aurora.provenance import git_metadata, utc_now
from aurora.training.checkpoint import validate_checkpoint
from aurora.utils import parameter_counts

FORMAT_VERSION = 1
AXES = ("parameters", "active_parameters", "tokens", "compute", "active_compute")
POINT_FIELDS = frozenset(
    {
        "name",
        "parameters",
        "total_parameters",
        "active_parameters",
        "tokens",
        "tokens_seen",
        "loss",
        "metadata",
        "compute",
        "active_compute",
    }
)


@dataclass(frozen=True)
class ScalingPoint:
    name: str
    parameters: float
    tokens: float
    loss: float
    active_parameters: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Scaling point name must be a non-empty string")
        object.__setattr__(
            self, "parameters", _number(self.parameters, "parameters", positive=True)
        )
        object.__setattr__(self, "tokens", _number(self.tokens, "tokens", positive=True))
        object.__setattr__(self, "loss", _number(self.loss, "loss"))
        if self.active_parameters is not None:
            active = _number(self.active_parameters, "active_parameters", positive=True)
            if active > self.parameters:
                raise ValueError("active_parameters must not exceed parameters")
            object.__setattr__(self, "active_parameters", active)

    @property
    def compute(self) -> float:
        return 6.0 * self.parameters * self.tokens

    @property
    def active_compute(self) -> float | None:
        if self.active_parameters is None:
            return None
        return 6.0 * self.active_parameters * self.tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "active_parameters": self.active_parameters,
            "tokens": self.tokens,
            "loss": self.loss,
            "compute": self.compute,
            "active_compute": self.active_compute,
            "metadata": self.metadata,
        }


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _axis_value(point: ScalingPoint, axis: str) -> float | None:
    value = getattr(point, axis)
    return None if value is None else float(value)


def point_from_checkpoint(name: str, path: str | Path) -> ScalingPoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    validate_checkpoint(checkpoint, check_finite=False)
    config_data = checkpoint.get("config")
    if not isinstance(config_data, dict):
        raise ValueError(f"{checkpoint_path}: checkpoint has no config")
    config = ExperimentConfig.from_dict(config_data)
    with torch.device("meta"):
        model = build_model(config.model)
    total_parameters, active_parameters = parameter_counts(model)

    training = config.training
    tokens_per_step = training.batch_size * training.grad_accum_steps * training.seq_len
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    tokens = metrics.get("tokens_seen")
    if tokens is None:
        step = checkpoint.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"{checkpoint_path}: checkpoint has no valid step")
        tokens = step * tokens_per_step
    loss = None
    last_validation = metrics.get("last_validation")
    if isinstance(last_validation, dict):
        loss = last_validation.get("loss")
    if loss is None:
        loss = checkpoint.get("best_val")
    if loss is None:
        raise ValueError(
            f"{checkpoint_path}: checkpoint has no validation loss; run evaluation first"
        )
    run_metadata = checkpoint.get("run_metadata")
    if not isinstance(run_metadata, dict):
        run_metadata = {}
    git = run_metadata.get("git")
    if not isinstance(git, dict):
        git = {}
    dataset = run_metadata.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    return ScalingPoint(
        name=name,
        parameters=float(total_parameters),
        active_parameters=float(active_parameters),
        tokens=_number(tokens, "tokens", positive=True),
        loss=_number(loss, "loss"),
        metadata={
            "source": str(checkpoint_path),
            "run_id": run_metadata.get("run_id"),
            "git_commit": git.get("commit"),
            "dataset_versions": dataset.get("versions", []),
            "architecture": config.model.architecture,
            "step": checkpoint.get("step"),
            "qat_active": checkpoint.get("qat_active"),
        },
    )


def _point_from_payload(payload: Mapping[str, Any], source: str) -> ScalingPoint:
    unknown = set(payload) - POINT_FIELDS
    if unknown:
        raise ValueError(f"{source}: unknown point fields: {sorted(unknown)}")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{source}: point requires a non-empty name")
    parameters = payload.get("parameters", payload.get("total_parameters"))
    if parameters is None:
        raise ValueError(f"{source}: point {name!r} requires parameters")
    tokens = payload.get("tokens", payload.get("tokens_seen"))
    if tokens is None:
        raise ValueError(f"{source}: point {name!r} requires tokens")
    loss = payload.get("loss")
    if loss is None:
        raise ValueError(f"{source}: point {name!r} requires loss")
    active = payload.get("active_parameters")
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: point {name!r} metadata must be an object")
    return ScalingPoint(
        name=name,
        parameters=parameters,
        active_parameters=active,
        tokens=tokens,
        loss=loss,
        metadata={"source": source, **metadata},
    )


def load_points_json(path: str | Path) -> list[ScalingPoint]:
    points_path = Path(path)
    if not points_path.is_file():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    try:
        raw = json.loads(points_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{points_path}: invalid JSON: {error.msg}") from error
    if isinstance(raw, dict) and "points" in raw:
        unknown = set(raw) - {"points"}
        if unknown:
            raise ValueError(f"{points_path}: unknown top-level fields: {sorted(unknown)}")
        items = raw["points"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{points_path}: expected a non-empty list of points")
    points = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"{points_path}: every point must be an object")
        points.append(_point_from_payload(item, str(points_path)))
    _require_unique_names(points)
    return points


def parse_point_args(items: Sequence[str]) -> list[ScalingPoint]:
    points: list[ScalingPoint] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Checkpoint point must be NAME=PATH, got: {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name:
            raise ValueError(f"Point name is empty in: {item!r}")
        if name in seen:
            raise ValueError(f"Duplicate point name: {name}")
        if path.suffix != ".pt":
            raise ValueError(f"--point expects a .pt checkpoint: {path}")
        points.append(point_from_checkpoint(name, path))
        seen.add(name)
    return points


def _require_unique_names(points: Sequence[ScalingPoint]) -> None:
    names: set[str] = set()
    for point in points:
        if point.name in names:
            raise ValueError(f"Duplicate point name: {point.name}")
        names.add(point.name)


def _fit_alpha(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> dict[str, float] | None:
    x_ref = float(np.max(x))
    best: dict[str, float] | None = None
    for raw_alpha in grid:
        alpha = float(raw_alpha)
        z = (x / x_ref) ** (-alpha)
        design = np.column_stack((np.ones_like(z), z))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        offset = float(coefficients[0])
        scaled_amplitude = float(coefficients[1])
        if not math.isfinite(offset) or not math.isfinite(scaled_amplitude):
            continue
        if offset < 0.0 or scaled_amplitude <= 0.0:
            continue
        amplitude = scaled_amplitude * x_ref**alpha
        predictions = offset + amplitude * x ** (-alpha)
        sse = float(np.sum((y - predictions) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "alpha": alpha,
                "A": amplitude,
                "E": offset,
                "sse": sse,
            }
    return best


def fit_power_law(x_values: Sequence[float], losses: Sequence[float]) -> dict[str, Any]:
    """Fit L = E + A * x^-alpha with a profile grid over alpha."""
    if len(x_values) != len(losses):
        raise ValueError("x_values and losses must have the same length")
    if len(x_values) < 4:
        raise ValueError("At least four points are required to fit a stable power law")
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(losses, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Scaling points must be finite")
    if np.any(x <= 0.0):
        raise ValueError("Scaling axis values must be positive")

    best = _fit_alpha(x, y, np.geomspace(1e-3, 3.0, 300))
    if best is None:
        raise ValueError("Could not fit a power law with non-negative offset and amplitude")
    radius = max(1e-6, 0.05 * best["alpha"])
    for _ in range(8):
        fine = np.linspace(best["alpha"] - radius, best["alpha"] + radius, 201)
        refined = _fit_alpha(x, y, fine)
        if refined is None:
            break
        if refined["sse"] <= best["sse"]:
            best = refined
        radius *= 0.1

    predictions = best["E"] + best["A"] * x ** (-best["alpha"])
    sse = float(np.sum((y - predictions) ** 2))
    mean = float(np.mean(y))
    sst = float(np.sum((y - mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else (1.0 if sse == 0.0 else 0.0)
    return {
        "alpha": best["alpha"],
        "A": best["A"],
        "E": best["E"],
        "rmse": math.sqrt(sse / len(x)),
        "r2": r2,
        "n": len(x),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }


def _fit_bivariate_grid(
    n: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
) -> dict[str, Any] | None:
    n_ref = float(np.max(n))
    d_ref = float(np.max(d))
    best: dict[str, Any] | None = None
    for raw_alpha in alpha_grid:
        alpha = float(raw_alpha)
        zn = (n / n_ref) ** (-alpha)
        for raw_beta in beta_grid:
            beta = float(raw_beta)
            zd = (d / d_ref) ** (-beta)
            design = np.column_stack((np.ones_like(zn), zn, zd))
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            offset = float(coefficients[0])
            scaled_a = float(coefficients[1])
            scaled_b = float(coefficients[2])
            if not all(math.isfinite(value) for value in (offset, scaled_a, scaled_b)):
                continue
            if offset < 0.0 or scaled_a <= 0.0 or scaled_b <= 0.0:
                continue
            amplitude = scaled_a * n_ref**alpha
            token_amplitude = scaled_b * d_ref**beta
            predictions = offset + amplitude * n ** (-alpha) + token_amplitude * d ** (-beta)
            sse = float(np.sum((y - predictions) ** 2))
            if best is None or sse < best["sse"]:
                best = {
                    "alpha": alpha,
                    "A": amplitude,
                    "beta": beta,
                    "B": token_amplitude,
                    "E": offset,
                    "sse": sse,
                    "condition": float(np.linalg.cond(design)),
                }
    return best


def fit_bivariate_power_law(points: Sequence[ScalingPoint]) -> dict[str, Any] | None:
    """Fit L = E + A*N^-alpha + B*D^-beta."""
    if len(points) < 5:
        return None
    n = np.asarray([point.parameters for point in points], dtype=float)
    d = np.asarray([point.tokens for point in points], dtype=float)
    y = np.asarray([point.loss for point in points], dtype=float)

    coarse = np.linspace(0.02, 1.5, 75)
    best = _fit_bivariate_grid(n, d, y, coarse, coarse)
    if best is None:
        return None
    radius_alpha = max(1e-6, 0.02 * best["alpha"])
    radius_beta = max(1e-6, 0.02 * best["beta"])
    for _ in range(5):
        fine = np.linspace(best["alpha"] - radius_alpha, best["alpha"] + radius_alpha, 21)
        fine_beta = np.linspace(best["beta"] - radius_beta, best["beta"] + radius_beta, 21)
        refined = _fit_bivariate_grid(n, d, y, fine, fine_beta)
        if refined is None:
            break
        if refined["sse"] <= best["sse"]:
            best = refined
        radius_alpha *= 0.1
        radius_beta *= 0.1

    predictions = best["E"] + best["A"] * n ** (-best["alpha"]) + best["B"] * d ** (-best["beta"])
    sse = float(np.sum((y - predictions) ** 2))
    mean = float(np.mean(y))
    sst = float(np.sum((y - mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else (1.0 if sse == 0.0 else 0.0)
    reliable = len(points) >= 5 and best["condition"] < 1e8
    result: dict[str, Any] = {
        "E": best["E"],
        "A": best["A"],
        "alpha": best["alpha"],
        "B": best["B"],
        "beta": best["beta"],
        "rmse": math.sqrt(sse / len(n)),
        "r2": r2,
        "n": len(points),
        "condition": best["condition"],
        "reliable": reliable,
        "parameters_min": float(np.min(n)),
        "parameters_max": float(np.max(n)),
        "tokens_min": float(np.min(d)),
        "tokens_max": float(np.max(d)),
    }
    if not reliable:
        result["warning"] = (
            "Bivariate fit is underdetermined or ill-conditioned; treat exponents "
            "and compute-optimal estimates as indicative only."
        )
    return result


def _golden_section(function: Any, low: float, high: float, iterations: int = 200) -> float:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = low, high
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = function(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = function(d)
    return (left + right) / 2.0


def estimate_compute_optimal(
    fit: Mapping[str, Any], compute_budgets: Sequence[float]
) -> list[dict[str, Any]]:
    """Minimize the bivariate law at 6ND = C and report optimal N and D."""
    if not fit.get("reliable", False):
        return []
    alpha = float(fit["alpha"])
    beta = float(fit["beta"])
    amplitude = float(fit["A"])
    token_amplitude = float(fit["B"])
    offset = float(fit["E"])
    estimates: list[dict[str, Any]] = []
    for raw_budget in compute_budgets:
        compute = _number(raw_budget, "compute budget", positive=True)
        max_parameters = min(1e13, compute / 6.0)
        if max_parameters <= 1e5:
            raise ValueError(f"Compute budget is too small: {compute}")
        low = math.log(1e5)
        high = math.log(max_parameters)

        def objective(log_parameters: float, compute: float = compute) -> float:
            parameters = math.exp(log_parameters)
            tokens = compute / (6.0 * parameters)
            return amplitude * parameters ** (-alpha) + token_amplitude * tokens ** (-beta)

        log_parameters = _golden_section(objective, low, high)
        parameters = math.exp(log_parameters)
        tokens = compute / (6.0 * parameters)
        loss = offset + objective(log_parameters)
        estimates.append(
            {
                "compute": compute,
                "parameters": parameters,
                "tokens": tokens,
                "tokens_per_parameter": tokens / parameters,
                "predicted_loss": loss,
                "extrapolated": bool(
                    parameters > fit["parameters_max"] or tokens > fit["tokens_max"]
                ),
            }
        )
    return estimates


def _fit_axes(points: Sequence[ScalingPoint]) -> dict[str, dict[str, Any]]:
    fits: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        raw_values = [_axis_value(point, axis) for point in points]
        if any(value is None for value in raw_values):
            continue
        values = [float(value) for value in raw_values if value is not None]
        fits[axis] = fit_power_law(
            values,
            [point.loss for point in points],
        )
        fits[axis]["axis"] = axis
    return fits


def _predict(fit: Mapping[str, Any], x: float) -> float:
    return float(fit["E"]) + float(fit["A"]) * x ** (-float(fit["alpha"]))


def _select_holdout(
    points: Sequence[ScalingPoint], holdout_name: str | None
) -> ScalingPoint | None:
    if holdout_name is not None:
        for point in points:
            if point.name == holdout_name:
                return point
        raise ValueError(f"Holdout point not found: {holdout_name}")
    if len(points) < 5:
        return None
    return max(points, key=lambda point: point.parameters)


def _evaluate_holdout(
    points: Sequence[ScalingPoint], holdout_name: str | None
) -> dict[str, Any] | None:
    held = _select_holdout(points, holdout_name)
    if held is None:
        return None
    training = [point for point in points if point.name != held.name]
    if len(training) < 4:
        return {
            "name": held.name,
            "status": "skipped",
            "reason": "At least four non-holdout points are required",
        }
    held_values = {
        axis: _axis_value(held, axis) for axis in AXES if _axis_value(held, axis) is not None
    }
    details: dict[str, Any] = {}
    for axis, fit in _fit_axes(training).items():
        x = held_values.get(axis)
        if x is None:
            continue
        predicted = _predict(fit, x)
        absolute_error = abs(predicted - held.loss)
        details[axis] = {
            "observed": held.loss,
            "predicted": predicted,
            "absolute_error": absolute_error,
            "relative_error": absolute_error / max(abs(held.loss), 1e-12),
        }
    return {
        "name": held.name,
        "status": "evaluated",
        "observed_loss": held.loss,
        "parameters": held.parameters,
        "tokens": held.tokens,
        "fits": details,
    }


def build_scaling_report(
    points: Sequence[ScalingPoint],
    *,
    holdout: str | None = None,
    compute_budgets: Sequence[float] = (),
) -> dict[str, Any]:
    if len(points) < 4:
        raise ValueError("At least four scaling points are required")
    _require_unique_names(points)
    fits = _fit_axes(points)
    bivariate = fit_bivariate_power_law(points)
    budgets = [_number(budget, "compute budget", positive=True) for budget in compute_budgets]
    compute_optimal = estimate_compute_optimal(bivariate, budgets) if bivariate is not None else []
    report = {
        "format_version": FORMAT_VERSION,
        "created_utc": utc_now(),
        "git": git_metadata(),
        "settings": {
            "holdout": holdout,
            "compute_budgets": budgets,
            "axes": list(fits),
        },
        "points": [point.to_dict() for point in points],
        "fits": fits,
        "holdout": _evaluate_holdout(points, holdout),
        "bivariate": bivariate,
        "compute_optimal": compute_optimal,
        "limitations": [
            "Fits use validation loss from checkpoints; comparability requires the "
            "same data, tokenizer, sequence length and evaluation protocol.",
            "Single-axis fits assume one dominant scaling variable; the bivariate "
            "fit needs points with independently varied parameters and tokens.",
            "Compute is estimated as 6*N*D with total parameters; for MoE models "
            "training compute follows active parameters more closely.",
            "Holdout prediction and compute-optimal estimates are extrapolations and "
            "are reported without confidence intervals or significance testing.",
        ],
    }
    return report


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# AURORA scaling law report",
        "",
        f"Generated: {report['created_utc']}",
        "",
        "## Points",
        "",
        "| Name | Parameters | Active | Tokens | Loss | Compute |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in report["points"]:
        lines.append(
            f"| {_cell(point['name'])} | {_format_number(point['parameters'])} | "
            f"{_format_number(point['active_parameters'])} | "
            f"{_format_number(point['tokens'])} | {point['loss']:.4f} | "
            f"{_format_number(point['compute'])} |"
        )
    lines.extend(
        [
            "",
            "## Power-law fits",
            "",
            "Model: `L = E + A * x^-alpha`.",
            "",
            "| Axis | E | A | Alpha | R² | RMSE | Points | Range |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for axis, fit in report["fits"].items():
        lines.append(
            f"| {_cell(axis)} | {_format_number(fit['E'])} | "
            f"{_format_number(fit['A'])} | {fit['alpha']:.4f} | "
            f"{fit['r2']:.4f} | {_format_number(fit['rmse'])} | {fit['n']} | "
            f"{_format_number(fit['x_min'])} – {_format_number(fit['x_max'])} |"
        )
    lines.extend(["", "## Holdout", ""])
    holdout = report["holdout"]
    if holdout is None:
        lines.append("No holdout point was selected.")
    elif holdout["status"] == "skipped":
        lines.append(f"Skipped: {holdout['reason']}.")
    else:
        lines.extend(
            [
                f"Held out `{_cell(holdout['name'])}`: observed loss "
                f"{holdout['observed_loss']:.4f}, parameters "
                f"{_format_number(holdout['parameters'])}, tokens "
                f"{_format_number(holdout['tokens'])}.",
                "",
                "| Axis | Predicted | Observed | Absolute error | Relative error |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for axis, fit in holdout["fits"].items():
            lines.append(
                f"| {_cell(axis)} | {fit['predicted']:.4f} | {fit['observed']:.4f} | "
                f"{fit['absolute_error']:.4f} | {fit['relative_error']:.2%} |"
            )
    lines.extend(["", "## Bivariate fit", ""])
    bivariate = report["bivariate"]
    if bivariate is None:
        lines.append("No feasible bivariate fit.")
    else:
        lines.extend(
            [
                "Model: `L = E + A*N^-alpha + B*D^-beta`.",
                "",
                f"- E = {_format_number(bivariate['E'])}",
                f"- A = {_format_number(bivariate['A'])}, alpha = {bivariate['alpha']:.4f}",
                f"- B = {_format_number(bivariate['B'])}, beta = {bivariate['beta']:.4f}",
                f"- R² = {bivariate['r2']:.4f}, RMSE = {_format_number(bivariate['rmse'])}, "
                f"condition = {_format_number(bivariate['condition'])}",
                f"- Reliable: {bivariate['reliable']}",
            ]
        )
        if "warning" in bivariate:
            lines.append(f"- Warning: {bivariate['warning']}")
    lines.extend(["", "## Compute-optimal estimates", ""])
    if not report["compute_optimal"]:
        lines.append(
            "No compute-optimal estimates were produced; pass --compute-budget "
            "and ensure the bivariate fit is reliable."
        )
    else:
        lines.extend(
            [
                "| Compute | Parameters | Tokens | Tokens/param | Predicted loss | Extrapolated |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for estimate in report["compute_optimal"]:
            lines.append(
                f"| {_format_number(estimate['compute'])} | "
                f"{_format_number(estimate['parameters'])} | "
                f"{_format_number(estimate['tokens'])} | "
                f"{estimate['tokens_per_parameter']:.2f} | "
                f"{estimate['predicted_loss']:.4f} | {estimate['extrapolated']} |"
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit scaling laws from checkpoints or explicit scaling points"
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="NAME=PATH.pt",
        help="Checkpoint with validation metrics; repeat for multiple sizes",
    )
    parser.add_argument(
        "--points",
        default=None,
        help="JSON file with explicit scaling points",
    )
    parser.add_argument(
        "--holdout",
        default=None,
        help="Point name to hold out (default: largest parameters)",
    )
    parser.add_argument(
        "--compute-budget",
        action="append",
        type=float,
        default=[],
        dest="compute_budgets",
        help="Training compute budget in FLOPs; repeat for multiple budgets",
    )
    parser.add_argument("--output", default="reports/scaling.json")
    parser.add_argument("--markdown", default="", help="Optional Markdown output path")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        points = parse_point_args(args.point)
        if args.points:
            points.extend(load_points_json(args.points))
        if not points:
            raise ValueError("At least one --point NAME=PATH.pt or --points file is required")
        report = build_scaling_report(
            points,
            holdout=args.holdout,
            compute_budgets=args.compute_budgets,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not args.output and not args.markdown:
        raise SystemExit("At least one of --output or --markdown is required")

    for axis, fit in report["fits"].items():
        print(
            f"{axis}: L = {fit['E']:.4f} + {fit['A']:.4g} * x^-{fit['alpha']:.4f} "
            f"| R2 {fit['r2']:.4f} | RMSE {fit['rmse']:.4f}"
        )
    holdout = report["holdout"]
    if holdout is not None and holdout["status"] == "evaluated":
        for axis, detail in holdout["fits"].items():
            print(
                f"holdout {holdout['name']} {axis}: predicted {detail['predicted']:.4f} "
                f"vs observed {detail['observed']:.4f} "
                f"({detail['relative_error']:.2%})"
            )
    for estimate in report["compute_optimal"]:
        print(
            f"compute {estimate['compute']:.3g}: N {estimate['parameters']:.3g}, "
            f"D {estimate['tokens']:.3g}, D/N "
            f"{estimate['tokens_per_parameter']:.2f}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json -> {output}")
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
        print(f"report -> {markdown}")


if __name__ == "__main__":
    main()
