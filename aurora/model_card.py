from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from aurora.architecture.factory import build_model
from aurora.config import ExperimentConfig, load_config
from aurora.training.checkpoint import load_checkpoint, validate_checkpoint
from aurora.utils import human_int, parameter_counts

DEFAULT_DESCRIPTION = (
    "AURORA is a non-Transformer research architecture with a multi-level recurrent "
    "core, associative memory and sparse expert routing. This card describes a "
    "prototype configuration or checkpoint and must not be read as evidence of "
    "quality or superiority over other architectures."
)

DEFAULT_INTENDED_USE = (
    "Research on recurrent and associative language models, reproducible ablations, "
    "quantization studies and inference-runtime prototyping."
)

DEFAULT_OUT_OF_SCOPE = (
    "Production deployment, safety-critical use, medical or legal advice, autonomous "
    "decision making, or any claim of parity with existing language models without "
    "equal-data, equal-compute evaluations."
)

DEFAULT_LIMITATIONS = (
    "The architecture has not demonstrated superiority over Transformer baselines.",
    "Public benchmark results and a fair equal-compute comparison are not available yet.",
    "The 80B configuration is a parameter-count target and has not been trained.",
    "Document-boundary state resets and long-context behavior are still under research.",
    "Fast ternary kernels and a memory-mapped expert runtime are not implemented.",
    "Safety alignment, red teaming and data-licensing review are incomplete.",
)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt_int(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "n/a"
    return f"{value:,}"


def _fmt_float(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.6g}"


def _fmt_bool(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "n/a"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_parameters(config: ExperimentConfig) -> tuple[int, int]:
    with torch.device("meta"):
        model = build_model(config.model)
    return parameter_counts(model)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _render_summary(
    config: ExperimentConfig,
    *,
    name: str,
    license_name: str,
    checkpoint: Mapping[str, Any] | None,
    checkpoint_path: str | Path | None,
) -> list[str]:
    total, active = _count_parameters(config)
    share = 100.0 * active / max(total, 1)
    checkpoint_sha = ""
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.is_file():
            checkpoint_sha = _sha256_file(path)
    checkpoint_name = str(checkpoint_path) if checkpoint_path is not None else "none"
    format_version = checkpoint.get("format_version") if checkpoint else None
    step = checkpoint.get("step") if checkpoint else None
    qat_active = checkpoint.get("qat_active") if checkpoint else None
    if checkpoint is None:
        status = "Configuration only (no trained checkpoint)"
    elif qat_active:
        status = "Checkpoint with QAT active"
    else:
        status = "Checkpoint (training status not independently verified)"

    return [
        "## Model summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model name | {_cell(name)} |",
        f"| Architecture | `{_cell(config.model.architecture)}` |",
        f"| Status | {_cell(status)} |",
        f"| Total parameters | {_fmt_int(total)} ({human_int(total)}) |",
        f"| Active parameters | {_fmt_int(active)} ({human_int(active)}) |",
        f"| Active share | {share:.1f}% |",
        f"| License | {_cell(license_name)} |",
        f"| Checkpoint | `{_cell(checkpoint_name)}` |",
        f"| Checkpoint SHA-256 | `{_cell(checkpoint_sha or 'n/a')}` |",
        f"| Checkpoint format | {_cell(format_version if format_version is not None else 'n/a')} |",
        f"| Training step | {_fmt_int(step)} |",
        f"| QAT active | {_fmt_bool(qat_active)} |",
        "",
    ]


def _render_architecture(config: ExperimentConfig) -> list[str]:
    model = config.model
    return [
        "## Architecture",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Vocabulary size | {_fmt_int(model.vocab_size)} |",
        f"| d_model | {_fmt_int(model.d_model)} |",
        f"| Layers | {_fmt_int(model.n_layers)} |",
        f"| Recurrent levels | {_fmt_int(model.n_levels)} |",
        f"| d_state | {_fmt_int(model.d_state)} |",
        f"| Attention heads | {_fmt_int(model.attention_heads)} |",
        f"| FFN hidden | {_fmt_int(model.ffn_hidden)} |",
        f"| Associative memory | {_fmt_bool(model.use_memory)} |",
        f"| Memory heads | {_fmt_int(model.memory_heads)} |",
        f"| MoE enabled | {_fmt_bool(model.use_moe)} |",
        f"| Experts per layer | {_fmt_int(model.n_experts)} |",
        f"| Top-k experts | {_fmt_int(model.top_k)} |",
        f"| Expert hidden | {_fmt_int(model.expert_hidden)} |",
        f"| Dropout | {_fmt_float(model.dropout)} |",
        f"| Tied embeddings | {_fmt_bool(model.tie_embeddings)} |",
        f"| LayerScale init | {_fmt_float(model.layer_scale_init)} |",
        f"| Memory loss coefficient | {_fmt_float(model.memory_loss_coef)} |",
        f"| Consistency loss coefficient | {_fmt_float(model.consistency_loss_coef)} |",
        "",
    ]


def _render_training_config(config: ExperimentConfig) -> list[str]:
    training = config.training
    train_data = ", ".join(training.train_paths) if training.train_paths else training.train_path
    lines = [
        "## Training",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Train data | `{_cell(train_data or 'not specified')}` |",
        f"| Validation data | `{_cell(training.val_path or 'not specified')}` |",
        f"| Sequence length | {_fmt_int(training.seq_len)} |",
        f"| Token dtype | `{_cell(training.dtype)}` |",
        f"| Batch size | {_fmt_int(training.batch_size)} |",
        f"| Gradient accumulation | {_fmt_int(training.grad_accum_steps)} |",
        f"| Max steps | {_fmt_int(training.max_steps)} |",
        f"| Max tokens | {_fmt_int(training.max_tokens) if training.max_tokens else 'unlimited'} |",
        f"| Learning rate | {_fmt_float(training.learning_rate)} |",
        f"| Warmup steps | {_fmt_int(training.warmup_steps)} |",
        f"| Minimum LR ratio | {_fmt_float(training.min_lr_ratio)} |",
        f"| Weight decay | {_fmt_float(training.weight_decay)} |",
        f"| Gradient clipping | {_fmt_float(training.grad_clip)} |",
        f"| AMP dtype | `{_cell(training.dtype_amp)}` |",
        f"| Seed | {_fmt_int(training.seed)} |",
        f"| QAT enabled | {_fmt_bool(training.qat_enabled)} |",
        "",
    ]
    return lines


def _render_provenance(checkpoint: Mapping[str, Any] | None) -> list[str]:
    if checkpoint is None:
        return [
            "## Provenance",
            "",
            "No checkpoint provenance is available because this card was generated "
            "from a configuration only.",
            "",
        ]
    run_metadata = _mapping(checkpoint.get("run_metadata"))
    metrics = _mapping(checkpoint.get("metrics"))
    git = _mapping(run_metadata.get("git"))
    environment = _mapping(run_metadata.get("environment"))
    dataset = _mapping(run_metadata.get("dataset"))
    sources = _list(dataset.get("sources"))
    last_train = _mapping(metrics.get("last_train"))
    last_validation = _mapping(metrics.get("last_validation"))

    lines = [
        "## Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run ID | `{_cell(run_metadata.get('run_id') or 'n/a')}` |",
        f"| Started (UTC) | `{_cell(run_metadata.get('started_utc') or 'n/a')}` |",
        f"| Resume count | {_fmt_int(run_metadata.get('resume_count'))} |",
        f"| Git commit | `{_cell(git.get('commit') or 'n/a')}` |",
        f"| Git branch | `{_cell(git.get('branch') or 'n/a')}` |",
        f"| Git dirty | {_fmt_bool(git.get('dirty'))} |",
        f"| Python | `{_cell(environment.get('python') or 'n/a')}` |",
        f"| PyTorch | `{_cell(environment.get('torch') or 'n/a')}` |",
        f"| Device | `{_cell(environment.get('device') or 'n/a')}` |",
        f"| Dataset version | `{_cell(dataset.get('version') or 'n/a')}` |",
        f"| Dataset fingerprint | `{_cell(dataset.get('fingerprint') or 'n/a')}` |",
        f"| Dataset verified | {_fmt_bool(dataset.get('verified'))} |",
        f"| Tokens seen | {_fmt_int(metrics.get('tokens_seen'))} |",
        f"| Last train loss | {_fmt_float(last_train.get('lm_loss'))} |",
        f"| Last validation loss | {_fmt_float(last_validation.get('loss'))} |",
        f"| Last validation perplexity | {_fmt_float(last_validation.get('perplexity'))} |",
        "",
    ]
    if sources:
        lines.extend(
            [
                "### Data sources",
                "",
                "| Role | Path | Tokens | SHA-256 | Dataset version |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            lines.append(
                f"| `{_cell(source.get('role') or 'n/a')}` | "
                f"`{_cell(source.get('path') or 'n/a')}` | "
                f"{_fmt_int(source.get('tokens'))} | "
                f"`{_cell(source.get('sha256') or 'n/a')}` | "
                f"`{_cell(source.get('dataset_version') or 'n/a')}` |"
            )
        lines.append("")
    return lines


def _render_evaluation(evaluation: Mapping[str, Any] | None) -> list[str]:
    if evaluation is None:
        return [
            "## Evaluation",
            "",
            "No evaluation report was provided. This card must not be used to claim "
            "quality, parity or superiority.",
            "",
        ]
    results = _mapping(evaluation.get("results"))
    settings = _mapping(evaluation.get("settings"))
    evaluation_meta = _mapping(evaluation.get("evaluation"))
    git = _mapping(evaluation_meta.get("git"))
    lines = [
        "## Evaluation",
        "",
        f"- Report created (UTC): `{_cell(evaluation.get('created_utc') or 'n/a')}`",
        f"- Git commit: `{_cell(git.get('commit') or 'n/a')}`",
        f"- Sequence length: {_fmt_int(settings.get('seq_len'))}",
        f"- Batch size: {_fmt_int(settings.get('batch_size'))}",
        f"- Device: `{_cell(settings.get('device') or 'n/a')}`",
        "",
        "| Split | Tokens | Loss | Perplexity | Dataset version |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    if not results:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
    for split, result in sorted(results.items()):
        if not isinstance(result, Mapping):
            continue
        lines.append(
            f"| `{_cell(split)}` | {_fmt_int(result.get('tokens'))} | "
            f"{_fmt_float(result.get('loss'))} | {_fmt_float(result.get('perplexity'))} | "
            f"`{_cell(result.get('dataset_version') or 'n/a')}` |"
        )
    lines.extend(
        [
            "",
            "Perplexity is dataset- and tokenizer-specific. It is not a public benchmark "
            "and is not comparable across different data, tokenizers or evaluation settings.",
            "",
        ]
    )
    return lines


def _render_sections(
    *,
    intended_use: str,
    out_of_scope: str,
    limitations: Sequence[str],
    has_checkpoint: bool,
) -> list[str]:
    lines = [
        "## Intended use",
        "",
        intended_use,
        "",
        "## Out of scope",
        "",
        out_of_scope,
        "",
        "## Limitations and risks",
        "",
    ]
    for item in limitations:
        lines.append(f"- {item}")
    lines.append(
        "- This card is generated from configuration and run provenance; it does not "
        "independently prove data licensing, deduplication quality or benchmark "
        "contamination checks."
    )
    lines.extend(["", "## Reproducibility", ""])
    if has_checkpoint:
        lines.append(
            "Training, evaluation and card generation are reproducible from the repository "
            "at the git commit recorded above. The exact tokenizer, dataset version and "
            "checksums must be preserved with the checkpoint."
        )
    else:
        lines.append(
            "This draft is reproducible from the configuration in the repository. It will be "
            "finalized after a trained checkpoint and evaluation report are available."
        )
    lines.append("")
    return lines


def render_model_card(
    config: ExperimentConfig,
    *,
    name: str = "AURORA model",
    description: str = "",
    license_name: str = "Not determined",
    checkpoint: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    evaluation: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
) -> str:
    """Render a Markdown model card from config, checkpoint metadata and evaluation."""
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if evaluation is not None and not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be a mapping")
    if evaluation is not None and not isinstance(evaluation.get("results"), Mapping):
        raise ValueError("Evaluation report must contain a results mapping")

    all_limitations = list(DEFAULT_LIMITATIONS)
    for item in limitations:
        text = str(item).strip()
        if text and text not in all_limitations:
            all_limitations.append(text)
    if checkpoint is None:
        all_limitations.insert(0, "No trained weights are associated with this card.")
    if evaluation is None:
        all_limitations.insert(0, "No evaluation results are associated with this card.")
    if checkpoint is not None and checkpoint.get("qat_active") is True:
        all_limitations.insert(
            0,
            "QAT results require an FP baseline on the same checkpoint and data before "
            "any quality recovery claim.",
        )

    lines = [f"# Model card: {name}", ""]
    lines.extend(
        [
            "> Research prototype. Do not interpret this card as evidence of quality, "
            "safety or superiority over Transformer or any other architecture.",
            "",
        ]
    )
    if description:
        lines.extend([description, ""])
    lines.extend(
        _render_summary(
            config,
            name=name,
            license_name=license_name,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
    )
    lines.extend(_render_architecture(config))
    lines.extend(_render_training_config(config))
    lines.extend(_render_provenance(checkpoint))
    lines.extend(_render_evaluation(evaluation))
    lines.extend(
        _render_sections(
            intended_use=DEFAULT_INTENDED_USE,
            out_of_scope=DEFAULT_OUT_OF_SCOPE,
            limitations=all_limitations,
            has_checkpoint=checkpoint is not None,
        )
    )
    return "\n".join(lines)


def _load_evaluation(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Evaluation report must be a JSON object")
    if not isinstance(data.get("results"), dict):
        raise ValueError("Evaluation report must contain a results mapping")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Markdown model card")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Checkpoint with config and run metadata")
    source.add_argument("--config", help="YAML config for a configuration-only card")
    parser.add_argument("--evaluation", default="", help="Optional evaluation JSON report")
    parser.add_argument("--output", default="", help="Output Markdown path (default: stdout)")
    parser.add_argument("--name", default="", help="Model name")
    parser.add_argument("--description", default="", help="Short model description")
    parser.add_argument("--license", dest="license_name", default="Not determined")
    parser.add_argument(
        "--limitation",
        action="append",
        default=[],
        help="Additional limitation bullet (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.checkpoint:
            checkpoint = load_checkpoint(args.checkpoint, validate=False)
            validate_checkpoint(checkpoint, check_finite=False)
            config = ExperimentConfig.from_dict(checkpoint["config"])
            checkpoint_path: str | None = args.checkpoint
            default_name = Path(args.checkpoint).stem
        else:
            checkpoint = None
            config = load_config(args.config)
            checkpoint_path = None
            default_name = f"AURORA {Path(args.config).stem}"
        evaluation = _load_evaluation(args.evaluation) if args.evaluation else None
        card = render_model_card(
            config,
            name=args.name or default_name,
            description=args.description or DEFAULT_DESCRIPTION,
            license_name=args.license_name,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            evaluation=evaluation,
            limitations=args.limitation,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(card, encoding="utf-8")
        print(f"model card -> {output}")
    else:
        print(card, end="")


if __name__ == "__main__":
    main()
