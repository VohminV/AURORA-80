from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from aurora.inference import inference_autocast, load_model_from_checkpoint, model_output
from aurora.provenance import environment_metadata, git_metadata, utc_now
from aurora.tokenizer import load_tokenizer
from aurora.tokenizer.tokenizer import BaseTokenizer
from aurora.training.checkpoint import load_checkpoint
from aurora.utils import resolve_device


@dataclass(frozen=True)
class MultipleChoiceSample:
    context: str
    choices: tuple[str, ...]
    answer: int


def parse_tasks(items: Sequence[str]) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"Task must be NAME=PATH, got: {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name:
            raise ValueError(f"Task name is empty in: {item!r}")
        if name in seen:
            raise ValueError(f"Duplicate task name: {name}")
        if not path.is_file():
            raise ValueError(f"Task file not found: {path}")
        seen.add(name)
        tasks.append((name, path))
    if not tasks:
        raise ValueError("At least one --task NAME=PATH is required")
    return tasks


def load_task(
    path: Path,
    context_field: str = "context",
    choices_field: str = "choices",
    answer_field: str = "answer",
    limit: int = 0,
) -> list[MultipleChoiceSample]:
    samples: list[MultipleChoiceSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            context = record.get(context_field)
            if not isinstance(context, str):
                raise ValueError(f"{path}:{line_number}: field {context_field!r} must be a string")
            raw_choices = record.get(choices_field)
            if (
                not isinstance(raw_choices, list)
                or not raw_choices
                or not all(isinstance(choice, str) and choice for choice in raw_choices)
            ):
                raise ValueError(
                    f"{path}:{line_number}: field {choices_field!r} must be a non-empty "
                    "list of non-empty strings"
                )
            answer = record.get(answer_field)
            if not isinstance(answer, int) or isinstance(answer, bool):
                raise ValueError(f"{path}:{line_number}: field {answer_field!r} must be an integer")
            if not 0 <= answer < len(raw_choices):
                raise ValueError(
                    f"{path}:{line_number}: answer {answer} is outside [0, {len(raw_choices) - 1}]"
                )
            samples.append(
                MultipleChoiceSample(
                    context=context,
                    choices=tuple(raw_choices),
                    answer=answer,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"{path}: no multiple-choice samples found")
    return samples


def _check_vocab(token_ids: Sequence[int], vocab_size: int) -> None:
    if token_ids and max(token_ids) >= vocab_size:
        raise ValueError(
            f"Tokenizer produced token id {max(token_ids)} but the model vocabulary "
            f"has {vocab_size} entries; use a compatible tokenizer"
        )


def _truncate(
    context_ids: list[int], choice_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    if len(context_ids) + len(choice_ids) <= max_length:
        return context_ids, choice_ids
    overflow = len(context_ids) + len(choice_ids) - max_length
    dropped_context = min(overflow, len(context_ids))
    context_ids = context_ids[dropped_context:]
    overflow -= dropped_context
    if overflow:
        choice_ids = choice_ids[overflow:]
    return context_ids, choice_ids


@torch.no_grad()
def score_choice(
    model: torch.nn.Module,
    context_ids: Sequence[int],
    choice_ids: Sequence[int],
    *,
    max_length: int,
    device: torch.device,
    dtype_amp: str,
    normalize_by_length: bool = False,
) -> tuple[float, int] | None:
    context = list(context_ids)
    choice = list(choice_ids)
    context, choice = _truncate(context, choice, max_length)
    scored = len(choice) - (0 if context else 1)
    if scored <= 0:
        return None
    input_ids = torch.tensor([context + choice], dtype=torch.long, device=device)
    with inference_autocast(device, dtype_amp):
        logits = model_output(model, input_ids).logits
    log_probs = F.log_softmax(logits.float()[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    start = max(len(context) - 1, 0)
    token_log_probs = log_probs[:, start:, :].gather(-1, targets[:, start:].unsqueeze(-1))
    total = float(token_log_probs.sum())
    if normalize_by_length:
        total /= scored
    return total, scored


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: BaseTokenizer,
    samples: Sequence[MultipleChoiceSample],
    *,
    vocab_size: int,
    max_length: int,
    device: torch.device,
    dtype_amp: str,
    normalize_by_length: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct = 0
    evaluated = 0
    skipped = 0
    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        context_ids = tokenizer.encode(sample.context)
        _check_vocab(context_ids, vocab_size)
        scores: list[float] = []
        scorable = True
        for choice in sample.choices:
            choice_ids = tokenizer.encode(choice)
            _check_vocab(choice_ids, vocab_size)
            score = score_choice(
                model,
                context_ids,
                choice_ids,
                max_length=max_length,
                device=device,
                dtype_amp=dtype_amp,
                normalize_by_length=normalize_by_length,
            )
            if score is None:
                scorable = False
                break
            scores.append(score[0])
        if not scorable:
            skipped += 1
            continue
        prediction = max(range(len(scores)), key=scores.__getitem__)
        is_correct = prediction == sample.answer
        correct += int(is_correct)
        evaluated += 1
        predictions.append(
            {
                "index": index,
                "answer": sample.answer,
                "prediction": prediction,
                "correct": is_correct,
                "scores": scores,
            }
        )
    result: dict[str, Any] = {
        "samples": len(samples),
        "evaluated": evaluated,
        "skipped": skipped,
        "correct": correct,
        "accuracy": correct / evaluated if evaluated else 0.0,
        "choices": sum(len(sample.choices) for sample in samples),
    }
    return result, predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot multiple-choice evaluation for AURORA checkpoints"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="JSONL file with context, choices and answer; repeat for multiple tasks",
    )
    parser.add_argument("--tokenizer", choices=["byte", "sentencepiece"], default="byte")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--context-field", default="context")
    parser.add_argument("--choices-field", default="choices")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Maximum tokens per context+choice; 0 uses the training sequence length",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every sample")
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--predictions",
        default="",
        help="Optional JSONL file with per-sample predictions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        tasks = parse_tasks(args.task)
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_model)
        loaded = [
            (
                name,
                load_task(
                    path, args.context_field, args.choices_field, args.answer_field, args.limit
                ),
            )
            for name, path in tasks
        ]
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.max_length < 0 or args.limit < 0:
        raise SystemExit("max_length and limit must be non-negative")
    device = resolve_device(args.device)
    model, config = load_model_from_checkpoint(args.checkpoint, device=device)
    max_length = args.max_length or config.training.seq_len
    if max_length <= 0:
        raise SystemExit("max_length must be positive")

    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name, samples in loaded:
        try:
            result, predictions = evaluate_task(
                model,
                tokenizer,
                samples,
                vocab_size=config.model.vocab_size,
                max_length=max_length,
                device=device,
                dtype_amp=config.training.dtype_amp,
                normalize_by_length=args.normalize_by_length,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        results[name] = result
        prediction_rows.extend({"task": name, **row} for row in predictions)
        print(
            f"{name}: accuracy {result['accuracy']:.4f} | "
            f"{result['correct']}/{result['evaluated']} | skipped {result['skipped']}"
        )

    if args.predictions:
        predictions_path = Path(args.predictions)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.output:
        checkpoint = load_checkpoint(args.checkpoint, validate=False)
        run_metadata = checkpoint.get("run_metadata")
        if not isinstance(run_metadata, dict):
            run_metadata = None
        report = {
            "created_utc": utc_now(),
            "run_metadata": run_metadata,
            "checkpoint": {
                "path": str(args.checkpoint),
                "step": checkpoint.get("step"),
                "qat_active": checkpoint.get("qat_active"),
                "architecture": config.model.architecture,
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "run_id": run_metadata.get("run_id") if run_metadata else None,
                "metrics": checkpoint.get("metrics"),
            },
            "evaluation": {
                "git": git_metadata(),
                "environment": environment_metadata(device),
            },
            "settings": {
                "tasks": {name: str(path) for name, path in tasks},
                "tokenizer": args.tokenizer,
                "tokenizer_model": args.tokenizer_model,
                "context_field": args.context_field,
                "choices_field": args.choices_field,
                "answer_field": args.answer_field,
                "max_length": max_length,
                "limit": args.limit,
                "normalize_by_length": args.normalize_by_length,
                "device": str(device),
                "dtype_amp": config.training.dtype_amp,
            },
            "tasks": results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {output}")


if __name__ == "__main__":
    main()
