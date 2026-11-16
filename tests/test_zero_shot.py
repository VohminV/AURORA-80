from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aurora.benchmarks.zero_shot import (
    MultipleChoiceSample,
    evaluate_task,
    load_task,
    main,
    parse_tasks,
    score_choice,
)
from aurora.config import ExperimentConfig, ModelConfig, TrainingConfig
from aurora.data.dataset import MemmapTokenDataset
from aurora.tokenizer.tokenizer import ByteTokenizer
from aurora.training.trainer import Trainer


class _BiasedModel(torch.nn.Module):
    def __init__(self, token_id: int, vocab_size: int = 300) -> None:
        super().__init__()
        self.token_id = token_id
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, state: object = None) -> SimpleNamespace:
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
        logits[:, :, self.token_id] = 5.0
        return SimpleNamespace(logits=logits)


def _tiny_config(tmp_path: Path, vocab_size: int = 260) -> ExperimentConfig:
    model = ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_levels=2,
        d_state=16,
        memory_heads=2,
        n_experts=4,
        top_k=2,
        expert_hidden=32,
        dropout=0.0,
    )
    training = TrainingConfig(
        seq_len=8,
        batch_size=2,
        grad_accum_steps=1,
        max_steps=2,
        warmup_steps=0,
        learning_rate=1e-3,
        eval_interval=0,
        log_interval=1,
        save_interval=2,
        output_dir=str(tmp_path / "zero_shot_run"),
        device="cpu",
        dtype_amp="fp32",
        seed=7,
    )
    return ExperimentConfig(model=model, training=training)


def _train_tiny_checkpoint(tmp_path: Path, config: ExperimentConfig) -> Path:
    train_path = tmp_path / "train.bin"
    rng = np.random.default_rng(11)
    rng.integers(0, config.model.vocab_size, size=512, dtype=np.uint16).tofile(train_path)
    Trainer(config, MemmapTokenDataset(train_path, config.training.seq_len)).fit()
    return Path(config.training.output_dir) / "latest.pt"


def _write_tasks(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_parse_tasks_validates_names_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    _write_tasks(path, {"context": "q", "choices": ["a", "b"], "answer": 0})
    assert parse_tasks([f"mmlu={path}"]) == [("mmlu", path)]
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_tasks([str(path)])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_tasks([f"mmlu={path}", f"mmlu={path}"])
    with pytest.raises(ValueError, match="not found"):
        parse_tasks([f"mmlu={tmp_path / 'missing.jsonl'}"])


def test_load_task_validates_jsonl_fields(tmp_path: Path) -> None:
    path = tmp_path / "task.jsonl"
    _write_tasks(
        path,
        {"context": "question", "choices": ["a", "b"], "answer": 1},
        {"context": "", "choices": ["c"], "answer": 0},
    )
    samples = load_task(path, limit=1)
    assert samples == [MultipleChoiceSample("question", ("a", "b"), 1)]

    cases = [
        ('{"context": "q", "choices": [], "answer": 0}', "non-empty list"),
        ('{"context": "q", "choices": ["a"], "answer": 2}', "outside"),
        ('{"context": "q", "choices": ["a"], "answer": true}', "integer"),
        ('{"context": 1, "choices": ["a"], "answer": 0}', "must be a string"),
        ("{broken", "invalid JSON"),
    ]
    for line, message in cases:
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_task(path)


def test_score_choice_prefers_biased_token() -> None:
    tokenizer = ByteTokenizer()
    model = _BiasedModel(tokenizer.encode("A")[0])
    context_ids = tokenizer.encode("question ")
    score_a = score_choice(
        model,
        context_ids,
        tokenizer.encode("A"),
        max_length=16,
        device=torch.device("cpu"),
        dtype_amp="fp32",
    )
    score_b = score_choice(
        model,
        context_ids,
        tokenizer.encode("B"),
        max_length=16,
        device=torch.device("cpu"),
        dtype_amp="fp32",
    )
    assert score_a is not None and score_b is not None
    assert score_a[0] > score_b[0]
    normalized = score_choice(
        model,
        context_ids,
        tokenizer.encode("A"),
        max_length=16,
        device=torch.device("cpu"),
        dtype_amp="fp32",
        normalize_by_length=True,
    )
    assert normalized is not None
    assert normalized[0] == pytest.approx(score_a[0])


def test_evaluate_task_selects_highest_scoring_choice() -> None:
    tokenizer = ByteTokenizer()
    model = _BiasedModel(tokenizer.encode("A")[0])
    samples = [MultipleChoiceSample("question ", ("A", "B"), 0)]

    result, predictions = evaluate_task(
        model,
        tokenizer,
        samples,
        vocab_size=300,
        max_length=16,
        device=torch.device("cpu"),
        dtype_amp="fp32",
    )

    assert result["evaluated"] == 1
    assert result["skipped"] == 0
    assert result["correct"] == 1
    assert result["accuracy"] == 1.0
    assert predictions[0]["prediction"] == 0
    assert predictions[0]["correct"] is True


def test_zero_shot_cli_writes_report_and_predictions(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    checkpoint = _train_tiny_checkpoint(tmp_path, config)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_tasks(
        first,
        {"context": "hello ", "choices": ["a", "b"], "answer": 0},
        {"context": "hello ", "choices": ["c", "d"], "answer": 1},
    )
    _write_tasks(
        second,
        {"context": "code ", "choices": ["x", "y"], "answer": 1},
    )
    output = tmp_path / "reports" / "zero_shot.json"
    predictions_path = tmp_path / "reports" / "predictions.jsonl"

    main(
        [
            "--checkpoint",
            str(checkpoint),
            "--task",
            f"first={first}",
            "--task",
            f"second={second}",
            "--limit",
            "1",
            "--max-length",
            "8",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--predictions",
            str(predictions_path),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checkpoint"]["step"] == 2
    assert report["checkpoint"]["architecture"] == "aurora"
    assert report["checkpoint"]["total_parameters"] > 0
    assert report["evaluation"]["environment"]["device"] == "cpu"
    assert report["settings"]["limit"] == 1
    assert report["settings"]["max_length"] == 8
    assert report["settings"]["tasks"] == {"first": str(first), "second": str(second)}
    assert report["tasks"]["first"]["evaluated"] == 1
    assert report["tasks"]["second"]["evaluated"] == 1
    assert 0.0 <= report["tasks"]["first"]["accuracy"] <= 1.0

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    assert [row["task"] for row in rows] == ["first", "second"]
    assert all("scores" in row and "prediction" in row for row in rows)


def test_zero_shot_cli_rejects_incompatible_tokenizer(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, vocab_size=64)
    checkpoint = _train_tiny_checkpoint(tmp_path, config)
    task = tmp_path / "task.jsonl"
    _write_tasks(task, {"context": "hello ", "choices": ["a", "b"], "answer": 0})

    with pytest.raises(SystemExit, match="vocabulary"):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--task",
                f"task={task}",
                "--device",
                "cpu",
            ]
        )


def test_zero_shot_cli_rejects_empty_limit(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    checkpoint = _train_tiny_checkpoint(tmp_path, config)
    task = tmp_path / "task.jsonl"
    _write_tasks(task, {"context": "q", "choices": ["a", "b"], "answer": 0})

    with pytest.raises(SystemExit, match="non-negative"):
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--task",
                f"task={task}",
                "--limit",
                "-1",
            ]
        )
