from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.data.contamination import check_contamination, main


def _write_text(path: Path, *documents: str) -> None:
    path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")


def test_contamination_detects_shared_ngram(tmp_path: Path) -> None:
    shared = "alpha beta gamma delta epsilon zeta"
    corpus = tmp_path / "corpus.txt"
    benchmark = tmp_path / "benchmark.txt"
    _write_text(corpus, f"{shared} one two three", "unrelated corpus document")
    _write_text(benchmark, f"{shared} extra", "another benchmark document")

    report = check_contamination([corpus], [benchmark], ngram_size=4)
    summary = report["summary"]
    assert summary["benchmark_documents"] == 2
    assert summary["benchmark_documents_with_overlap"] == 1
    assert summary["matched_ngram_occurrences"] == 3
    assert summary["unique_matched_ngrams"] == 3
    assert report["benchmarks"][0]["matched_documents"] == 1
    assert report["benchmarks"][0]["contamination_fraction"] == 0.5
    assert report["corpus"][0]["documents_with_overlap"] == 1
    assert report["examples"][0]["ngram"] == "alpha beta gamma delta"
    assert report["examples"][0]["corpus"]["document_index"] == 0


def test_contamination_ignores_unrelated_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    benchmark = tmp_path / "benchmark.txt"
    _write_text(corpus, "red green blue yellow orange purple")
    _write_text(benchmark, "one two three four five six")

    report = check_contamination([corpus], [benchmark], ngram_size=4)
    assert report["summary"]["benchmark_documents_with_overlap"] == 0
    assert report["summary"]["matched_ngram_occurrences"] == 0
    assert report["examples"] == []


def test_contamination_supports_jsonl(tmp_path: Path) -> None:
    phrase = "red green blue yellow orange purple"
    corpus = tmp_path / "corpus.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    corpus.write_text(json.dumps({"text": phrase}) + "\n", encoding="utf-8")
    benchmark.write_text(json.dumps({"text": phrase + " extra"}) + "\n", encoding="utf-8")

    report = check_contamination([corpus], [benchmark], ngram_size=4)
    assert report["summary"]["matched_ngram_occurrences"] == 3
    assert report["benchmarks"][0]["format"] == "jsonl"
    assert report["examples"][0]["corpus"]["line"] == 1


def test_contamination_cli_writes_report_and_fails_on_match(tmp_path: Path) -> None:
    phrase = "alpha beta gamma delta epsilon"
    corpus = tmp_path / "corpus.txt"
    benchmark = tmp_path / "benchmark.txt"
    output = tmp_path / "reports" / "contamination.json"
    _write_text(corpus, phrase)
    _write_text(benchmark, phrase)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--corpus",
                str(corpus),
                "--benchmark",
                str(benchmark),
                "--ngram-size",
                "3",
                "--output",
                str(output),
                "--fail-on-match",
            ]
        )
    assert error.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["matched_ngram_occurrences"] == 3
    assert report["settings"]["ngram_size"] == 3


def test_contamination_rejects_invalid_ngram_size(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    benchmark = tmp_path / "benchmark.txt"
    _write_text(corpus, "alpha beta gamma")
    _write_text(benchmark, "alpha beta gamma")

    with pytest.raises(SystemExit, match="ngram_size"):
        main(
            [
                "--corpus",
                str(corpus),
                "--benchmark",
                str(benchmark),
                "--ngram-size",
                "0",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
