from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.data.corpus_stats import analyze_corpus, main


def _write_text(path: Path, *documents: str) -> None:
    path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")


def test_corpus_stats_text_files(tmp_path: Path) -> None:
    ru = tmp_path / "ru.txt"
    en = tmp_path / "en.txt"
    code = tmp_path / "code.py"
    _write_text(ru, "привет мир", "второй документ")
    _write_text(en, "hello world")
    code.write_text("def hello():\n    return 1\n", encoding="utf-8")

    report = analyze_corpus([ru, en, code])

    domains = report["domains"]
    assert domains["ru"]["documents"] == 2
    assert domains["ru"]["words"] == 4
    assert domains["ru"]["script_fractions"] == {"cyrillic": 1.0}
    assert domains["ru"]["language_hint"] == "cyrillic"
    assert domains["en"]["language_hint"] == "latin"
    assert domains["code"]["language_hint"] == "code"
    assert domains["code"]["code_line_fraction"] == 1.0

    summary = report["summary"]
    assert summary["documents"] == 4
    assert summary["words"] == 10
    assert report["inputs"][0]["format"] == "text"
    assert report["format_version"] == 1
    assert report["limitations"]


def test_corpus_stats_jsonl_domain_field(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"text": "first russian document", "source": "ru"},
        {"text": "second russian document", "source": "ru"},
        {"text": "an english document", "source": "en"},
    ]
    corpus.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = analyze_corpus([corpus], domain_field="source")

    assert report["domains"]["ru"]["documents"] == 2
    assert report["domains"]["ru"]["words"] == 6
    assert report["domains"]["en"]["documents"] == 1
    assert report["inputs"][0]["format"] == "jsonl"


def test_corpus_stats_max_documents_limits_work(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    _write_text(corpus, "one document", "two document", "three document")

    report = analyze_corpus([corpus], max_documents=2)

    assert report["summary"]["documents"] == 2
    assert report["inputs"][0]["documents"] == 2


def test_corpus_stats_rejects_invalid_settings(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    _write_text(corpus, "a document")

    with pytest.raises(ValueError, match="format"):
        analyze_corpus([corpus], input_format="xml")
    with pytest.raises(ValueError, match="chunk_size"):
        analyze_corpus([corpus], chunk_size=0)
    with pytest.raises(ValueError, match="code_line_threshold"):
        analyze_corpus([corpus], code_line_threshold=1.5)
    with pytest.raises(ValueError, match="input path"):
        analyze_corpus([])


def test_corpus_stats_cli_writes_report(tmp_path: Path) -> None:
    corpus = tmp_path / "ru.jsonl"
    corpus.write_text(
        json.dumps({"text": "привет мир", "domain": "ru"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "corpus_stats.json"

    main(
        [
            "--input",
            str(corpus),
            "--format",
            "jsonl",
            "--domain-field",
            "domain",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["documents"] == 1
    assert report["summary"]["language_hint"] == "cyrillic"
    assert report["domains"]["ru"]["words"] == 2
    assert report["settings"]["domain_field"] == "domain"


def test_corpus_stats_cli_rejects_invalid_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="invalid JSON"):
        main(
            [
                "--input",
                str(corpus),
                "--format",
                "jsonl",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
