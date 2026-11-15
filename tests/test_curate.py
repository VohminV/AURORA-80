from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora.data.curate import CurationConfig, curate
from aurora.data.curate import main as curate_main

BASE = (
    "AURORA combines recurrent state, associative memory and adaptive experts. "
    "The model predicts the next token from the entire preceding context. "
    "Associative memory retrieves information by content rather than position. "
    "Experts specialize during training instead of being assigned by hand. "
)
OTHER = (
    "Tokenized corpora require deterministic splits and content addressed versions. "
    "Every prepared file carries a checksum and a manifest dataset version. "
    "Training refuses to resume when the dataset fingerprint changes. "
    "Evaluation reports link metrics back to the exact dataset version. "
)


def _config(**overrides: object) -> CurationConfig:
    defaults: dict[str, object] = {
        "min_chars": 40,
        "max_chars": 10_000,
        "min_words": 5,
        "near_dedup": False,
        "pii_action": "drop",
    }
    defaults.update(overrides)
    return CurationConfig(**defaults)  # type: ignore[arg-type]


def _run(
    tmp_path: Path,
    documents: list[str],
    *,
    config: CurationConfig | None = None,
    rejects: bool = False,
) -> tuple[Path, dict[str, object]]:
    source = tmp_path / "raw.txt"
    source.write_text("\n\n".join(documents) + "\n", encoding="utf-8")
    output = tmp_path / "curated.txt"
    report_path = tmp_path / "report.json"
    report = curate(
        [source],
        output,
        config=config or _config(),
        report_path=report_path,
        rejects_path=tmp_path / "rejects.jsonl" if rejects else None,
    )
    return output, report


def test_curate_drops_exact_duplicates_and_writes_report(tmp_path: Path) -> None:
    output, report = _run(tmp_path, [BASE, BASE.upper(), OTHER])

    assert output.read_text(encoding="utf-8").count("AURORA combines") == 1
    assert "Tokenized corpora" in output.read_text(encoding="utf-8")
    assert report["counts"]["kept_documents"] == 2
    assert report["counts"]["dropped_exact_duplicate"] == 1
    assert report["output"]["documents"] == 2
    assert report["dataset_version"].startswith("sha256:")
    assert report["output"]["sha256"].startswith("sha256:")
    assert report["run_metadata"]["environment"]["torch"]
    assert report["inputs"][0]["sha256"]
    assert (tmp_path / "report.json").is_file()


def test_curation_version_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text(BASE + "\n\n" + OTHER + "\n", encoding="utf-8")
    first = curate([source], tmp_path / "first.txt", config=_config())
    second = curate([source], tmp_path / "second.txt", config=_config())

    assert first["dataset_version"] == second["dataset_version"]
    assert (tmp_path / "first.txt").read_bytes() == (tmp_path / "second.txt").read_bytes()


def test_curate_quality_filters(tmp_path: Path) -> None:
    documents = [
        "short",
        "a" * 50,
        "!@# %^& *() " * 6,
        "word another value " + "x" * 120,
        "\n".join(["repeated line with enough words here"] * 8),
        BASE,
    ]
    _, report = _run(
        tmp_path,
        documents,
        config=_config(
            min_words=3,
            max_symbol_ratio=0.1,
            max_repeated_char_run=20,
            max_duplicate_line_ratio=0.1,
        ),
    )
    reasons = report["counts"]["drop_reasons"]
    assert reasons["too_short"] == 1
    assert reasons["too_few_words"] == 1
    assert reasons["low_letter_ratio"] == 1
    assert reasons["repeated_characters"] == 1
    assert reasons["duplicate_lines"] == 1
    assert report["counts"]["kept_documents"] == 1


def test_pii_drop_and_redaction(tmp_path: Path) -> None:
    with_email = BASE + " Contact person@example.com for details."
    _, dropped = _run(tmp_path, [with_email, OTHER], rejects=True)
    assert dropped["counts"]["dropped_pii"] == 1
    assert dropped["counts"]["kept_documents"] == 1
    reject_lines = (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    reject = json.loads(reject_lines[0])
    assert reject["reason"] == "pii"
    assert reject["preview"] is None
    assert reject["pii"] == ["email"]

    output, redacted = _run(
        tmp_path,
        [with_email],
        config=_config(pii_action="redact"),
    )
    text = output.read_text(encoding="utf-8")
    assert "person@example.com" not in text
    assert "[EMAIL]" in text
    assert redacted["counts"]["redacted_documents"] == 1
    assert redacted["counts"]["kept_documents"] == 1


def test_pii_patterns_cover_ips_and_cards(tmp_path: Path) -> None:
    ip_document = BASE + " Server address is 192.168.10.20."
    card_document = OTHER + " Card 4111 1111 1111 1111 was used."
    _, report = _run(
        tmp_path,
        [ip_document, card_document, BASE.replace("AURORA", "Aurora")],
        rejects=True,
    )
    assert report["counts"]["dropped_pii"] == 2
    assert report["counts"]["kept_documents"] == 1
    reasons = [
        json.loads(line)["pii"]
        for line in (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert reasons == [["ipv4"], ["card"]]


def test_near_dedup(tmp_path: Path) -> None:
    generated = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
        "omicron pi rho sigma tau upsilon phi chi psi omega"
    ).split()
    words = [generated[index % len(generated)] + chr(97 + index % 26) for index in range(120)]
    original = " ".join(words) + "."
    changed = " ".join(words[:60] + ["changed"] + words[61:]) + "."
    output, report = _run(
        tmp_path,
        [original, changed],
        config=_config(near_dedup=True, near_dedup_threshold=7),
    )
    assert report["counts"]["dropped_near_duplicate"] == 1
    assert output.read_text(encoding="utf-8").count("changed") == 0


def test_jsonl_input_and_invalid_records(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps({"text": BASE})
        + "\n"
        + "not-json\n"
        + json.dumps({"text": OTHER})
        + "\n"
        + json.dumps({"payload": BASE})
        + "\n",
        encoding="utf-8",
    )
    report = curate(
        [source],
        tmp_path / "curated.txt",
        config=_config(),
        input_format="auto",
    )
    assert report["inputs"][0]["format"] == "jsonl"
    assert report["counts"]["input_records"] == 4
    assert report["counts"]["invalid_records"] == 2
    assert report["counts"]["kept_documents"] == 2


def test_max_documents_marks_input_truncated(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("\n\n".join([BASE, OTHER, BASE.replace("AURORA", "Aurora")]) + "\n")
    report = curate([source], tmp_path / "curated.txt", config=_config(), max_documents=1)
    assert report["counts"]["kept_documents"] == 1
    assert report["inputs"][0]["truncated"] is True
    assert report["inputs"][0]["sha256"]


def test_report_counts_scripts(tmp_path: Path) -> None:
    russian = (
        "Аврора объединяет рекуррентное состояние, ассоциативную память и экспертов. "
        "Модель предсказывает следующий токен по всему контексту."
    )
    greek = "αλφα βητα γαμμα δελτα εψιλον ζητα ητα θητα ιωτα καππα"
    _, report = _run(tmp_path, [BASE, russian, greek])
    scripts = report["counts"]["scripts"]
    assert scripts["latin"] > 0
    assert scripts["cyrillic"] > 0
    assert scripts["other"] > 0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        CurationConfig(min_chars=100, max_chars=100)
    with pytest.raises(ValueError, match="min_letter_ratio"):
        CurationConfig(min_letter_ratio=1.5)
    with pytest.raises(ValueError, match="near_dedup_threshold"):
        CurationConfig(near_dedup_threshold=8)
    with pytest.raises(ValueError, match="pii_action"):
        CurationConfig(pii_action="mask")


def test_short_curate_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one input"):
        curate([], tmp_path / "out.txt")
    with pytest.raises(ValueError, match="chunk_size"):
        curate([tmp_path / "missing.txt"], tmp_path / "out.txt", chunk_size=0)


def test_curate_rejects_empty_result(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text("tiny\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No documents survived"):
        curate([source], tmp_path / "out.txt", config=_config(min_chars=100))


def test_curate_cli(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    source.write_text(BASE + "\n\n" + OTHER + "\n", encoding="utf-8")
    output = tmp_path / "curated.txt"
    report = tmp_path / "report.json"
    curate_main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--report",
            str(report),
            "--min-chars",
            "20",
            "--min-words",
            "5",
            "--no-near-dedup",
        ]
    )
    assert output.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["counts"]["kept_documents"] == 2
