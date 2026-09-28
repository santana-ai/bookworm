import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import StubEncoder
from typer.testing import CliRunner

from bookworm import ConfigError, Segmentation, UdvRecord, load_udv_jsonl, sha256_of_file
from bookworm.cli import create_app
from bookworm.data.io import JsonObject, write_json
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.signals import (
    HEAVY_ARTIFACT_MANIFEST,
    load_site_signals,
    missing_signal_files,
    translation_key,
)

runner = CliRunner()
app = create_app(encoder_factory=lambda config, hearings: StubEncoder())
CONFIG = "udv_mini.toml"
RECORDS = "out/mini.jsonl"
REPORT = "verifier/report.json"
SIGNATURE = "f" * 64
ABBREVIATIONS = ["Sr."]
SEGMENTATION: JsonObject = {"join_abbreviations": ABBREVIATIONS, "join_short_parts": True}
QUESTIONS = {
    "p1_nli": {
        "payload": {
            "type": "choice",
            "instructions": "What is the relationship between `premise` and `hypothesis`?",
            "criteria": {"entailment": "a", "neutral": "b", "contradiction": "c"},
        },
        "support_option": "entailment",
        "reverses": None,
    },
    "p3_inferable": {
        "payload": {"type": "noul", "instructions": "Can it be inferred?"},
        "support_option": None,
        "reverses": None,
    },
    "p7_coverage": {
        "payload": {"type": "score", "instructions": "How much?", "criteria": ["none", "all"]},
        "support_option": None,
        "reverses": None,
    },
}
LAYA = ("laya_multi_pt", "laya_en_en")
SCORERS = (*LAYA, "xnli_mdeberta")
LANGUAGE = {"laya_multi_pt": "pt", "laya_en_en": "en", "xnli_mdeberta": "pt"}
SEGMENTER = Segmentation(frozenset(ABBREVIATIONS), True)


def file_entry(path: str) -> JsonObject:
    return {"path": path, "sha256": sha256_of_file(Path(path))}


def write_rows(path: str, rows: list[JsonObject]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def english(text: str) -> str:
    return f"EN[{text}]"


def probability(position: int) -> float:
    return round(0.1 + 0.07 * position, 4)


def laya_row(record: UdvRecord, position: int, shift: float) -> JsonObject:
    signals = {key: round(probability(position) * shift, 4) for key in QUESTIONS}
    return {"id": record.id, "items": [{"signals": signals, "truncated": False}]}


def xnli_row(record: UdvRecord, position: int) -> JsonObject:
    value = probability(position)
    probabilities = {"entailment": value, "neutral": 1 - value, "contradiction": 0.0}
    return {"id": record.id, "items": [{"probabilities": probabilities, "truncated": False}]}


def score_report(name: str) -> JsonObject:
    report: JsonObject = {
        "scorer": name,
        "language": LANGUAGE[name],
        "model": {"name": f"model/{name}", "revision": "rev-1"},
    }
    if name in LAYA:
        report["questions"] = QUESTIONS
    if name == "laya_en_en":
        report["translation"] = {
            "store": {
                "path": "cache/translations.jsonl",
                "signature_sha256": SIGNATURE,
                "segmentation": SEGMENTATION,
            },
            "model": {"name": "translator/one", "revision": "rev-t", "license": "CC0"},
        }
    return report


def translation_texts(records: list[UdvRecord]) -> list[str]:
    texts: list[str] = []
    for record in records:
        if record.evidence is None:
            continue
        texts.append(normalize_whitespace(record.proposition))
        texts.extend(SEGMENTER.segments(record.evidence.text))
    return list(dict.fromkeys(texts))


def write_translations(texts: list[str]) -> None:
    write_rows(
        "cache/translations.jsonl",
        [
            {
                "key": translation_key(SIGNATURE, text),
                "signature_sha256": SIGNATURE,
                "source": text,
                "translation": english(text),
            }
            for text in texts
        ],
    )


def write_report() -> None:
    report = {
        "inputs": {
            "udv": file_entry(RECORDS),
            "udv_score_files": {name: file_entry(f"scores/{name}_udv.jsonl") for name in SCORERS},
        },
        "score_runs": {name: file_entry(f"scores/{name}_report.json") for name in SCORERS},
        "outputs": file_entry("verifier/udv.jsonl"),
        "primary": {
            "candidate": "stub:primary",
            "fit": {"threshold": 0.5},
            "final_test_result": {"threshold": 0.5, "threshold_fitted_on": "train"},
        },
    }
    write_json(report, Path(REPORT))


def write_sources(records: list[UdvRecord]) -> None:
    verifier = []
    for position, record in enumerate(records):
        scored = record.evidence is not None
        verifier.append(
            {
                "udv_id": record.id,
                "hearing_id": record.hearing_id,
                "tier": record.tier,
                "support_type": None if record.evidence is None else record.evidence.support_type,
                "scored": scored,
                "primary_probability": probability(position) if scored else None,
                "supported_at_train_threshold": probability(position) >= 0.5 if scored else None,
            }
        )
    write_rows("verifier/udv.jsonl", verifier)
    scored_records = [(p, r) for p, r in enumerate(records) if r.evidence is not None]
    write_rows("scores/laya_multi_pt_udv.jsonl", [laya_row(r, p, 1.0) for p, r in scored_records])
    write_rows("scores/laya_en_en_udv.jsonl", [laya_row(r, p, 0.5) for p, r in scored_records])
    write_rows("scores/xnli_mdeberta_udv.jsonl", [xnli_row(r, p) for p, r in scored_records])
    for name in SCORERS:
        write_json(score_report(name), Path(f"scores/{name}_report.json"))
    write_translations(translation_texts(records))
    write_report()


@pytest.fixture
def signal_workdir(mini_workdir: Path) -> Path:
    result = runner.invoke(app, ["build-udvs", "--config", CONFIG, "--run-name", "mini"])
    assert result.exit_code == 0, result.output
    write_sources(load_udv_jsonl(Path(RECORDS)))
    return mini_workdir


def export(output: str, *options: str) -> Any:
    return runner.invoke(
        app,
        [
            "export-hearing",
            "--config",
            CONFIG,
            "--run-name",
            "mini",
            "--hearing",
            "2",
            "--output",
            output,
            *options,
        ],
    )


def without_signals(payload: JsonObject) -> JsonObject:
    plain = {key: value for key, value in payload.items() if key != "signals"}
    plain["udvs"] = [
        {key: value for key, value in udv.items() if key != "signals"} for udv in payload["udvs"]
    ]
    return plain


def test_segmentation_joins_abbreviations_and_short_parts() -> None:
    segmenter = Segmentation(frozenset({"Sr."}), True)
    text = "Obrigado, Sr. Presidente, pela palavra hoje. Sim. Eu quero falar agora mesmo."
    assert segmenter.segments(text) == [
        "Obrigado, Sr. Presidente, pela palavra hoje.",
        "Sim. Eu quero falar agora mesmo.",
    ]


def test_segmentation_keeps_a_trailing_short_part_with_the_previous_unit() -> None:
    segmenter = Segmentation(frozenset(), True)
    assert segmenter.segments("Esta frase tem palavras suficientes. Fim.") == [
        "Esta frase tem palavras suficientes. Fim."
    ]
    assert segmenter.segments("Curta.") == ["Curta."]


def test_segmentation_without_joins_gives_one_unit_per_boundary_part() -> None:
    segmenter = Segmentation(frozenset(), False)
    assert segmenter.segments("Sim.  Não.\nTalvez!") == ["Sim.", "Não.", "Talvez!"]


def test_export_adds_signal_blocks_and_nothing_else(signal_workdir: Path) -> None:
    assert export("plain.json").exit_code == 0
    result = export("signals.json", "--verifier-report", REPORT)
    assert result.exit_code == 0, result.output
    plain_bytes = Path("plain.json").read_bytes()
    with_signals = json.loads(Path("signals.json").read_text(encoding="utf-8"))
    write_json(without_signals(with_signals), Path("stripped.json"))
    assert Path("stripped.json").read_bytes() == plain_bytes
    assert "signals" not in json.loads(plain_bytes)
    assert list(with_signals)[-1] == "signals"
    assert all(list(udv)[-1] == "signals" for udv in with_signals["udvs"])


def test_export_signal_values_come_from_the_sources(signal_workdir: Path) -> None:
    assert export("signals.json", "--verifier-report", REPORT).exit_code == 0
    payload = json.loads(Path("signals.json").read_text(encoding="utf-8"))
    records = {record.id: record for record in load_udv_jsonl(Path(RECORDS))}
    positions = {record_id: position for position, record_id in enumerate(records)}
    summary = payload["signals"]
    assert summary["verifier"] == {
        "name": "stub:primary",
        "threshold": 0.5,
        "threshold_fitted_on": "train",
        "report_sha256": sha256_of_file(Path(REPORT)),
    }
    assert [question["id"] for question in summary["questions"]] == list(QUESTIONS)
    assert summary["questions"][0]["options"] == ["entailment", "neutral", "contradiction"]
    assert summary["questions"][1]["options"] is None
    assert summary["translation"] == {
        "name": "translator/one",
        "revision": "rev-t",
        "license": "CC0",
    }
    assert summary["scorers"]["laya_en_en"] == {
        "model": "model/laya_en_en",
        "revision": "rev-1",
        "language": "en",
    }
    kinds = set()
    for udv in payload["udvs"]:
        record = records[udv["id"]]
        signals = udv["signals"]
        if record.evidence is None:
            kinds.add("unscored")
            assert signals == {
                "scored": False,
                "verifier": None,
                "laya": None,
                "xnli": None,
                "translation": None,
            }
            continue
        kinds.add("scored")
        value = probability(positions[record.id])
        assert signals["verifier"] == {"probability": value, "supported": value >= 0.5}
        assert signals["laya"]["laya_en_en"]["p3_inferable"] == round(value * 0.5, 4)
        assert signals["xnli"]["entailment"] == value
        assert signals["translation"] == {
            "premise": " ".join(english(part) for part in SEGMENTER.segments(record.evidence.text)),
            "hypothesis": english(normalize_whitespace(record.proposition)),
        }
    assert kinds == {"scored", "unscored"}


def test_export_site_with_signals_keeps_the_index(signal_workdir: Path) -> None:
    base = ["export-site", "--config", CONFIG, "--run-name", "mini"]
    assert runner.invoke(app, [*base, "--output", "plain"]).exit_code == 0
    result = runner.invoke(app, [*base, "--output", "rich", "--verifier-report", REPORT])
    assert result.exit_code == 0, result.output
    assert Path("rich/index.json").read_bytes() == Path("plain/index.json").read_bytes()
    for hearing_id in (1, 2):
        rich = json.loads(Path(f"rich/hearings/{hearing_id}.json").read_text(encoding="utf-8"))
        plain = json.loads(Path(f"plain/hearings/{hearing_id}.json").read_text(encoding="utf-8"))
        assert without_signals(rich) == plain
        assert all("signals" in udv for udv in rich["udvs"])


def drop_verifier_row() -> None:
    rows = Path("verifier/udv.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    Path("verifier/udv.jsonl").write_text("".join(rows[:-1]), encoding="utf-8")
    write_report()


def rename_verifier_row() -> None:
    text = Path("verifier/udv.jsonl").read_text(encoding="utf-8")
    Path("verifier/udv.jsonl").write_text(
        text.replace('"udv-2-0-0"', '"udv-2-9-9"'), encoding="utf-8"
    )
    write_report()


def change_score_file() -> None:
    with Path("scores/laya_en_en_udv.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")


def drop_translation() -> None:
    rows = Path("cache/translations.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    Path("cache/translations.jsonl").write_text("".join(rows[1:]), encoding="utf-8")


def remove_score_report() -> None:
    Path("scores/xnli_mdeberta_report.json").unlink()


def change_run_records() -> None:
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    report["inputs"]["udv"]["sha256"] = "0" * 64
    write_json(report, Path(REPORT))


def change_tier() -> None:
    text = Path("verifier/udv.jsonl").read_text(encoding="utf-8")
    first = json.loads(text.splitlines()[0])
    row = {
        **first,
        "tier": "semantic_match_weak" if first["tier"] != "semantic_match_weak" else "quote_found",
    }
    lines = text.splitlines(keepends=True)
    lines[0] = json.dumps(row) + "\n"
    Path("verifier/udv.jsonl").write_text("".join(lines), encoding="utf-8")
    write_report()


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (drop_verifier_row, "UDV ids do not match the run (1 missing"),
        (rename_verifier_row, "e.g. ['udv-2-9-9']"),
        (change_score_file, "laya_en_en_udv.jsonl: sha256"),
        (drop_translation, "1 of"),
        (remove_score_report, "are missing (1): scores/xnli_mdeberta_report.json"),
        (change_run_records, "not the run records"),
        (change_tier, "tier or support_type different from the run record"),
    ],
)
def test_export_with_broken_signal_sources_exits_two(
    signal_workdir: Path, breakage: Callable[[], None], message: str
) -> None:
    breakage()
    result = export("signals.json", "--verifier-report", REPORT)
    assert result.exit_code == 2
    assert message in result.stderr
    assert not Path("signals.json").exists()


def test_missing_heavy_score_files_are_named_with_the_manifest(signal_workdir: Path) -> None:
    Path("scores/laya_en_en_udv.jsonl").unlink()
    Path("scores/xnli_mdeberta_udv.jsonl").unlink()
    assert missing_signal_files(Path(REPORT)) == [
        Path("scores/laya_en_en_udv.jsonl"),
        Path("scores/xnli_mdeberta_udv.jsonl"),
    ]
    result = export("signals.json", "--verifier-report", REPORT)
    assert result.exit_code == 2
    assert (
        "are missing (2): scores/laya_en_en_udv.jsonl, scores/xnli_mdeberta_udv.jsonl"
        in result.stderr
    )
    assert HEAVY_ARTIFACT_MANIFEST in result.stderr
    assert "Traceback" not in result.stderr


def test_export_with_a_missing_report_exits_two(signal_workdir: Path) -> None:
    result = export("signals.json", "--verifier-report", "verifier/absent.json")
    assert result.exit_code == 2
    assert "verifier/absent.json: verifier report not found" in result.stderr


def test_load_site_signals_rejects_different_thresholds(signal_workdir: Path) -> None:
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    report["primary"]["final_test_result"]["threshold"] = 0.6
    write_json(report, Path(REPORT))
    records = load_udv_jsonl(Path(RECORDS))
    with pytest.raises(ConfigError, match="thresholds differ"):
        load_site_signals(Path(REPORT), records, sha256_of_file(Path(RECORDS)))


UDV_CUT = 0.3


def add_udv_threshold(consistent: bool = True) -> None:
    rows = [
        json.loads(line)
        for line in Path("verifier/udv.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        value = row["primary_probability"]
        row["supported_at_udv_threshold"] = None if value is None else value >= UDV_CUT
    scored = next(row for row in rows if row["scored"])
    if not consistent:
        scored["supported_at_udv_threshold"] = not scored["supported_at_udv_threshold"]
    write_rows("verifier/udv.jsonl", rows)
    report = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    report["outputs"] = file_entry("verifier/udv.jsonl")
    report["udv_threshold"] = {
        "value": UDV_CUT,
        "exact": UDV_CUT,
        "rule": "legacy_random",
        "interval": {"low": 0.25, "high": 0.35},
        "path": "calibration/udv_threshold.json",
    }
    write_json(report, Path(REPORT))


def test_signals_carry_the_udv_premise_cut_when_the_report_has_one(signal_workdir: Path) -> None:
    add_udv_threshold()
    records = load_udv_jsonl(Path(RECORDS))
    signals = load_site_signals(Path(REPORT), records, sha256_of_file(Path(RECORDS)))
    assert signals.summary["verifier"]["threshold"] == 0.5
    assert signals.summary["verifier"]["udv_threshold"] == {
        "value": UDV_CUT,
        "rounded": UDV_CUT,
        "rule": "legacy_random",
        "interval": [0.25, 0.35],
        "source": "calibration/udv_threshold.json",
    }
    scored = [record for record in records if record.evidence is not None]
    for record in scored:
        verifier = signals.for_record(record)["verifier"]
        assert verifier["supported"] == (verifier["probability"] >= 0.5)
        assert verifier["supported_at_udv_threshold"] == (verifier["probability"] >= UDV_CUT)


def test_signals_refuse_a_udv_decision_that_does_not_follow_the_cut(
    signal_workdir: Path,
) -> None:
    add_udv_threshold(consistent=False)
    records = load_udv_jsonl(Path(RECORDS))
    with pytest.raises(ConfigError, match="supported_at_udv_threshold does not follow"):
        load_site_signals(Path(REPORT), records, sha256_of_file(Path(RECORDS)))


def write_precision_report() -> Path:
    path = Path("validation/precision.json")
    write_json(
        {
            "status": "final",
            "sample_name": "mini_sample",
            "splits_used": ["test"],
            "confidence_level": 0.95,
            "criteria_declared_on": "2026-01-01",
            "judged_items": 3,
            "label_semantics": {"trecho_sustenta": "one annotator's reading"},
            "mini": {
                "strata": {
                    "direct_quote": {
                        "question": "trecho_sustenta",
                        "udv_ids_by_judgment": {
                            "correta": ["udv-1-0-0"],
                            "parcial": ["udv-2-0-0"],
                            "incorreta": [],
                        },
                    },
                    "speaker_check": {
                        "question": "pessoa_falou",
                        "udv_ids_by_judgment": {"falou": ["udv-2-3-0"]},
                    },
                },
                "criteria": [],
            },
        },
        path,
    )
    return path


def test_export_site_adds_the_human_validation_to_the_index(signal_workdir: Path) -> None:
    add_udv_threshold()
    report = write_precision_report()
    base = ["export-site", "--config", CONFIG, "--run-name", "mini", "--output", "site"]
    result = runner.invoke(
        app, [*base, "--verifier-report", REPORT, "--human-validation", str(report)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["validation"] == {"judged_udvs": 3, "bands": True}
    index = json.loads(Path("site/index.json").read_text(encoding="utf-8"))
    validation = index["validation"]
    assert validation["run"] == "mini"
    assert validation["tiers"]["quote_found"]["udvs"] == 2
    assert validation["speaker_check"] == {"udvs": 1, "falou": 1, "nao_falou": 0, "nao_sei": 0}
    bands = validation["verifier_bands"]
    assert bands["cuts"] == [UDV_CUT, 0.5]
    assert sum(band["udvs"] for band in bands["bands"]) + bands["unscored"] == 2


def test_export_site_refuses_a_validation_of_other_udvs(signal_workdir: Path) -> None:
    report = write_precision_report()
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["mini"]["strata"]["direct_quote"]["udv_ids_by_judgment"]["correta"] = ["udv-9-9-9"]
    write_json(payload, report)
    base = ["export-site", "--config", CONFIG, "--run-name", "mini", "--output", "site"]
    result = runner.invoke(app, [*base, "--human-validation", str(report)])
    assert result.exit_code == 2
    assert "udv-9-9-9 is not in the run records" in result.stderr
