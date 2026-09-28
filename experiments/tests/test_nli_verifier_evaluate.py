import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from utils import nli_verifier_experiments as e3
from utils.dataset_io import write_json, write_jsonl
from utils.decision_models import FakeDecisionModel
from utils.decision_scoring import apply_holm

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nli_verifier.toml"
JUDGES = ("prompt_1_gpt-4o-mini-2024-07-18", "prompt_2_deepseek-chat")
HEARINGS = {"train": (1, 2, 3, 4, 5, 6), "validation": (7, 8, 9, 10)}
PER_HEARING = 6
FAMILY_SIZES = {"main": 10, "translation_model": 11, "jev": 4}


def benchmark_rows(rng: np.random.Generator) -> list[dict]:
    rows = []
    for split, hearings in HEARINGS.items():
        for hearing in hearings:
            for index in range(PER_HEARING):
                label = bool(index % 3)
                rows.append(
                    {
                        "id": f"nli-{hearing}-0-{index}",
                        "hearing_id": hearing,
                        "split": split,
                        "opinion": f"Opinião {hearing}.{index} sobre o projeto.",
                        "label_inferable": label,
                        "judge_inferable": {
                            key: bool(label if rng.random() < 0.8 else not label) for key in JUDGES
                        },
                    }
                )
    return rows


def units_of(rows: list[dict]) -> list[e3.PremiseUnit]:
    return [
        e3.PremiseUnit(
            row["id"],
            row["hearing_id"],
            row["split"],
            row["opinion"],
            (f"Trecho A de {row['id']}.", "", f"Trecho B de {row['id']}."),
        )
        for row in rows
    ]


def report_stub(key: str, files: dict) -> dict:
    return {
        "subset": None,
        "files": files,
        "model": {"name": key},
        "label_probes": None,
        "counts": {},
        "truncation": {},
        "timing": {},
        "premise": {},
        "created_at": "2026-09-24T00:00:00+00:00",
        "code": {},
    }


def write_scores(config: e3.VerifierConfig, key: str, rows: list[dict], directory: Path) -> None:
    files = e3.write_split_scores(rows, directory, key)
    write_json(report_stub(key, files), e3.score_report_file(directory, key))


def numeric_rows(spec: e3.ScorerSpec, units: list[e3.PremiseUnit], rng) -> list[dict]:
    return [
        {
            **e3.unit_header(unit),
            "items": [],
            "concatenated": None,
            "scores": {name: float(rng.random()) for name in spec.scores},
        }
        for unit in units
    ]


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    base = e3.load_config(CONFIG)
    rows = benchmark_rows(rng)
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl(rows, benchmark)
    digest = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    write_json({"artifact": {"sha256": digest}}, tmp_path / "benchmark_report.json")
    manifest = {
        "split_version": "synthetic",
        "dataset": {"sha256": base.lds_sha256},
        "train": list(HEARINGS["train"]),
        "validation": list(HEARINGS["validation"]),
        "test": [],
    }
    write_json(manifest, tmp_path / "manifest.json")
    config = dataclasses.replace(
        base,
        benchmark_path=benchmark,
        benchmark_report_path=tmp_path / "benchmark_report.json",
        manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "runs",
        bootstrap_samples=25,
    )
    units = units_of(rows)
    declaration = config.declarations["nli_verifier_v2"]
    run_dir = config.output_dir / "smoke"
    monkeypatch.setattr(e3, "load_decision_model", lambda *args: FakeDecisionModel())
    for key in declaration.scorers:
        spec = config.scorers[key]
        if spec.kind in e3.DECISION_KINDS:
            scored, _ = e3.score_units_decision(
                units, spec, config, "cpu", True, e3.portuguese_probes(config), "live"
            )
        else:
            scored = numeric_rows(spec, units, rng)
        write_scores(config, key, scored, run_dir)
    for key, run in declaration.imported.items():
        spec = config.scorers[key]
        write_scores(config, key, numeric_rows(spec, units, rng), config.output_dir / run)
    args = argparse.Namespace(
        run_name="smoke", output_dir=None, final_test=False, declaration="nli_verifier_v2"
    )
    e3.command_evaluate(args, config)
    return run_dir


def read_report(run_dir: Path) -> dict:
    return json.loads((run_dir / "evaluation_report.json").read_text())


def test_evaluate_builds_one_table_with_every_scorer(evaluated):
    report = read_report(evaluated)
    assert report["scorers_missing"] == []
    scorers = {system.split(".", 1)[0] for system in report["systems_evaluated"]}
    assert scorers == {
        "cosine_serafim",
        "xnli_mdeberta",
        "assin2_mdeberta",
        "laya_multi_pt",
        "laya_multi_en",
        "laya_en_en",
        "xnli_mdeberta_en",
        "laya_multi_en_m2m100",
        "laya_en_en_m2m100",
        "xnli_mdeberta_en_m2m100",
        "jev_en",
        "jev_pt",
    }
    assert "laya_multi_pt.max.consensus" in report["systems"]
    assert "jev_pt.max.stacked" in report["systems"]
    declared = report["declared_comparisons"]["validation"]
    assert declared["family_order"] == list(FAMILY_SIZES)
    assert declared["rule_superseded_by"] == "families_amendment"
    assert declared["holm_scope"] == "each_family"
    assert "once over the whole list" in declared["rule"]
    assert {name: f["family_size"] for name, f in declared["families"].items()} == FAMILY_SIZES
    assert all(family["complete"] for family in declared["families"].values())
    assert len(declared["comparisons"]) == sum(FAMILY_SIZES.values())
    assert all("missing" not in entry for entry in declared["comparisons"])
    assert all(0 <= entry["p_holm"] <= 1 for entry in declared["comparisons"])
    for name, family in declared["families"].items():
        entries = [entry for entry in declared["comparisons"] if entry["family"] == name]
        assert [entry["name"] for entry in entries] == family["comparisons"]
        recomputed = [{"p_value": entry["p_value"]} for entry in entries]
        apply_holm(recomputed)
        assert [entry["p_holm"] for entry in entries] == [e["p_holm"] for e in recomputed]
    order = report["robustness"]["laya_en_en"]["validation"]["order_changes"]
    assert order["nli_order_mean"]["chunk"]["premises"] == 4 * PER_HEARING * 2
    with open(evaluated / "comparison_table.csv") as f:
        table = list(csv.DictReader(f))
    systems = [row for row in table if row["kind"] not in ("llm_judge", "declared_comparison")]
    roles = {row["system"]: row["role"] for row in systems}
    assert roles["laya_multi_pt.max.panel"] == "primary"
    assert roles["xnli_mdeberta.max.entailment"] == "compared"
    assert roles["laya_multi_en_m2m100.max.panel"] == "compared"
    assert sum(1 for row in table if row["kind"] == "llm_judge") == len(JUDGES)
    compared = [row for row in table if row["kind"] == "declared_comparison"]
    assert len(systems) == len(report["systems_evaluated"])
    assert len(compared) == sum(FAMILY_SIZES.values())
    by_name = {entry["name"]: entry for entry in declared["comparisons"]}
    for row in compared:
        entry = by_name[row["comparison"]]
        assert row["family"] == entry["family"]
        assert float(row["p_holm"]) == pytest.approx(entry["p_holm"])
    assert report["table"]["columns"] == list(e3.TABLE_COLUMNS)
    assert all(set(row) == set(e3.TABLE_COLUMNS) for row in report["table"]["rows"])
    assert report["declared"]["primary_system"] == "laya_multi_pt.max.panel"


def test_a_missing_jev_leaves_the_other_families_unchanged(evaluated):
    before = read_report(evaluated)["declared_comparisons"]["validation"]
    for path in evaluated.glob("scores/jev_*"):
        path.unlink()
    config = dataclasses.replace(
        e3.load_config(CONFIG), output_dir=evaluated.parent, bootstrap_samples=25
    )
    manifest = evaluated.parent.parent / "manifest.json"
    config = dataclasses.replace(
        config,
        manifest_path=manifest,
        benchmark_path=evaluated.parent.parent / "benchmark.jsonl",
        benchmark_report_path=evaluated.parent.parent / "benchmark_report.json",
    )
    args = argparse.Namespace(
        run_name="smoke", output_dir=None, final_test=False, declaration="nli_verifier_v2"
    )
    e3.command_evaluate(args, config)
    report = read_report(evaluated)
    assert report["scorers_missing"] == ["jev_en", "jev_pt"]
    after = report["declared_comparisons"]["validation"]
    entries = {e["name"]: e for e in after["comparisons"]}
    assert entries["C07_translation_effect_jev"]["missing"] == [
        "jev_en.max.panel",
        "jev_pt.max.panel",
    ]
    jev = after["families"]["jev"]
    assert jev["family_size"] == FAMILY_SIZES["jev"] and not jev["complete"]
    assert jev["missing"] == jev["comparisons"]
    assert all(entries[name]["p_holm"] == 1.0 for name in jev["comparisons"])
    old = {e["name"]: e for e in before["comparisons"]}
    for family in ("main", "translation_model"):
        assert after["families"][family] == before["families"][family]
        for name in after["families"][family]["comparisons"]:
            assert entries[name] == old[name]
