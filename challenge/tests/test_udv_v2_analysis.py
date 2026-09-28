import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils import calibrate_udv_v2 as calibration
from utils import udv_v2_analysis as analysis
from utils.calibrate_threshold import CalibrationConfig
from utils.generate_validation_sample import CSV_COLUMNS


def udv(udv_id: str, tier: str, text: str | None, start: int = 0, prefix: str | None = None):
    evidence = None
    if text is not None:
        evidence = {
            "text": text,
            "start_char": start,
            "end_char": start + len(text),
            "speaker_turn": 1,
            "score": None if tier == "quote_found" else 0.6,
            "support_type": "direct_quote" if tier == "quote_found" else "semantic_similarity",
            "quote_prefix": prefix,
        }
    return {"id": udv_id, "hearing_id": 1, "tier": tier, "evidence": evidence, "proposition": ""}


def test_change_kinds():
    first = udv("a", "quote_found", "Uma frase aqui.")
    longer = udv("a", "quote_found", "Uma frase aqui. E outra frase.")
    assert analysis.change_kind(first, first) == "same_tier_same_evidence"
    assert analysis.change_kind(first, longer) == "quote_extended"
    sentence = udv("b", "semantic_match_high", "Frase do meio.", 10)
    window = udv("b", "semantic_match_weak", "Frase do meio. Frase seguinte.", 10)
    elsewhere = udv("b", "semantic_match_high", "Outra coisa dita.", 50)
    assert analysis.change_kind(sentence, window) == "window_contains_v1_sentence"
    assert analysis.change_kind(sentence, elsewhere) == "window_elsewhere"
    moved = udv("b", "semantic_match_weak", "Frase do meio.", 10)
    assert analysis.change_kind(sentence, moved) == "tier_only"


def test_closing_words_found_ignores_trailing_punctuation():
    quote = "o hospital está fechado há dois anos."
    assert analysis.closing_words_found(quote, "O hospital está fechado há dois anos, disse.")
    assert not analysis.closing_words_found(quote, "O hospital está fechado.")


def key_with_items():
    strata = [
        {"name": "direct_quote", "tiers": ["quote_found"], "support_types": ["direct_quote"]},
        {
            "name": "semantic_match_high",
            "tiers": ["semantic_match_high"],
            "support_types": ["semantic_similarity"],
        },
        {
            "name": "speaker_check",
            "tiers": ["no_evidence", "person_not_resolved"],
            "support_types": ["none"],
        },
    ]
    items = {
        "A001": {"question": "trecho_sustenta", "stratum": "direct_quote", "udv_ids": ["a"]},
        "A002": {"question": "trecho_sustenta", "stratum": "semantic_match_high", "udv_ids": ["b"]},
        "A003": {"question": "pessoa_falou", "stratum": "speaker_check", "udv_ids": ["c", "d"]},
    }
    return {"sample_name": "s", "strata": strata, "items": items}


def test_annotation_plan_inherits_only_identical_evidence():
    v1 = {
        "a": udv("a", "quote_found", "Uma frase."),
        "b": udv("b", "semantic_match_high", "Outra frase."),
        "c": udv("c", "person_not_resolved", None),
        "d": udv("d", "no_evidence", None),
    }
    v2 = dict(v1)
    v2["b"] = udv("b", "semantic_match_high", "Outra frase. E mais.")
    plan = analysis.annotation_plan(key_with_items(), v1, v2)
    actions = {item: entry["action"] for item, entry in plan["item_plan"].items()}
    assert actions == {"A001": "inherit", "A002": "reannotate", "A003": "inherit"}
    assert plan["item_plan"]["A001"]["v2_stratum"] == "direct_quote"
    assert plan["item_plan"]["A003"]["v2_stratum"] == "speaker_check"
    assert plan["inherited_by_v2_stratum"] == {"direct_quote": 1, "speaker_check": 1}


DISPLAY = {"hearing_id": "1", "pergunta": "p", "participante": "x", "afirmacao": "a", "trecho": "t"}


def scored_key():
    key = key_with_items()
    strata = [{**stratum, "question": "trecho_sustenta"} for stratum in key["strata"][:2]]
    strata.append({**key["strata"][2], "question": "pessoa_falou"})
    tiers = {"A001": "quote_found", "A002": "semantic_match_high", "A003": "no_evidence"}
    items = {
        item_id: {**item, "tier": tiers[item_id], "quote_cue_in_trecho": False, "display": DISPLAY}
        for item_id, item in key["items"].items()
    }
    items["A004"] = {**items["A002"], "udv_ids": ["e"]}
    names = [stratum["name"] for stratum in strata]
    return {
        **key,
        "strata": strata,
        "items": items,
        "role": "annotation",
        "splits_used": ["train"],
        "dry_run": False,
        "population": {
            "sampled_splits": {"direct_quote": 5, "semantic_match_high": 9, "speaker_check": 2}
        },
        "sample": {name: {} for name in names},
        "criteria": {
            "declared_on": "2026-09-23",
            "declaration": "d",
            "rules": [
                {
                    "name": "quote",
                    "stratum": "direct_quote",
                    "metric": "strict_precision",
                    "min_wilson_lower": 0.9,
                }
            ],
        },
    }


def write_sheet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), delimiter=";", restval="")
        writer.writeheader()
        writer.writerows([{**DISPLAY, **row} for row in rows])


def test_partial_judgments_skip_empty_rows_and_list_invalid_labels(tmp_path):
    key = scored_key()
    config = analysis.load_validation_config(analysis.Path("configs/validation_sample.toml"))
    sheet = tmp_path / "annotation.csv"
    write_sheet(
        sheet,
        [
            {"item_id": "A001", "julgamento": "Correta"},
            {"item_id": "A002", "julgamento": "acho que sim"},
            {"item_id": "A003", "julgamento": ""},
            {"item_id": "A004", "julgamento": "parcial", "existe_trecho_melhor": "nao"},
        ],
    )
    found = analysis.partial_judgments(analysis.read_annotation_csv(sheet, config), key, config)
    assert sorted(found["valid"]) == ["A001", "A004"]
    assert found["invalid_labels"] == [{"item_id": "A002", "julgamento": "acho que sim"}]
    assert found["better_passage_missing"] == ["A001"]
    write_sheet(sheet, [{"item_id": "A001", "julgamento": "correta", "trecho": "mudou"}])
    with pytest.raises(SystemExit, match="trecho differs"):
        analysis.partial_judgments(analysis.read_annotation_csv(sheet, config), key, config)


def test_score_command_reports_interim_precision_for_both_runs(tmp_path):
    key = scored_key()
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "annotation_key.json").write_text(json.dumps(key))
    sheet = tmp_path / "filled.csv"
    write_sheet(
        sheet,
        [
            {"item_id": "A001", "julgamento": "correta"},
            {"item_id": "A002", "julgamento": "parcial"},
            {"item_id": "A003", "julgamento": ""},
            {"item_id": "A004", "julgamento": "incorreta"},
        ],
    )
    item_plan = {
        "A001": {"action": "reannotate", "v2_stratum": "direct_quote", "v2_tiers": ["quote_found"]},
        "A002": {
            "action": "inherit",
            "v2_stratum": "semantic_match_high",
            "v2_tiers": ["semantic_match_high"],
        },
        "A003": {"action": "inherit", "v2_stratum": "speaker_check", "v2_tiers": ["no_evidence"]},
        "A004": {"action": "inherit", "v2_stratum": None, "v2_tiers": ["semantic_match_weak"]},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "rule": "r",
                "caveat": "c",
                "item_plan": item_plan,
                "v2_population_sampled_splits": {"direct_quote": 5, "semantic_match_high": 8},
                "inputs": {
                    "annotation_key": {
                        "sha256": analysis.sha256_of_file(sample / "annotation_key.json")
                    }
                },
            }
        )
    )
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        config=analysis.Path("configs/validation_sample.toml"),
        sample_dir=str(sample),
        annotation=str(sheet),
        plan=str(plan),
        final_test=False,
        output=str(output),
        supplement_dir=None,
        supplement_annotation=None,
    )
    analysis.command_score(args)
    report = json.loads(output.read_text())
    assert report["status"] == "interim" and report["judged"] == "3 of 4"
    v1 = report["udv_v1"]["strata"]
    assert v1["direct_quote"]["strict_precision"]["trials"] == 1
    assert v1["semantic_match_high"]["judged_of_sample"] == "2 of 2"
    assert v1["semantic_match_high"]["tolerant_precision"]["successes"] == 1
    assert report["udv_v1"]["criteria"][0]["status"] == "INTERIM_FAIL"
    assert report["udv_v1"]["criteria"][0]["valid_for_decision"] is False
    v2 = report["udv_v2_unchanged"]
    assert v2["items_in_sample"] == 2
    assert v2["strata"]["semantic_match_high"]["judged_of_sample"] == "1 of 1"
    assert v2["strata"]["semantic_match_high"]["population"] == 8
    assert v2["strata"]["direct_quote"]["judged_udvs"] == 0
    assert v2["criteria"][0]["status"] == "INTERIM_NOT_EVALUABLE"
    with pytest.raises(SystemExit, match="never overwritten"):
        analysis.command_score(args)
    write_sheet(sheet, [{"item_id": item, "julgamento": ""} for item in key["items"]])
    args.output = str(tmp_path / "empty.json")
    with pytest.raises(SystemExit, match=analysis.NO_JUDGMENTS):
        analysis.command_score(args)


def test_stratum_population_counts_sampled_splits_only():
    records = [
        udv("a", "quote_found", "x"),
        {**udv("b", "semantic_match_high", "y"), "hearing_id": 2},
        udv("c", "no_evidence", None),
    ]
    strata = key_with_items()["strata"]
    counts = analysis.stratum_population(records, strata, {1: "test", 2: "train"}, ["test"])
    assert counts == {"direct_quote": 1, "semantic_match_high": 0, "speaker_check": 1}


def test_window_query_maps_sentence_targets_to_windows():
    windows = calibration.WindowUnits(
        texts=["s0 s1", "s1 s2", "s3"],
        spans=["s0 s1", "s1 s2", "s3"],
        positions=[(0, 1), (1, 2), (3,)],
    )
    query = {"id": "q", "target_indices": [2]}
    converted = calibration.window_query(query, windows, 10)
    assert converted["target_indices"] == [1]
    assert converted["sentence_target_indices"] == [2]
    assert converted["sentence_offset"] == 10
    assert calibration.window_query({"id": "q", "target_indices": [9]}, windows, 0) is None


def calibration_config():
    return CalibrationConfig(
        version="v",
        splits=("train",),
        manifest_path=None,
        masked_benchmark_path=None,
        output_dir=None,
        primary_rule="legacy_random",
        seed=42,
        bootstrap_samples=50,
        bootstrap_unit="hearing",
        confidence_level=0.95,
        negatives_per_query=1,
        positive_quantile=0.25,
        negative_quantile=0.75,
        threshold_decimals=2,
    )


def test_verifier_rule_is_the_midpoint_of_the_quantiles():
    queries = [f"q{index}" for index in range(8)]
    hearings = {query: index % 3 for index, query in enumerate(queries)}
    positive = {query: 0.8 + index / 100 for index, query in enumerate(queries)}
    negative = {query: 0.1 + index / 100 for index, query in enumerate(queries)}
    result = calibration.verifier_rule(
        queries, hearings, positive, negative, calibration_config(), 0
    )
    expected = (
        np.quantile(list(positive.values()), 0.25) + np.quantile(list(negative.values()), 0.75)
    ) / 2
    assert result["threshold_exact"] == pytest.approx(expected)
    assert result["pairs"]["queries"] == 8
    assert result["bootstrap"]["threshold"]["replicates"] == 50
    assert result["positives_below_threshold"] == 0


def evidence_udv(udv_id: str, tier: str, text: str, start: int) -> dict:
    record = udv(udv_id, tier, text, start)
    record["evidence"]["speaker_turn"] = 0
    record["proposition"] = "Disse que a frase importa."
    record["actor"] = {"name": "Fulana", "role": "Deputada"}
    return record


def plan_entry(action: str, stratum: str | None, udv_ids: list[str], question: str) -> dict:
    return {
        "action": action,
        "v2_stratum": stratum,
        "v2_tiers": ["semantic_match_high"],
        "udv_ids": udv_ids,
        "question": question,
    }


def test_supplement_rows_show_the_v2_evidence_of_reannotated_items_only():
    transcript = "Primeira frase dita. Segunda frase dita aqui. Terceira frase."
    window = "Segunda frase dita aqui. Terceira frase."
    start = transcript.index(window)
    v2 = {
        "a": evidence_udv("a", "semantic_match_high", window, start),
        "b": evidence_udv("b", "quote_found", "Primeira frase dita.", 0),
    }
    plan = {
        "item_plan": {
            "A001": plan_entry("reannotate", "semantic_match_high", ["a"], "trecho_sustenta"),
            "A002": plan_entry("inherit", "direct_quote", ["b"], "trecho_sustenta"),
        }
    }
    entries = analysis.reannotated_records(plan, v2)
    assert [(item_id, record["id"], name) for item_id, record, name in entries] == [
        ("A001", "a", "semantic_match_high")
    ]
    view = {
        "transcript": transcript,
        "turns": [{"turn_index": 0, "start_char": 0, "end_char": len(transcript)}],
        "headers": ["A SRA. FULANA"],
    }
    strata = analysis.key_strata(
        {"strata": [{**key_with_items()["strata"][1], "question": "trecho_sustenta"}]}
    )
    items = analysis.supplement_items(entries, {1: view}, strata, 350, np.random.default_rng(0))
    assert list(items) == ["C001"]
    item = items["C001"]
    assert item["v1_item_id"] == "A001" and item["stratum"] == "semantic_match_high"
    assert item["display"]["trecho"] == window
    assert item["display"]["contexto_antes"].endswith("Primeira frase dita.")
    rows = analysis.sheet_rows(items)
    assert rows[0]["julgamento"] == "" and "v1_item_id" not in rows[0]


def test_reannotated_records_refuse_items_without_v2_evidence():
    plan = {"item_plan": {"A001": plan_entry("reannotate", None, ["a"], "trecho_sustenta")}}
    with pytest.raises(SystemExit, match="no evidence stratum"):
        analysis.reannotated_records(plan, {"a": udv("a", "no_evidence", None)})
    plan = {"item_plan": {"A001": plan_entry("reannotate", "speaker_check", ["a"], "pessoa_falou")}}
    with pytest.raises(SystemExit, match="single-UDV evidence"):
        analysis.reannotated_records(plan, {"a": udv("a", "no_evidence", None)})


def scored_plan(sample):
    item_plan = {
        "A001": {"action": "reannotate", "v2_stratum": "direct_quote", "v2_tiers": ["quote_found"]},
        "A002": {
            "action": "inherit",
            "v2_stratum": "semantic_match_high",
            "v2_tiers": ["semantic_match_high"],
        },
        "A003": {"action": "inherit", "v2_stratum": "speaker_check", "v2_tiers": ["no_evidence"]},
        "A004": {"action": "inherit", "v2_stratum": None, "v2_tiers": ["semantic_match_weak"]},
    }
    return {
        "rule": "r",
        "caveat": "c",
        "item_plan": item_plan,
        "v2_population_sampled_splits": {"direct_quote": 5, "semantic_match_high": 8},
        "inputs": {
            "annotation_key": {"sha256": analysis.sha256_of_file(sample / "annotation_key.json")}
        },
    }


SUPPLEMENT_DISPLAY = {**DISPLAY, "trecho": "t v2"}


def write_supplement(folder, key, sample, plan_path):
    folder.mkdir()
    criteria = key["criteria"]
    supplement = {
        "role": "annotation",
        "kind": analysis.SUPPLEMENT_KIND,
        "sample_name": analysis.SUPPLEMENT_NAME,
        "dry_run": False,
        "splits_used": key["splits_used"],
        "base_sample": {
            "annotation_key": {"sha256": analysis.sha256_of_file(sample / "annotation_key.json")}
        },
        "plan": {"sha256": analysis.sha256_of_file(plan_path)},
        "criteria": criteria,
        "criteria_sha256": analysis.canonical_sha256(criteria),
        "strata": key["strata"],
        "items": {
            "C001": {
                "question": "trecho_sustenta",
                "stratum": "direct_quote",
                "udv_ids": ["a"],
                "tier": "quote_found",
                "quote_cue_in_trecho": True,
                "display": SUPPLEMENT_DISPLAY,
                "v1_item_id": "A001",
            }
        },
    }
    (folder / "annotation_key.json").write_text(json.dumps(supplement))
    return supplement


def test_score_command_combines_inherited_and_supplementary_judgments(tmp_path):
    key = scored_key()
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "annotation_key.json").write_text(json.dumps(key))
    sheet = tmp_path / "filled.csv"
    write_sheet(
        sheet,
        [
            {"item_id": "A001", "julgamento": "correta"},
            {"item_id": "A002", "julgamento": "parcial"},
            {"item_id": "A003", "julgamento": "falou"},
            {"item_id": "A004", "julgamento": "incorreta"},
        ],
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(scored_plan(sample)))
    supplement_dir = tmp_path / "supplement"
    write_supplement(supplement_dir, key, sample, plan_path)
    supplement_sheet = supplement_dir / "annotation.csv"
    with open(supplement_sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), delimiter=";", restval="")
        writer.writeheader()
        writer.writerow({**SUPPLEMENT_DISPLAY, "item_id": "C001", "julgamento": ""})
    args = SimpleNamespace(
        config=analysis.Path("configs/validation_sample.toml"),
        sample_dir=str(sample),
        annotation=str(sheet),
        plan=str(plan_path),
        final_test=False,
        output=str(tmp_path / "interim.json"),
        supplement_dir=str(supplement_dir),
        supplement_annotation=None,
    )
    analysis.command_score(args)
    interim = json.loads((tmp_path / "interim.json").read_text())
    assert interim["status"] == "interim"
    v2 = interim["udv_v2"]
    assert v2["judged"] == "2 of 3" and v2["judged_reannotated"] == 0
    assert v2["strata"]["direct_quote"]["judged_udvs"] == 0
    assert v2["strata"]["semantic_match_high"]["tolerant_precision"]["successes"] == 1
    assert v2["strata"]["semantic_match_high"]["population"] == 8
    assert v2["criteria"][0]["status"] == "INTERIM_NOT_EVALUABLE"
    with open(supplement_sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), delimiter=";", restval="")
        writer.writeheader()
        writer.writerow(
            {
                **SUPPLEMENT_DISPLAY,
                "item_id": "C001",
                "julgamento": "incorreta",
                "existe_trecho_melhor": "nao",
            }
        )
    args.output = str(tmp_path / "final.json")
    analysis.command_score(args)
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["status"] == "final" and final["udv_v2"]["status"] == "final"
    quotes = final["udv_v2"]["strata"]["direct_quote"]
    assert quotes["strict_precision"]["successes"] == 0 and quotes["judged_udvs"] == 1
    assert final["udv_v1"]["strata"]["direct_quote"]["strict_precision"]["successes"] == 1
    assert final["udv_v2"]["criteria"][0]["status"] == "FAIL"
    assert final["udv_v2"]["sources_by_stratum"]["direct_quote"] == {
        "inherited": 0,
        "reannotated": 1,
    }


def test_load_supplement_refuses_another_plan(tmp_path):
    key = scored_key()
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "annotation_key.json").write_text(json.dumps(key))
    plan_path = tmp_path / "plan.json"
    plan = scored_plan(sample)
    plan_path.write_text(json.dumps(plan))
    folder = tmp_path / "supplement"
    write_supplement(folder, key, sample, plan_path)
    loaded = analysis.load_supplement(folder, sample / "annotation_key.json", plan, plan_path)
    assert list(loaded["items"]) == ["C001"]
    plan_path.write_text(json.dumps({**plan, "rule": "changed"}))
    with pytest.raises(SystemExit, match="another annotation plan"):
        analysis.load_supplement(folder, sample / "annotation_key.json", plan, plan_path)
