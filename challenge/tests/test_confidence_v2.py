import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from utils import confidence_v2 as cv
from utils import grounding_scorers as gs
from utils.nli_verifier_experiments import PremiseUnit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "confidence_v2.toml"


class FakeScorer:
    def __init__(self) -> None:
        self.signature = "fake-signature"
        self.info = {"name": "fake"}
        self.calls: list[tuple[str, str]] = []

    def score(self, pairs):
        self.calls.extend(pairs)
        return [
            gs.PairScore(value=len(premise) / 100, tokens=len(premise.split()), truncated=False)
            for premise, _ in pairs
        ]


def unit(unit_id: str, items: tuple[str, ...], hearing: int = 1, split: str = "train"):
    return PremiseUnit(unit_id, hearing, split, f"hipotese {unit_id}", items)


def test_config_declares_candidates_and_drops_llm_judges():
    config = cv.load_config(CONFIG)
    assert set(config.order) == set(config.candidates)
    assert "qwen3_judge" not in config.order and "granite_guardian" not in config.order
    assert set(config.raw["new"]["not_run"]) == {"qwen3_judge", "granite_guardian"}
    for spec in config.candidates.values():
        assert len(spec.revision) == 40
    laya = config.raw["laya"]
    assert "p8_similarity" not in laya["support_questions"]
    assert set(laya["new_questions"]["pt"]) == set(laya["new_questions"]["keys"])
    assert set(laya["new_questions"]["en"]) == set(laya["new_questions"]["keys"])


def test_score_rows_max_concatenated_and_cache(tmp_path):
    units = [unit("a", ("x" * 10, "", "y" * 30)), unit("b", ("z" * 20,)), unit("c", ("", ""))]
    scorer = FakeScorer()
    cache = gs.ScoreCache(tmp_path / "cache.jsonl", scorer.signature)
    rows = cv.score_rows(scorer, units, True, cache)
    assert rows[0]["scores"]["max"] == pytest.approx(0.30)
    assert rows[0]["scores"]["concatenated"] == pytest.approx(41 / 100)
    assert rows[0]["items"][1] is None
    assert rows[1]["scores"]["concatenated"] == pytest.approx(0.20)
    assert rows[1]["concatenated"] is None
    assert rows[2]["scores"] == {"max": None, "concatenated": None}
    first_calls = len(scorer.calls)
    assert first_calls == 4
    again = gs.ScoreCache(tmp_path / "cache.jsonl", scorer.signature)
    assert cv.score_rows(scorer, units, True, again) == rows
    assert len(scorer.calls) == first_calls
    assert again.hits == 4 and again.computed == 0


def test_cache_ignores_other_signatures(tmp_path):
    path = tmp_path / "cache.jsonl"
    first = gs.ScoreCache(path, "one")
    first.append(["k"], [gs.PairScore(0.5, 3, False)])
    assert gs.ScoreCache(path, "two").get("k") is None
    assert gs.ScoreCache(path, "one").get("k")["value"] == 0.5


def test_filled_uses_zero_for_probabilities_and_minimum_for_raw_scores():
    assert cv.filled([0.4, None], True).tolist() == [0.4, 0.0]
    assert cv.filled([-2.0, 3.0, None], False).tolist() == [-2.0, 3.0, -2.0]


def test_two_way_probability_and_combinations():
    logits = np.array([[0.0, 1.0, 3.0]])
    assert gs.two_way_probability(logits, 2, 1)[0] == pytest.approx(1 / (1 + np.exp(-2)))
    systems = {"cosine_serafim": np.array([0.1, 0.5, 0.9]), "e3x_primary": np.array([3, 2, 1.0])}
    combined = cv.with_combinations(systems)
    assert combined["rank_mean"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3])
    assert combined["rank_max"].tolist() == pytest.approx([1.0, 2 / 3, 1.0])


def test_delong_matches_sklearn_auc_and_is_null_for_identical_scores():
    rng = np.random.default_rng(0)
    positive = rng.random(80) > 0.4
    first = rng.random(80) + positive
    second = rng.random(80)
    result = cv.delong_test(first, second, positive)
    assert result["auc_first"] == pytest.approx(roc_auc_score(positive, first), abs=1e-4)
    assert result["auc_second"] == pytest.approx(roc_auc_score(positive, second), abs=1e-4)
    assert result["p_value"] < 0.05
    same = cv.delong_test(first, first, positive)
    assert same["delta"] == 0 and same["p_value"] == 1.0


def test_paired_comparisons_use_same_replicates_and_holm():
    rng = np.random.default_rng(1)
    positive = rng.random(200) > 0.3
    hearings = np.repeat(np.arange(20), 10)
    systems = {
        "cosine_serafim": rng.random(200),
        "good": positive + rng.random(200),
        "noise": rng.random(200),
    }
    result = cv.evaluate_set(systems, positive, hearings, 200, 3, 0.95)
    pairs = [("good", "cosine_serafim"), ("noise", "cosine_serafim")]
    entries = cv.paired_comparisons(systems, positive, result["_replicates"], pairs, 0.95)
    assert len(entries) == 4
    good_auc = next(e for e in entries if e["candidate"] == "good" and e["metric"] == "roc_auc")
    good_aurc = next(e for e in entries if e["candidate"] == "good" and e["metric"] == "aurc")
    assert good_auc["delta"] > 0 and good_auc["interval"]["low"] > 0
    assert good_aurc["delta"] < 0
    assert all(e["p_holm"] >= e["p_value"] for e in entries)
    assert cv.main_answer([good_auc]) == "better"
    assert set(result["systems"]["good"]) >= set(cv.REPORTED_METRICS)


def test_laya_aggregates_spread_and_merge():
    per_unit = [
        [{"a": 0.9, "b": 0.1, "c": 0.5}, {"a": 0.4, "b": 0.4, "c": 0.4}],
        [],
    ]
    assert cv.aggregate_scores(per_unit, ["a", "b", "c"], "mean").tolist() == pytest.approx(
        [0.5, 0.0]
    )
    assert cv.aggregate_scores(per_unit, ["a", "b", "c"], "median").tolist() == [0.5, 0.0]
    assert cv.aggregate_scores(per_unit, ["a", "b", "c"], "min").tolist() == [0.4, 0.0]
    std, spread = cv.spread_scores(per_unit, ["a", "b", "c"])
    assert std[0] == pytest.approx(np.std([0.9, 0.1, 0.5]))
    assert spread[0] == pytest.approx(0.8)
    assert cv.aggregate_scores(per_unit, ["a"], "mean").tolist() == [0.9, 0.0]
    with pytest.raises(ValueError):
        cv.aggregate_scores(per_unit, ["a"], "max")
    merged = cv.merge_signals(per_unit, [[{"n": 1.0}, {"n": 0.0}], []])
    assert merged[0][0] == {"a": 0.9, "b": 0.1, "c": 0.5, "n": 1.0}
    assert std[1] == 0.0
    with pytest.raises(SystemExit):
        cv.merge_signals(per_unit, [[{"n": 1.0}], []])


def test_added_value_pairs_and_split_extra():
    extra = {
        "laya_x_mean_q11": np.zeros(2),
        "laya_x_spread_q11": np.zeros(2),
        "laya_x_n1_entails": np.zeros(2),
    }
    assert cv.added_value_pairs(extra) == [("laya_x_mean_q11", "laya_x_mean_q7")]
    others, spreads = cv.split_extra(extra)
    assert set(spreads) == {"laya_x_spread_q11"}
    assert "laya_x_n1_entails" in others


def test_laya_rows_orient_true_probability():
    answers = {
        ("p1", "hipotese a"): {"n1": SimpleNamespace(probabilities={"true": 0.8, "false": 0.2})},
        ("p1 p2", "hipotese a"): {"n1": SimpleNamespace(probabilities={"true": 0.3, "false": 0.7})},
        ("p2", "hipotese a"): {"n1": SimpleNamespace(probabilities={"true": 0.1, "false": 0.9})},
    }
    rows = cv.laya_rows([unit("a", ("p1", "p2", ""))], answers, ["n1"], True)
    assert [item and item["signals"]["n1"] for item in rows[0]["items"]] == [0.8, 0.1, None]
    assert rows[0]["concatenated"]["signals"]["n1"] == 0.3
    per_unit = cv.item_signals({"a": rows[0]}, ["a"], "max")
    assert cv.aggregate_scores(per_unit, ["n1"], "mean").tolist() == [0.8]


def test_spread_flag_table_counts_negative_rates():
    positive = np.array([True, False, True, True, False, True, True, True])
    spread = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.1, 0.3, 0.2])
    hearings = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    table = cv.spread_flag_table(spread, 0.5, positive, hearings, 50, 0, 0.95)
    assert table["flagged"] == 3 and table["unflagged"] == 5
    assert table["negative_rate_flagged"] == pytest.approx(2 / 3, abs=1e-4)
    assert table["negative_rate_unflagged"] == 0.0


def test_annotated_udv_ids_reads_only_the_question(tmp_path):
    key = tmp_path / "key.json"
    items = {
        "A001": {"question": "trecho_sustenta", "udv_ids": ["udv-2"]},
        "A002": {"question": "outra", "udv_ids": ["udv-3", "udv-4"]},
        "A003": {"question": "trecho_sustenta", "udv_ids": ["udv-1"]},
    }
    key.write_text(json.dumps({"items": items}))
    assert cv.annotated_udv_ids(key, "trecho_sustenta") == ["udv-1", "udv-2"]
    items["A004"] = {"question": "trecho_sustenta", "udv_ids": ["udv-5", "udv-6"]}
    key.write_text(json.dumps({"items": items}))
    with pytest.raises(SystemExit):
        cv.annotated_udv_ids(key, "trecho_sustenta")


def test_smoke_decision_rules():
    spec = cv.load_config(CONFIG).candidates["hhem_open"]
    scored = {"status": "scored", "projected_hours": 1.0, "truncation": {}}
    assert cv.smoke_decision(scored, spec, 5.0)[0] == "run"
    assert cv.smoke_decision({**scored, "projected_hours": 6.0}, spec, 5.0)[0] == "dropped"
    failed = {**scored, "probe": {"passed": False, "max_abs_gap": 0.2}}
    assert cv.smoke_decision(failed, spec, 5.0)[0] == "dropped"
    assert cv.smoke_decision({"status": "load_failed", "error": "x"}, None, 5.0)[0] == "dropped"
