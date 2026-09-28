from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bookworm import write_jsonl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.verifier import nli_exploration as exploration
from experiments.verifier import udv_verifier as uv
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.config import ScorerSpec
from experiments.verifier.nli.scoring import score_file
from experiments.verifier.nli_exploration import (
    BATTERY_SIGNALS,
    Candidate,
    check_reading,
    load_exploration_config,
    load_scorer,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "udv_verifier.toml"
EXPLORATION = ROOT / "configs" / "nli_verifier_exploration_v2.toml"
PANEL = ("nli_order_mean", "p3_inferable", "p4_supports", "position_order_mean", "p7_coverage")


def record(udv_id: str, tier: str, text: str | None, hearing: int = 1) -> dict:
    evidence = None
    if text is not None:
        kind = "direct_quote" if tier == "quote_found" else "semantic_similarity"
        score = None if tier == "quote_found" else 0.6
        evidence = {"text": text, "support_type": kind, "score": score}
    return {
        "id": udv_id,
        "hearing_id": hearing,
        "proposition": f"  Proposta   {udv_id} ",
        "evidence": evidence,
        "tier": tier,
    }


def decision_signals(seed: float) -> dict[str, float]:
    values = {name: (seed + index / 10) % 1.0 for index, name in enumerate(BATTERY_SIGNALS)}
    values["nli_order_mean"] = (values["p1_nli"] + values["p2_nli_reversed"]) / 2
    values["position_order_mean"] = (values["p5_position"] + values["p6_position_reversed"]) / 2
    values["panel"] = float(np.mean([values[name] for name in PANEL]))
    return values


def fake_row(unit: PremiseUnit, spec: ScorerSpec, seed: float) -> dict:
    header = {
        "id": unit.unit_id,
        "hearing_id": unit.hearing_id,
        "split": unit.split,
        "item_count": 1,
        "nonempty_items": 1,
        "no_premise": False,
    }
    if spec.kind == "laya":
        signals = decision_signals(seed)
        scores = {f"max.{k}": v for k, v in signals.items()}
        scores |= {f"concatenated.{k}": v for k, v in signals.items()}
        item = {"signals": signals}
        return {**header, "items": [item], "concatenated": item, "scores": scores}
    if spec.kind == "nli":
        item = {"probabilities": {"entailment": seed, "neutral": 1 - seed, "contradiction": 0.0}}
        scores = {"max.entailment": seed, "concatenated.entailment": seed}
        return {**header, "items": [item], "concatenated": item, "scores": scores}
    item = {"cosine": seed, "sentence_max": seed}
    scores = {"max.cosine": seed, "sentence_max.cosine": seed}
    return {**header, "items": [item], "concatenated": None, "scores": scores}


def spec(key: str, kind: str, language: str = "pt") -> ScorerSpec:
    return ScorerSpec(key, kind, key, "0", (), 512, "float32", 1, (), (), language=language)


SPECS = [
    spec("laya_en_en", "laya", "en"),
    spec("laya_multi_pt", "laya"),
    spec("xnli_mdeberta", "nli"),
    spec("cosine_serafim", "cosine"),
]


@pytest.fixture
def configs(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(exploration, "SCORES_ROOT", tmp_path)
    return uv.load_config(CONFIG), load_exploration_config(EXPLORATION)


def test_units_keep_only_evidence_tiers_and_count_the_rest():
    records = [
        record("a", "quote_found", "Frase   citada."),
        record("b", "semantic_match_weak", "Outra frase."),
        record("c", "person_not_resolved", None),
        record("d", "no_evidence", None),
    ]
    units, unscored = uv.udv_units(records, ("quote_found", "semantic_match_weak"), {1: "train"})
    assert [u.unit_id for u in units] == ["a", "b"]
    assert units[0].items == ("Frase citada.",)
    assert units[0].hypothesis == "Proposta a"
    assert units[0].split == "train"
    assert unscored == {"person_not_resolved": 1, "no_evidence": 1}


def test_units_refuse_an_evidence_tier_without_text():
    with pytest.raises(SystemExit):
        uv.udv_units([record("a", "quote_found", "  ")], ("quote_found",), {1: "train"})
    with pytest.raises(SystemExit):
        uv.udv_units([record("a", "quote_found", "Texto.", 9)], ("quote_found",), {1: "train"})


def test_score_scorers_routes_english_scorers_through_the_translation(monkeypatch):
    units = [PremiseUnit("a", 1, "train", "Proposta.", ("Frase.",))]
    store = SimpleNamespace(
        translate_chunk=lambda text: f"EN {text}", translate_opinion=lambda text: f"EN {text}"
    )
    translations = SimpleNamespace(store=lambda _: store)
    verifier = SimpleNamespace(probe_premises=("P",), probe_hypotheses=("H",))
    seen: dict[str, tuple] = {}

    def fake_score(spec_units, spec_, verifier_, device, concatenate_single, prefix, probes, mode):
        seen[spec_.key] = (spec_units[0], concatenate_single, probes)
        return [fake_row(u, spec_, 0.5) for u in spec_units], {"timing": {}}

    monkeypatch.setattr(uv, "translation_summary", lambda *args: {"model": "fake"})
    results = uv.score_scorers(units, SPECS[:2], verifier, "cpu", translations, "t", fake_score)
    english, single, probes = seen["laya_en_en"]
    assert english.hypothesis == "EN Proposta." and english.items == ("EN Frase.",)
    assert single is True and probes == [("EN P", "EN H")]
    assert seen["laya_multi_pt"][0] is units[0]
    assert results["laya_en_en"][1]["translation"] == {"model": "fake"}


def test_primary_reads_the_declared_scorers(configs):
    config, exploration_config = configs
    candidate = uv.primary_candidate(exploration_config, config.primary)
    assert len(candidate.features) == 2 * 24 + 5
    assert set(uv.candidate_scorers(candidate)) == set(config.scorers)
    uv.check_scorer_list(config, candidate)


def test_fake_scores_flow_to_probabilities(configs):
    config, exploration_config = configs
    candidate = uv.primary_candidate(exploration_config, config.primary)
    units = [PremiseUnit(f"u{i}", 1, "train", "H", ("P",)) for i in range(4)]
    for index, spec_ in enumerate(SPECS):
        rows = [fake_row(u, spec_, (0.1 * (i + 1) + 0.05 * index) % 1) for i, u in enumerate(units)]
        write_jsonl(rows, score_file(config.run_dir, spec_.key, uv.UDV_SPLIT))
    ids = [u.unit_id for u in units]
    data = {}
    for key in config.scorers:
        kind, _ = exploration_config.scorers[key]
        data[key] = load_scorer(key, kind, config.name, uv.UDV_SPLIT, ids, exploration_config)
    check_reading(exploration_config, data)
    pools = uv.pool_agreement(candidate, data)
    assert all(gap == 0.0 for gap in pools["max_abs_gap_by_scorer"].values())
    rng = np.random.default_rng(0)
    matrix = rng.random((40, len(candidate.features)))
    labels = rng.random(40) > 0.5
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression().fit(scaler.transform(matrix), labels)
    fitted = uv.FittedPrimary(candidate, scaler, model, 0.5, {})
    probabilities = fitted.probabilities(data)
    assert probabilities.shape == (4,)
    assert ((probabilities > 0) & (probabilities < 1)).all()
    secondary = uv.candidate_scores(uv.feature_candidate(config.secondary), data)
    assert secondary.tolist() == pytest.approx(
        [decision_signals((0.1 * (i + 1) + 0.05) % 1)["p4_supports"] for i in range(4)]
    )


def test_pool_agreement_stops_when_pools_differ():
    data = {
        "toy": exploration.ScorerData(
            "toy",
            "decision",
            Path("toy"),
            {"a": [[0.4]]},
            {"a": np.array([0.9])},
            {},
        )
    }
    candidate = Candidate("c", "learned", "toy", (), "learned", ("toy:a:max", "toy:a:concatenated"))
    with pytest.raises(SystemExit):
        uv.pool_agreement(candidate, data)


def test_refit_check_compares_every_recorded_value():
    fitted = uv.FittedPrimary(
        Candidate("k", "learned", "s", (), "learned"),
        StandardScaler(),
        LogisticRegression(),
        0.75,
        {"c": 0.01, "inner_means": {"0.01": 0.8}, "coefficients": {"f": 0.1}},
    )
    final_test = {
        "learned": {"k": {"c": 0.01, "inner_means": {"0.01": 0.8}, "coefficients": {"f": 0.1}}},
        "results": {"k": {"threshold": 0.75}},
    }
    assert all(uv.refit_check(fitted, final_test, "k")["checks"].values())
    final_test["results"]["k"]["threshold"] = 0.7
    with pytest.raises(SystemExit):
        uv.refit_check(fitted, final_test, "k")


def test_summaries_and_lowest():
    values = np.array([0.9, 0.1, 0.5, 0.3])
    groups = ["x", "x", "y", "y"]
    summary = uv.grouped_summaries(values, groups, (0.0, 0.5, 1.0))
    assert summary["x"]["quantiles"]["0.0"] == 0.1 and summary["all"]["n"] == 4
    shares = uv.supported_shares(values >= 0.5, groups)
    assert shares["x"] == {"n": 2, "supported": 1, "share": 0.5}
    assert shares["all"]["supported"] == 2
    records = [
        record(name, tier, "t")
        for name, tier in zip("abcd", ["quote_found"] * 3 + ["x"], strict=True)
    ]
    lowest = uv.lowest_udvs(records, values, values, ("quote_found",), 2)
    assert [row["udv_id"] for row in lowest["quote_found"]] == ["b", "c"]


def test_output_rows_keep_unscored_udvs_with_null_scores():
    records = [record("a", "quote_found", "t"), record("b", "no_evidence", None)]
    rows = uv.output_rows(records, {1: "test"}, {"a": (0.8, True, 0.6)})
    assert rows[0]["scored"] and rows[0]["supported_at_train_threshold"] is True
    assert rows[1] == {
        "udv_id": "b",
        "hearing_id": 1,
        "split": "test",
        "tier": "no_evidence",
        "support_type": None,
        "scored": False,
        "primary_probability": None,
        "supported_at_train_threshold": None,
        "p4_supports": None,
    }


def test_evidence_score_check_on_sentence_runs_compares_every_semantic_udv():
    records = [
        record("a", "semantic_match_high", "Uma frase inteira de evidência."),
        record("b", "quote_found", "Outra frase com aspas no texto."),
        record("c", "semantic_match_weak", "Mais uma frase de evidência aqui."),
    ]
    check = uv.evidence_score_check(
        np.array([0.6, 0.1, 0.4]),
        np.array([0.6, np.nan, 0.6]),
        records,
        ("semantic_match_high", "semantic_match_weak"),
    )
    assert check == {
        "rule": uv.SENTENCE_CHECK_RULE,
        "n": 2,
        "max_abs_gap": 0.2,
        "within_1e-4": 1,
    }


def test_evidence_score_check_on_window_runs_separates_texts_the_encoder_did_not_read():
    records = [
        record(
            "a", "semantic_match_high", "Primeira frase da janela aqui. Segunda frase da janela."
        ),
        record(
            "b", "semantic_match_high", "Primeira frase da janela aqui. Curta. Segunda frase dela."
        ),
        record("c", "quote_found", "Citação direta encontrada na fala."),
    ]
    check = uv.evidence_score_check(
        np.array([0.6, 0.5, 0.1]),
        np.array([0.6, 0.6, np.nan]),
        records,
        ("semantic_match_high", "semantic_match_weak"),
        "window2",
    )
    assert check["semantic_udvs"] == 2
    assert (check["n"], check["max_abs_gap"], check["within_1e-4"]) == (1, 0.0, 1)
    assert check["encoded_text_differs"] == {
        "n": 1,
        "max_abs_gap": 0.1,
        "within_1e-4": 0,
        "udv_ids": ["b"],
    }


def test_semantic_unit_is_read_from_the_coverage_pipeline(tmp_path):
    udv_path = tmp_path / "run.jsonl"
    (tmp_path / "run_coverage.json").write_text('{"pipeline": {"semantic_unit": "window2"}}')
    assert uv.udv_semantic_unit(udv_path) == "window2"
    assert uv.evidence_cosine_feature("window2") == "cosine_serafim:cosine:max"
    (tmp_path / "run_coverage.json").write_text('{"pipeline": {}}')
    assert uv.udv_semantic_unit(udv_path) == "sentence"
    assert uv.evidence_cosine_feature("sentence") == "cosine_serafim:sentence_max:max"
