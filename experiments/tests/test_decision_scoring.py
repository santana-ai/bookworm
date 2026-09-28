import tomllib
from pathlib import Path

import numpy as np
import pytest

from experiments.verifier.decision_models import DecisionAnswer
from experiments.verifier.decision_scoring import (
    BatteryError,
    check_selection,
    component_agreement,
    consensus_votes,
    fit_stacked,
    item_signals,
    opinion_scores,
    order_changes,
    parse_battery,
    score_names,
)
from experiments.verifier.stats import apply_holm, bootstrap_p_value

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nli_verifier.toml"
ALL = (
    "p1_nli",
    "p2_nli_reversed",
    "p3_inferable",
    "p4_supports",
    "p5_position",
    "p6_position_reversed",
    "p7_coverage",
    "p8_similarity",
)


@pytest.fixture(scope="module")
def battery():
    with open(CONFIG, "rb") as f:
        return parse_battery(tomllib.load(f)["decision_battery"])


def answer(battery, key: str, values: dict[str, float]) -> DecisionAnswer:
    question = battery.questions[key].question
    return DecisionAnswer(
        key, question.type, {name: values[name] for name in question.option_names}
    )


def answers(battery, entail: float, entail_reversed: float, contra: float = 0.1) -> dict:
    return {
        "p1_nli": answer(
            battery,
            "p1_nli",
            {"entailment": entail, "neutral": 1 - entail - contra, "contradiction": contra},
        ),
        "p2_nli_reversed": answer(
            battery,
            "p2_nli_reversed",
            {
                "contradiction": 0.05,
                "neutral": 0.95 - entail_reversed,
                "entailment": entail_reversed,
            },
        ),
        "p3_inferable": answer(battery, "p3_inferable", {"false": 0.4, "true": 0.6}),
        "p4_supports": answer(battery, "p4_supports", {"false": 0.3, "true": 0.7}),
        "p5_position": answer(
            battery,
            "p5_position",
            {"same_position": 0.5, "opposite_position": 0.2, "no_position": 0.2, "unrelated": 0.1},
        ),
        "p6_position_reversed": answer(
            battery,
            "p6_position_reversed",
            {"unrelated": 0.1, "no_position": 0.1, "opposite_position": 0.1, "same_position": 0.7},
        ),
        "p7_coverage": answer(
            battery, "p7_coverage", {"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.5, "4": 0.0}
        ),
        "p8_similarity": answer(
            battery, "p8_similarity", {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}
        ),
    }


def test_battery_reversed_questions_mirror_the_forward_ones(battery):
    forward = battery.questions["p1_nli"].question
    reverse = battery.questions["p2_nli_reversed"].question
    assert reverse.option_names == tuple(reversed(forward.option_names))
    assert reverse.instructions == forward.instructions
    assert dict(zip(reverse.option_names, reverse.option_texts, strict=True)) == dict(
        zip(forward.option_names, forward.option_texts, strict=True)
    )
    assert battery.questions["p2_nli_reversed"].support_option == "entailment"
    assert battery.questions["p6_position_reversed"].reverses == "p5_position"


def test_signals_map_the_reversed_order_back_by_option_name(battery):
    values = item_signals(battery, ALL, answers(battery, entail=0.6, entail_reversed=0.8))
    assert values["p1_nli"] == pytest.approx(0.6)
    assert values["p2_nli_reversed"] == pytest.approx(0.8)
    assert values["nli_order_mean"] == pytest.approx(0.7)
    assert values["position_order_mean"] == pytest.approx(0.6)
    assert values["p7_coverage"] == pytest.approx(2.5 / 4)
    assert values["p8_similarity"] == pytest.approx(1.0)
    assert values["not_contradiction"] == pytest.approx(0.9)
    assert values["not_opposite"] == pytest.approx(0.8)
    panel = np.mean([0.7, 0.6, 0.7, 0.6, 2.5 / 4])
    assert values["panel"] == pytest.approx(panel)


def test_opinion_scores_aggregate_over_chunks(battery):
    first = item_signals(battery, ALL, answers(battery, 0.2, 0.3, contra=0.5))
    second = item_signals(battery, ALL, answers(battery, 0.9, 0.7, contra=0.05))
    joined = item_signals(battery, ALL, answers(battery, 0.4, 0.4))
    scores = opinion_scores(battery, ALL, [first, second], joined, True)
    assert set(scores) == set(score_names(battery, ALL, True))
    assert scores["max.p1_nli"] == pytest.approx(0.9)
    assert scores["max.nli_order_mean"] == pytest.approx(0.8)
    assert scores["min.not_contradiction"] == pytest.approx(0.5)
    assert scores["max.panel"] == pytest.approx(max(first["panel"], second["panel"]))
    assert scores["concatenated.p1_nli"] == pytest.approx(0.4)
    empty = opinion_scores(battery, ALL, [], None, True)
    assert set(empty) == set(scores) and set(empty.values()) == {0.0}


def test_dropped_reverse_questions_leave_the_forward_signal(battery):
    selected = check_selection(
        battery,
        [q for q in ALL if q not in ("p2_nli_reversed", "p6_position_reversed", "p8_similarity")],
    )
    values = item_signals(battery, selected, answers(battery, 0.6, 0.8))
    assert values["nli_order_mean"] == pytest.approx(0.6)
    assert "max.p8_similarity" not in score_names(battery, selected, False)
    assert not any(
        name.startswith("concatenated.") for name in score_names(battery, selected, False)
    )
    with pytest.raises(BatteryError):
        check_selection(battery, ["p2_nli_reversed", "p3_inferable"])
    with pytest.raises(BatteryError):
        check_selection(battery, [q for q in ALL if q != "p7_coverage"])


def test_order_changes_count_argmax_flips(battery):
    flip = {key: a.record() for key, a in answers(battery, 0.6, 0.1).items()}
    same = {key: a.record() for key, a in answers(battery, 0.6, 0.8).items()}
    rows = [
        {"items": [{"answers": flip}, None, {"answers": same}], "concatenated": {"answers": same}},
        {"items": [], "concatenated": None},
    ]
    result = order_changes(battery, ALL, rows)["nli_order_mean"]
    assert result["chunk"] == {"premises": 2, "changed": 1, "change_rate": 0.5}
    assert result["concatenated"]["changed"] == 0
    assert result["pooled"]["premises"] == 3


def test_consensus_stacked_and_agreement():
    scores = {"a": np.array([0.9, 0.1, 0.6]), "b": np.array([0.8, 0.2, 0.1])}
    votes = consensus_votes(scores, {"a": 0.5, "b": 0.5})
    assert votes.tolist() == [1.0, 0.0, 0.5]
    agreement = component_agreement({"a": votes >= 0.5, "b": np.array([True, False, True])})
    assert agreement["share_all_agree"] == 1.0
    rng = np.random.default_rng(0)
    features = {"x": rng.normal(size=200), "y": rng.normal(size=200)}
    labels = features["x"] + 0.3 * rng.normal(size=200) > 0
    model = fit_stacked(features, labels, ["x", "y"], 1.0, 42)
    again = fit_stacked(features, labels, ["x", "y"], 1.0, 42)
    assert np.allclose(model.predict(features), again.predict(features))
    assert model.describe()["coefficients_standardized"]["x"] > 0


def test_bootstrap_p_value_and_holm_with_missing():
    assert bootstrap_p_value(np.array([0.1] * 99)) == pytest.approx(2 / 100)
    assert bootstrap_p_value(np.array([-0.1, 0.1])) == pytest.approx(1.0)
    assert bootstrap_p_value(np.array([np.nan])) is None
    entries = [{"p_value": 0.01}, {"p_value": None}, {"p_value": 0.04}]
    apply_holm(entries)
    assert entries[0]["p_holm"] == pytest.approx(0.03)
    assert entries[2]["p_holm"] == pytest.approx(0.08)
    assert entries[1]["p_holm"] == 1.0 and entries[1]["p_holm_counts_missing_as_one"]
