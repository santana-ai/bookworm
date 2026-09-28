from pathlib import Path

import numpy as np
import pytest

from experiments.common.udv_run import load_config as load_udv_config
from experiments.retrieval.data import HearingData, Query, SpeakerContext, Unit
from experiments.retrieval.experiments import decision_rerank_spec, load_config
from experiments.retrieval.models import (
    RERANK_OFFSET,
    DecisionRerankRetriever,
    Ranking,
    Runtime,
    order_by,
)
from experiments.verifier.decision_models import DecisionQuestion, FakeDecisionModel, LayaSpec
from experiments.verifier.decision_scoring import BatteryQuestion

CONFIG = Path("configs/retrieval_experiments.toml")
TEXTS = ["zero", "um", "dois", "tres", "quatro", "cinco"]
BASE_SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
SUPPORT = {"zero": 0.1, "um": 0.2, "dois": 0.9, "tres": 0.9, "quatro": 0.99, "cinco": 1.0}
TOP_K = 4


class FixedBase:
    retriever_id = "fixed"

    def __init__(self) -> None:
        self.closed = False

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        return {
            query.query_id: Ranking(order_by(BASE_SCORES), BASE_SCORES, BASE_SCORES)
            for query in queries
        }

    def describe(self) -> dict:
        return {"kind": "fixed"}

    def close(self) -> None:
        self.closed = True


def support_answer(state: dict, question: DecisionQuestion) -> dict[str, float]:
    value = SUPPORT[state["premise"]]
    return {"false": 1.0 - value, "true": value}


def hearing_with_queries() -> tuple[HearingData, list[Query]]:
    units = [
        Unit(f"u{index}", 0, (index,), text, 10 * index, 10 * index + len(text))
        for index, text in enumerate(TEXTS)
    ]
    context = SpeakerContext("1:0", 1, (0,), [], {"sentence": units})
    hearing = HearingData(1, "train", "", [], {}, [], contexts={"1:0": context})
    queries = [
        Query("q1", "nli", "train", 1, "1:0", "a opiniao", {"sentence": (2,)}),
        Query("q2", "nli", "train", 1, "1:0", "outra opiniao", {"sentence": (0,)}),
    ]
    return hearing, queries


def make_spec(tmp_path: Path, top_k: int = TOP_K):
    config = load_config(CONFIG)
    raw = {**config.retrievers["rerank_laya_p4"], "top_k": top_k}
    return decision_rerank_spec("rerank_laya_p4", raw, "cpu", tmp_path / "answers")


def runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        device="cpu",
        embeddings_dir=tmp_path / "embeddings",
        rerank_dir=tmp_path / "rerank",
        shard_size=100,
        udv_config=load_udv_config(Path("configs/udv.toml")),
    )


def test_spec_reads_the_laya_scorer_and_the_battery_question(tmp_path):
    spec = make_spec(tmp_path)
    assert spec.question.question.key == "p4_supports"
    assert spec.question.question.type == "noul"
    assert spec.base == "serafim_335m"
    assert spec.laya.subfolder == "multilingual"
    assert spec.laya.truncate_key == "premise"
    assert spec.laya.cache_dir == tmp_path / "answers"


def test_spec_refuses_translated_scorers_and_other_revisions(tmp_path):
    raw = load_config(CONFIG).retrievers["rerank_laya_p4"]
    with pytest.raises(SystemExit, match="language pt"):
        decision_rerank_spec("x", {**raw, "scorer": "laya_multi_en"}, "cpu", tmp_path)
    with pytest.raises(SystemExit, match="revision"):
        decision_rerank_spec("x", {**raw, "revision": "0" * 40}, "cpu", tmp_path)
    with pytest.raises(SystemExit, match="does not ask"):
        decision_rerank_spec("x", {**raw, "question": "p9_missing"}, "cpu", tmp_path)


def test_top_k_is_sorted_by_support_and_the_rest_keeps_the_base_order(tmp_path):
    models: list[FakeDecisionModel] = []

    def load(spec: LayaSpec) -> FakeDecisionModel:
        models.append(FakeDecisionModel(answer_fn=support_answer))
        return models[-1]

    hearing, queries = hearing_with_queries()
    base = FixedBase()
    retriever = DecisionRerankRetriever(make_spec(tmp_path), base, runtime(tmp_path), load)
    rankings = retriever.rank_hearing(hearing, "sentence", queries)
    ranking = rankings["q1"]
    assert ranking.order.tolist() == [2, 3, 1, 0, 4, 5]
    assert ranking.sort_key[2] == pytest.approx(RERANK_OFFSET + 0.9)
    assert ranking.sort_key[4] == -5.0
    assert np.isnan(ranking.display[5])
    assert ranking.display[3] == pytest.approx(0.9, abs=1e-6)
    assert len(models) == 1
    assert models[0].calls == [(2 * TOP_K, ("p4_supports",))]
    retriever.close()
    assert base.closed
    details = retriever.describe()
    assert details["kind"] == "decision_rerank"
    assert details["counts"]["pairs_scored"] == 2 * TOP_K
    assert details["decision_model"]["kind"] == "fake"


def test_state_puts_the_unit_in_the_premise_and_the_query_in_the_hypothesis(tmp_path):
    seen: list[dict] = []

    def record(state: dict, question: DecisionQuestion) -> dict[str, float]:
        seen.append(state)
        return support_answer(state, question)

    hearing, queries = hearing_with_queries()
    retriever = DecisionRerankRetriever(
        make_spec(tmp_path),
        FixedBase(),
        runtime(tmp_path),
        lambda spec: FakeDecisionModel(answer_fn=record),
    )
    retriever.rank_hearing(hearing, "sentence", queries[:1])
    assert [list(state) for state in seen] == [["premise", "hypothesis"]] * TOP_K
    assert {state["hypothesis"] for state in seen} == {"a opiniao"}
    assert [state["premise"] for state in seen] == TEXTS[:TOP_K]


def test_scores_are_cached_by_pair_and_reused_without_loading_the_model(tmp_path):
    hearing, queries = hearing_with_queries()
    first = DecisionRerankRetriever(
        make_spec(tmp_path),
        FixedBase(),
        runtime(tmp_path),
        lambda spec: FakeDecisionModel(answer_fn=support_answer),
    )
    expected = first.rank_hearing(hearing, "sentence", queries)
    first.close()

    def refuse(spec: LayaSpec) -> FakeDecisionModel:
        raise AssertionError("the model must not load when every pair is cached")

    second = DecisionRerankRetriever(make_spec(tmp_path), FixedBase(), runtime(tmp_path), refuse)
    again = second.rank_hearing(hearing, "sentence", queries)
    for query in queries:
        assert again[query.query_id].order.tolist() == expected[query.query_id].order.tolist()
    assert second.describe()["counts"].get("model_loads", 0) == 0


def test_the_question_and_device_separate_the_score_stores(tmp_path):
    spec = make_spec(tmp_path)
    retriever = DecisionRerankRetriever(spec, FixedBase(), runtime(tmp_path))
    parts = retriever.store.directory.relative_to(tmp_path / "rerank").parts
    assert parts[-1] == "cpu"
    assert parts[-2].startswith("p4_supports-")
    other = BatteryQuestion(
        DecisionQuestion("p4_supports", "noul", "Other wording?", ("false", "true"), (None, None))
    )
    changed = DecisionRerankRetriever(
        spec.__class__(**{**spec.__dict__, "question": other}), FixedBase(), runtime(tmp_path)
    )
    assert changed.store.directory != retriever.store.directory
