import copy
import dataclasses
from pathlib import Path

import pytest
from bookworm import write_json

from experiments.verifier.decision_models import AnswerCache, FakeDecisionModel, choice_question
from experiments.verifier.nli import decision as nli_decision
from experiments.verifier.nli.benchmark import PremiseUnit, concatenated_premise
from experiments.verifier.nli.config import (
    ScorerSpec,
    declared_systems,
    load_config,
    narrowed_splits,
    parse_declaration,
    parse_families,
    parse_twins,
    scorer_translation_model,
)
from experiments.verifier.nli.cross_encoder import open_logit_cache, portuguese_probes, request_key
from experiments.verifier.nli.decision import (
    decision_truncation_summary,
    laya_spec,
    score_units_decision,
)
from experiments.verifier.nli.plan import compute_estimate, plan_units
from experiments.verifier.nli.scoring import check_score_names, score_report_file
from experiments.verifier.nli.translated import (
    Translations,
    check_translations,
    english_probes,
    english_units,
    translation_texts,
)
from experiments.verifier.translate import config as translate_config
from experiments.verifier.translate.config import Segmenter
from experiments.verifier.translate.store import TranslationOutput, TranslationStore

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nli_verifier.toml"
SIGNATURE = {"model": "fake/translator", "revision": "0" * 40}
SEGMENTER = Segmenter(join_abbreviations=frozenset(), join_short_parts=False)
UNIT = PremiseUnit(
    unit_id="nli-1-0-0",
    hearing_id=1,
    split="validation",
    hypothesis="O deputado apoia o projeto de lei.",
    items=(
        "O deputado disse que apoia o projeto de lei.",
        "",
        "A comissão discutiu o orçamento da saúde.",
        "O deputado disse que apoia o projeto de lei.",
    ),
)
EMPTY_UNIT = PremiseUnit("nli-2-0-0", 2, "validation", "Uma opinião.", ("", ""))
FAMILY_SIZES = {"main": 10, "translation_model": 11, "jev": 4}
M2M100_TWINS = {
    "laya_multi_en_m2m100": "laya_multi_en",
    "laya_en_en_m2m100": "laya_en_en",
    "xnli_mdeberta_en_m2m100": "xnli_mdeberta_en",
}
TWIN_FIELDS_THAT_DIFFER = {"key", "translation_model", "source"}


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG)


def english(text: str) -> str:
    return f"EN[{text}]"


def filled_store(tmp_path: Path, texts: list[str]) -> TranslationStore:
    store = TranslationStore.open(tmp_path / "t.jsonl", SIGNATURE, SEGMENTER, writable=True)
    outputs = [TranslationOutput(english(t), 1, 1, 1, False, False) for t in texts]
    store.append(
        texts, outputs, {"device": "cpu", "batch_size": 1, "batch": 1, "batch_seconds": 1.0}
    )
    return store


def test_v1_scorers_are_unchanged(config):
    xnli = config.scorers["xnli_mdeberta"]
    assert xnli.language == "pt" and xnli.derived == ()
    assert xnli.scores == (
        "max.entailment",
        "min.not_contradiction",
        "concatenated.entailment",
        "concatenated.not_contradiction",
    )
    assert config.evaluation_scorers == ("cosine_serafim", "xnli_mdeberta", "assin2_mdeberta")
    assert config.primary_system == "xnli_mdeberta.max.entailment"


def test_v2_declaration_names_declared_systems(config):
    declaration = config.declarations["nli_verifier_v2"]
    assert declaration.primary_system == "laya_multi_pt.max.panel"
    assert len(declaration.comparisons) == sum(FAMILY_SIZES.values())
    assert set(declaration.imported) == {"cosine_serafim", "xnli_mdeberta", "assin2_mdeberta"}
    systems = declared_systems(config.scorers)
    for comparison in declaration.comparisons:
        assert comparison["system"] in systems
    assert "laya_multi_pt.max.consensus" in systems
    raw = copy.deepcopy(config.source["declarations"]["nli_verifier_v2"])
    raw["comparisons"][0]["system"] = "laya_multi_pt.max.unknown"
    with pytest.raises(SystemExit, match="undeclared systems"):
        parse_declaration("x", raw, config.scorers, config.source["evaluation"])


def test_v2_families_split_the_comparisons(config):
    declaration = config.declarations["nli_verifier_v2"]
    families = {family.name: family.comparisons for family in declaration.families}
    assert list(families) == list(FAMILY_SIZES)
    assert {name: len(members) for name, members in families.items()} == FAMILY_SIZES
    members = [name for family in families.values() for name in family]
    assert sorted(members) == sorted(c["name"] for c in declaration.comparisons)
    assert len(set(members)) == len(members)
    assert all(name.startswith("T") for name in families["translation_model"])
    jev_systems = {
        c["system"].split(".", 1)[0]
        for c in declaration.comparisons
        if c["name"] in families["jev"]
    }
    assert jev_systems <= {"jev_en", "jev_pt"}
    raw = copy.deepcopy(config.source["declarations"]["nli_verifier_v2"])
    raw["families"]["main"]["comparisons"].append(families["jev"][0])
    with pytest.raises(SystemExit, match="exactly one family"):
        parse_families("x", raw, [c["name"] for c in declaration.comparisons])
    raw = copy.deepcopy(config.source["declarations"]["nli_verifier_v2"])
    raw["family_order"] = ["main", "jev"]
    with pytest.raises(SystemExit, match="every family once"):
        parse_families("x", raw, [c["name"] for c in declaration.comparisons])


def test_english_scorers_name_a_translation_condition(config):
    conditions = translate_config.load_config(config.translation_config_path).models
    english = [spec for spec in config.scorers.values() if spec.language == "en"]
    assert {spec.key for spec in english} >= set(M2M100_TWINS) | set(M2M100_TWINS.values())
    assert all(spec.translation_model in conditions for spec in english)
    assert all(
        spec.translation_model is None for spec in config.scorers.values() if spec.language != "en"
    )
    for twin, first in M2M100_TWINS.items():
        a, b = config.scorers[twin], config.scorers[first]
        assert (a.translation_model, b.translation_model) == ("m2m100", "nllb")
        for field in dataclasses.fields(ScorerSpec):
            if field.name not in TWIN_FIELDS_THAT_DIFFER:
                assert getattr(a, field.name) == getattr(b, field.name), field.name
    with pytest.raises(SystemExit, match="needs translation_model"):
        scorer_translation_model("x", {}, "en")
    with pytest.raises(SystemExit, match="only for language en"):
        scorer_translation_model("x", {"translation_model": "nllb"}, "pt")


def test_laya_twins_share_a_cache_file_without_sharing_keys(config, tmp_path):
    laya_twins = [twin for twin in M2M100_TWINS if config.scorers[twin].kind == "laya"]
    assert laya_twins == ["laya_multi_en_m2m100", "laya_en_en_m2m100"]
    for twin in laya_twins:
        assert laya_spec(config.scorers[twin], config, "cpu") == laya_spec(
            config.scorers[M2M100_TWINS[twin]], config, "cpu"
        )
    question = choice_question("p1_nli", "Relation?", [("entailment", "a"), ("neutral", "b")])
    cache = AnswerCache.open(tmp_path / "laya.jsonl", "laya@x@repo@rev@multilingual@cpu")
    hypothesis = "The deputy supports the bill."
    nllb = {"premise": "The deputy said he supports the bill.", "hypothesis": hypothesis}
    m2m100 = {"premise": "The deputy said that he supports the bill.", "hypothesis": hypothesis}
    assert cache.key(question, nllb) != cache.key(question, m2m100)
    assert cache.key(question, nllb) == cache.key(question, dict(nllb))
    other_hypothesis = {**nllb, "hypothesis": "The deputy backs the bill."}
    assert cache.key(question, nllb) != cache.key(question, other_hypothesis)


def test_nli_twins_write_separate_logit_files(config, tmp_path):
    local = dataclasses.replace(config, cache_dir=tmp_path)
    first = open_logit_cache(local, config.scorers["xnli_mdeberta_en"], "cpu")
    twin = open_logit_cache(local, config.scorers["xnli_mdeberta_en_m2m100"], "cpu")
    assert first.path != twin.path and first.path.parent == twin.path.parent == tmp_path
    assert first.signature == twin.signature
    hypothesis = "The deputy supports the bill."
    assert request_key(first.signature, "He supports it.", hypothesis) != request_key(
        twin.signature, "He backs it.", hypothesis
    )


def test_plan_units_use_each_model_store_or_mark_a_proxy(config, tmp_path):
    probes = portuguese_probes(config)
    reader = TranslationStore.open(tmp_path / "a.jsonl", SIGNATURE, SEGMENTER)
    texts = translation_texts([UNIT], probes, reader)
    full = filled_store(tmp_path / "nllb", texts)
    empty = TranslationStore.open(tmp_path / "m2m100.jsonl", SIGNATURE, SEGMENTER)
    conditions = translate_config.load_config(config.translation_config_path)
    stores = Translations(conditions, {"nllb": full, "m2m100": empty})
    units, source = plan_units(config.scorers["laya_multi_en"], [UNIT], config, stores)
    assert source["texts"] == "english" and source["translation_model"] == "nllb"
    assert units[0].hypothesis == english(UNIT.hypothesis)
    twin = config.scorers["laya_multi_en_m2m100"]
    units, source = plan_units(twin, [UNIT], config, stores)
    assert source["texts"] == "portuguese_proxy" and source["translation_model"] == "m2m100"
    assert source["missing_translations"] == source["distinct_texts"] == len(texts)
    assert units == [UNIT]
    units, source = plan_units(twin, [UNIT], config, None)
    assert source["texts"] == "portuguese_proxy" and units == [UNIT]
    assert plan_units(config.scorers["laya_multi_pt"], [UNIT], config, None)[1] == {
        "texts": "portuguese"
    }


def test_test_split_needs_final_test(config):
    with pytest.raises(SystemExit, match="final-test"):
        narrowed_splits(config, False, ["test"])
    assert narrowed_splits(config, False, ["validation"]) == ("validation",)


def test_english_lookup_failure_names_the_model_cache_and_count(config, tmp_path):
    store = TranslationStore.open(tmp_path / "empty.jsonl", SIGNATURE, SEGMENTER)
    texts = translation_texts([UNIT], portuguese_probes(config), store)
    for key, model in (("laya_multi_en", "nllb"), ("laya_multi_en_m2m100", "m2m100")):
        with pytest.raises(SystemExit) as error:
            check_translations(config.scorers[key], [UNIT], config, store)
        message = str(error.value)
        assert "no translation" in message and str(tmp_path / "empty.jsonl") in message
        assert key in message and f"translation model {model}" in message
        assert f"translate --model {model}" in message
        assert f"{len(texts)} of {len(texts)} distinct texts" in message
        assert SIGNATURE["model"] in message
        assert UNIT.hypothesis not in message and UNIT.items[0] not in message


def test_english_units_keep_positions_and_empty_chunks(config, tmp_path):
    probes = portuguese_probes(config)
    reader = TranslationStore.open(tmp_path / "a.jsonl", SIGNATURE, SEGMENTER)
    texts = translation_texts([UNIT], probes, reader)
    store = filled_store(tmp_path / "b", texts)
    assert (
        check_translations(config.scorers["laya_multi_en"], [UNIT], config, store)["missing"] == 0
    )
    (unit,) = english_units([UNIT], store)
    assert unit.hypothesis == english(UNIT.hypothesis)
    assert unit.items[0] == english(UNIT.items[0]) and unit.items[1] == ""
    assert unit.items[0] == unit.items[3]
    assert concatenated_premise(unit, " ", True) == " ".join(
        english(item) for item in UNIT.items if item
    )
    assert english_probes(config, store)[0][1] == english(probes[0][1])


def test_decision_rows_from_a_fake_model(config, monkeypatch):
    spec = config.scorers["laya_multi_pt"]
    model = FakeDecisionModel()
    monkeypatch.setattr(nli_decision, "load_decision_model", lambda *args: model)
    rows, details = score_units_decision(
        [UNIT, EMPTY_UNIT], spec, config, "cpu", True, portuguese_probes(config), "live"
    )
    check_score_names(rows, spec)
    row, empty = rows
    assert row["items"][1] is None and row["items"][0] == row["items"][3]
    assert row["concatenated"]["items_joined"] == 3
    chunk_panels = [row["items"][i]["signals"]["panel"] for i in (0, 2)]
    assert row["scores"]["max.panel"] == pytest.approx(max(chunk_panels))
    p1 = [row["items"][i]["answers"]["p1_nli"]["probabilities"] for i in (0, 2)]
    assert row["scores"]["min.not_contradiction"] == pytest.approx(
        min(1 - p["contradiction"] for p in p1)
    )
    assert row["scores"]["concatenated.panel"] == pytest.approx(
        row["concatenated"]["signals"]["panel"]
    )
    assert empty["no_premise"] and set(empty["scores"].values()) == {0.0}
    assert details["label_probes"]["total"] == 3
    assert details["timing"]["premise_texts"] == 3
    assert model.calls[0][0] == 3 and len(model.calls[0][1]) == 8
    summary = decision_truncation_summary(rows, spec.questions)
    assert summary["items_scored"] == 3 and summary["concatenated_premises"] == 1


def test_twins_are_declared_and_checked(config):
    declaration = config.declarations["nli_verifier_v2"]
    assert declaration.twins == M2M100_TWINS
    raw = copy.deepcopy(config.source["declarations"]["nli_verifier_v2"])
    raw["twins"] = {"laya_multi_en_m2m100": "laya_en_en"}
    with pytest.raises(SystemExit, match="must equal laya_en_en"):
        parse_twins("x", raw, config.scorers, tuple(raw["scorers"]))
    raw = copy.deepcopy(config.source["declarations"]["nli_verifier_v2"])
    raw["compute"]["local_scorers"].remove("laya_en_en_m2m100")
    with pytest.raises(SystemExit, match="both scorers of the twins"):
        parse_twins("x", raw, config.scorers, tuple(raw["scorers"]))


def write_smoke_report(directory: Path, key: str, kind: str) -> None:
    if kind == "laya":
        timing = {"seconds_per_input_token": 1.0}
    else:
        timing = {"computed": 10, "requests": 10, "seconds": 10.0, "model_input_tokens": 10}
    report = {"kind": kind, "timing": timing, "model": {"device": "cpu"}}
    write_json(report, score_report_file(directory, key))


def fake_plan(spec: ScorerSpec) -> dict:
    if spec.kind != "laya":
        return {"model_input_tokens": 1000}
    per_mode = {"chunk": {"model_input_tokens": 1000}, "concatenated": {"model_input_tokens": 1000}}
    return {"by_question": {question: copy.deepcopy(per_mode) for question in spec.questions}}


def test_compute_estimate_drops_twins_together_and_needs_every_local_scorer(config, tmp_path):
    declaration = config.declarations["nli_verifier_v2"]
    local = declaration.source["compute"]["local_scorers"]
    for key in local:
        write_smoke_report(tmp_path, key, config.scorers[key].kind)
    plans = {key: fake_plan(config.scorers[key]) for key in local}
    estimate = compute_estimate(plans, tmp_path, declaration)
    assert [step["step"] for step in estimate["steps_applied"]] == list(
        declaration.source["compute"]["drop_order"]
    )
    for step in estimate["steps_applied"]:
        named = declaration.source["compute"]["drop_steps"][step["step"]]["scorers"]
        assert set(step["scorers"]) == set(named) | {
            twin for twin, first in M2M100_TWINS.items() if first in named
        }
    after = estimate["hours_after_steps"]
    for twin, first in M2M100_TWINS.items():
        assert after[twin] == after[first]
        assert estimate["hours_full"][twin] == estimate["hours_full"][first]
    assert after["xnli_mdeberta_en_m2m100"] == estimate["hours_full"]["xnli_mdeberta_en_m2m100"]
    assert after["laya_multi_pt"] == estimate["hours_full"]["laya_multi_pt"]
    assert after["laya_multi_en_m2m100"] < estimate["hours_full"]["laya_multi_en_m2m100"]
    partial = {key: plan for key, plan in plans.items() if key != "laya_en_en_m2m100"}
    with pytest.raises(SystemExit, match="lacks \\['laya_en_en_m2m100'\\]"):
        compute_estimate(partial, tmp_path, declaration)
