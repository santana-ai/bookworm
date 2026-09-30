import json
from typing import Any

import pytest
from conftest import StubEncoder

from bookworm import (
    CachedEncoder,
    EvidenceSettings,
    HearingRecord,
    UdvConfig,
    build_udvs,
    find_opinion_turn_quote_match,
    resolve_person_speech,
    split_into_turns,
)
from bookworm.errors import ConfigError
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.coverage import pipeline_description, summarize_run
from bookworm.udv.export import export_hearing
from bookworm.udv.quotes import (
    DEFAULT_QUOTE_POLICY,
    extend_quote_match,
    find_quote_end,
    quote_part_count,
    quote_suffixes,
)
from bookworm.udv.verify import coverage_settings, verify_udv_run
from bookworm.udv.windows import embedding_label, person_units

HOSPITAL = "O hospital regional está fechado há dois anos por falta de verba."
FAMILIES = "Nós precisamos de uma solução imediata para as famílias."
CARE = "Sem isso, a população continua sem atendimento nenhum."
THANKS = "Quero agradecer a presença dos colegas aqui."
GREETING = "Bom dia a todos."
TRANSCRIPT = (
    "O SR. PRESIDENTE (Carlos Lima) - Declaro aberta a reunião de hoje sobre saúde pública. "
    f"A SRA. ANA SOUZA - {GREETING} Obrigada. {HOSPITAL}  {FAMILIES} {CARE}\n{THANKS} "
    "O SR. PRESIDENTE (Carlos Lima) - Agradeço a presença de todos os convidados hoje."
)
MULTI_SENTENCE_OPINION = (
    'Afirmou que "o hospital regional está fechado há dois anos por falta de verba. Nós '
    'precisamos de uma solução imediata para as famílias".'
)
UNFINISHED_OPINION = (
    'Disse que "sem isso, a população continua sem atendimento nenhum. E o governo federal '
    'precisa assumir a responsabilidade agora".'
)
SINGLE_OPINION = 'Disse que "quero agradecer a presença dos colegas aqui".'
SEMANTIC_OPINION = "Pediu uma solução para as famílias sem hospital."
WINDOW_HOSPITAL_FAMILIES = f"{HOSPITAL} {FAMILIES}"
V2_VECTORS = {
    SEMANTIC_OPINION: (1.0, 0.0, 0.0, 0.0),
    WINDOW_HOSPITAL_FAMILIES: (1.0, 0.0, 0.0, 0.0),
}
THRESHOLD = 0.6
V2_SETTINGS = EvidenceSettings(THRESHOLD, semantic_unit="window2", quote_extent="full_quote")


def hearing(opinions: tuple[str, ...]) -> HearingRecord:
    return HearingRecord.model_validate_json(
        json.dumps(
            {
                "id": 7,
                "materia": "Matéria.",
                "metadados": {
                    "assunto": "Saúde",
                    "envolvidos": [{"nome": "Ana Souza", "cargo": "Médica", "opinioes": opinions}],
                },
                "transcricao": TRANSCRIPT,
            }
        )
    )


def ana_turns() -> list[Any]:
    turns, _ = resolve_person_speech("Ana Souza", split_into_turns(TRANSCRIPT))
    return list(turns)


def v2_coverage(records: Any, run: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    return summarize_run(
        "v2",
        records,
        run.people,
        run.hearings,
        encoder_runtime={},
        hearing_seconds=run.hearing_seconds,
        config_source={"evidence": evidence},
        settings=V2_SETTINGS,
        environment={},
    )


def test_quote_suffixes_drop_trailing_punctuation() -> None:
    suffixes = quote_suffixes("a b c d e f g h fim.", DEFAULT_QUOTE_POLICY)
    assert suffixes == [("d e f g h fim", 6), ("f g h fim", 4), ("g h fim", 3)]
    assert quote_suffixes("curto", DEFAULT_QUOTE_POLICY) == []


def test_quote_part_count_counts_sentence_parts() -> None:
    assert quote_part_count("Uma frase. Outra frase aqui! E mais") == 3
    assert quote_part_count("uma frase só") == 1


def test_multi_sentence_quote_covers_the_whole_quote() -> None:
    turns = ana_turns()
    match = find_opinion_turn_quote_match(MULTI_SENTENCE_OPINION, turns)
    assert match is not None
    assert match.words == 10
    assert match.sentence == HOSPITAL
    extent = extend_quote_match(MULTI_SENTENCE_OPINION, match, turns)
    assert extent.rule == "suffix"
    assert extent.suffix_words == 6
    assert extent.text == f"{HOSPITAL} {FAMILIES}"


def test_quote_whose_end_is_not_found_covers_as_many_sentences_as_the_quote() -> None:
    turns = ana_turns()
    match = find_opinion_turn_quote_match(UNFINISHED_OPINION, turns)
    assert match is not None
    assert match.words == 6
    extent = extend_quote_match(UNFINISHED_OPINION, match, turns)
    assert extent.rule == "sentence_count"
    assert extent.suffix_words is None
    assert extent.text == f"{CARE} {THANKS}"


def test_single_sentence_quote_keeps_the_v1_sentence() -> None:
    turns = ana_turns()
    match = find_opinion_turn_quote_match(SINGLE_OPINION, turns)
    assert match is not None
    extent = extend_quote_match(SINGLE_OPINION, match, turns)
    assert extent.rule == "suffix"
    assert extent.text == match.sentence == THANKS


def test_suffix_beyond_the_span_limit_is_ignored() -> None:
    quote = "alfa beta gama delta épsilon zeta fim do texto"
    text = f"alfa beta gama delta épsilon zeta. {'palavra ' * 40}fim do texto."
    assert find_quote_end(quote, text, 0, 32) is None
    near = "alfa beta gama delta épsilon zeta e fim do texto."
    assert find_quote_end(quote, near, 0, 32) == (len(near) - 1, 3)


def test_suffix_must_end_after_the_prefix() -> None:
    quote = "fim do texto e depois alfa beta gama delta"
    text = "fim do texto antes. alfa beta gama delta fim do texto."
    start = text.index("alfa")
    assert find_quote_end(quote, text, start, start + 10) == (len(text) - 14, 4)


def test_window_units_keep_exact_offsets() -> None:
    units = person_units(ana_turns(), TRANSCRIPT, 2)
    assert [unit.text for unit in units] == [
        f"{GREETING} {HOSPITAL}",
        WINDOW_HOSPITAL_FAMILIES,
        f"{FAMILIES} {CARE}",
        f"{CARE} {THANKS}",
    ]
    assert units[0].evidence_text == f"{GREETING} Obrigada. {HOSPITAL}"
    for unit in units:
        span = TRANSCRIPT[unit.span.start_char : unit.span.end_char]
        assert normalize_whitespace(span) == unit.evidence_text
    assert units[1].sentence_positions == (1, 2)


def test_turn_with_one_sentence_gives_one_unit() -> None:
    turns, _ = resolve_person_speech("Carlos Lima", split_into_turns(TRANSCRIPT))
    units = person_units(turns, TRANSCRIPT, 3)
    assert [unit.text for unit in units] == [
        "Declaro aberta a reunião de hoje sobre saúde pública.",
        "Agradeço a presença de todos os convidados hoje.",
    ]


def test_build_with_windows_and_full_quotes() -> None:
    record_hearing = hearing(
        (MULTI_SENTENCE_OPINION, UNFINISHED_OPINION, SINGLE_OPINION, SEMANTIC_OPINION)
    )
    encoder = StubEncoder(V2_VECTORS)
    run = build_udvs([record_hearing], CachedEncoder(encoder), V2_SETTINGS)
    by_id = {record.id: record for record in run.records}
    texts = {record_id: record.evidence for record_id, record in by_id.items()}
    assert all(evidence is not None for evidence in texts.values())
    first = by_id["udv-7-0-0"]
    assert first.tier == "quote_found"
    assert first.evidence is not None
    assert first.evidence.text == f"{HOSPITAL} {FAMILIES}"
    assert first.evidence.start_char is not None
    assert first.evidence.end_char is not None
    span = TRANSCRIPT[first.evidence.start_char : first.evidence.end_char]
    assert normalize_whitespace(span) == first.evidence.text
    semantic = by_id["udv-7-0-3"]
    assert semantic.tier == "semantic_match_high"
    assert semantic.evidence is not None
    assert semantic.evidence.text == WINDOW_HOSPITAL_FAMILIES
    assert semantic.evidence.score == pytest.approx(1.0)
    assert encoder.calls[0] == [unit.text for unit in person_units(ana_turns(), TRANSCRIPT, 2)]
    assert embedding_label("window2", 7) == "window2s_7"


def test_verify_accepts_a_v2_run_and_rejects_it_under_v1_rules() -> None:
    record_hearing = hearing(
        (MULTI_SENTENCE_OPINION, UNFINISHED_OPINION, SINGLE_OPINION, SEMANTIC_OPINION)
    )
    run = build_udvs([record_hearing], CachedEncoder(StubEncoder(V2_VECTORS)), V2_SETTINGS)
    v2 = {
        "embedding_threshold": THRESHOLD,
        "semantic_unit": "window2",
        "quote_extent": "full_quote",
    }
    report = verify_udv_run("v2", run.records, v2_coverage(run.records, run, v2), [record_hearing])
    assert report.problems == {}
    v1 = {"embedding_threshold": THRESHOLD}
    report = verify_udv_run("v2", run.records, v2_coverage(run.records, run, v1), [record_hearing])
    assert set(report.problems) >= {"quote_text_mismatch", "evidence_not_person_sentence"}


def test_coverage_settings_read_the_run_config() -> None:
    settings = coverage_settings(
        {"config": {"evidence": {"embedding_threshold": 0.5, "semantic_unit": "window3"}}}
    )
    assert (settings.semantic_unit, settings.quote_extent) == ("window3", "prefix_sentence")
    with pytest.raises(ConfigError, match="semantic_unit"):
        coverage_settings(
            {"config": {"evidence": {"embedding_threshold": 0.5, "semantic_unit": "turn"}}}
        )


def test_pipeline_description_adds_keys_only_for_v2() -> None:
    assert "semantic_unit" not in pipeline_description(settings=EvidenceSettings(THRESHOLD))
    description = pipeline_description(settings=V2_SETTINGS)
    assert description["semantic_unit"] == "window2"
    assert description["quote_extent"] == "full_quote"
    assert description["quote_suffix_lengths"] == [6, 4, 3]


def minimal_config(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {"lds_path": "lds.jsonl", "sha256": "0" * 64},
        "encoder": {"name": "m", "revision": "r", "batch_size": 1, "device": "cpu"},
        "evidence": evidence,
        "run": {"seed": 1, "output_dir": "out", "cache_dir": "cache"},
    }


def test_udv_config_reads_the_v2_options() -> None:
    default = UdvConfig.from_mapping(minimal_config({"embedding_threshold": 0.45}))
    assert (default.semantic_unit, default.quote_extent) == ("sentence", "prefix_sentence")
    v2 = UdvConfig.from_mapping(
        minimal_config(
            {"embedding_threshold": 0.5, "semantic_unit": "window2", "quote_extent": "full_quote"}
        )
    )
    assert (v2.semantic_unit, v2.quote_extent) == ("window2", "full_quote")
    with pytest.raises(ConfigError):
        UdvConfig.from_mapping(
            minimal_config({"embedding_threshold": 0.5, "quote_extent": "whole_turn"})
        )


def test_export_of_a_window_run_ranks_the_windows_the_run_encoded() -> None:
    record_hearing = hearing((SEMANTIC_OPINION,))
    encoder = CachedEncoder(StubEncoder(V2_VECTORS))
    run = build_udvs([record_hearing], encoder, V2_SETTINGS)
    pipeline = pipeline_description(V2_SETTINGS.quote_policy, V2_SETTINGS)
    payload = export_hearing(
        record_hearing, run.records, encoder, run_name="v2", pipeline=pipeline, settings=V2_SETTINGS
    )
    assert payload["run"]["semantic_unit"] == "window2"
    assert payload["run"]["quote_extent"] == "full_quote"
    udv = payload["udvs"][0]
    units = person_units(ana_turns(), TRANSCRIPT, 2)
    assert udv["n_candidates"] == len(units)
    assert [candidate["text"] for candidate in udv["candidates"]] == sorted(
        (unit.evidence_text for unit in units),
        key=lambda text: text != WINDOW_HOSPITAL_FAMILIES,
    )
    top = udv["candidates"][0]
    assert (top["start"], top["end"]) == (
        udv["evidence"]["start_char"],
        udv["evidence"]["end_char"],
    )
    with pytest.raises(ConfigError, match="pipeline of the config in semantic_unit"):
        export_hearing(record_hearing, run.records, encoder, run_name="v2", pipeline=pipeline)
