import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from utils import translation
from utils.translation import (
    DecodingSpec,
    MissingTranslationError,
    ModelSpec,
    Segmenter,
    TranslationOutput,
    TranslationStore,
    boundary_parts,
    join_segments,
    model_signature,
    repeated_ngram,
    resolve_splits,
    translate_missing,
)
from utils.udv_pipeline import split_sentences

MODEL = ModelSpec(
    role="primary",
    name="fake/translator",
    revision="0" * 40,
    src_lang="por_Latn",
    tgt_lang="eng_Latn",
    src_token="por_Latn",
    tgt_token="eng_Latn",
    dtype="float32",
    license="none",
)
DECODING = DecodingSpec(
    num_beams=4,
    do_sample=False,
    length_penalty=1.0,
    early_stopping=False,
    no_repeat_ngram_size=0,
    repetition_penalty=1.0,
    max_input_tokens=512,
    max_new_tokens_ratio=2.0,
    max_new_tokens_margin=16,
    max_new_tokens_cap=1020,
    batch_size=2,
)
SIGNATURE = model_signature(MODEL, DECODING)
NO_JOIN = Segmenter(join_abbreviations=frozenset(), join_short_parts=False)
JOIN = Segmenter(join_abbreviations=frozenset({"Sr.", "Dra.", "V.Exa."}), join_short_parts=True)


class Interrupted(RuntimeError):
    pass


@dataclass
class FakeTranslator:
    signature: dict = field(default_factory=lambda: dict(SIGNATURE))
    device: str = "cpu"
    batch_size: int = 2
    fail_after_batches: int | None = None
    calls: list[list[str]] = field(default_factory=list)

    def model_input_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]

    def translate(self, texts: list[str]) -> list[TranslationOutput]:
        if self.fail_after_batches is not None and len(self.calls) >= self.fail_after_batches:
            raise Interrupted("simulated interruption")
        self.calls.append(list(texts))
        return [
            TranslationOutput(
                text=f"EN<{text}>",
                input_tokens=len(text.split()),
                model_input_tokens=len(text.split()) + 2,
                output_tokens=len(text.split()),
                input_truncated=False,
                hit_max_new_tokens=False,
            )
            for text in texts
        ]


def store_at(
    tmp_path: Path, writable: bool = True, segmenter: Segmenter = NO_JOIN
) -> TranslationStore:
    return TranslationStore.open(tmp_path / "cache.jsonl", SIGNATURE, segmenter, writable)


def test_lookup_raises_clear_error_for_missing_text(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with pytest.raises(MissingTranslationError) as error:
        store.lookup("Uma frase que nunca foi traduzida.")
    assert "python -m utils.translation translate --model <the condition" in str(error.value)
    assert "Uma frase" not in str(error.value)


def test_store_round_trip_and_normalized_keys(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    translate_missing(FakeTranslator(), store, ["O deputado  falou.", "Outra frase aqui."])
    reopened = store_at(tmp_path, writable=False)
    assert reopened.lookup("O deputado falou.") == "EN<O deputado falou.>"
    assert reopened.lookup("  O deputado\nfalou. ") == "EN<O deputado falou.>"
    assert reopened.translate_opinion("Outra frase aqui.") == "EN<Outra frase aqui.>"
    assert reopened.record("Outra frase aqui.")["device"] == "cpu"


def test_other_signature_is_not_served(tmp_path: Path) -> None:
    translate_missing(FakeTranslator(), store_at(tmp_path), ["O deputado falou."])
    other = replace(DECODING, num_beams=5)
    store = TranslationStore.open(tmp_path / "cache.jsonl", model_signature(MODEL, other), NO_JOIN)
    assert store.other_signature_lines == 1
    with pytest.raises(MissingTranslationError):
        store.lookup("O deputado falou.")


def test_translator_with_other_signature_is_refused(tmp_path: Path) -> None:
    fake = FakeTranslator(signature=model_signature(MODEL, replace(DECODING, num_beams=1)))
    with pytest.raises(SystemExit):
        translate_missing(fake, store_at(tmp_path), ["O deputado falou."])


def test_boundary_parts_keep_every_part_in_order() -> None:
    chunk = "Muito obrigado. Isso nos preocupa. (Palmas.) O projeto chega ao plenário amanhã. )"
    segments = boundary_parts(chunk)
    assert segments == [
        "Muito obrigado.",
        "Isso nos preocupa.",
        "(Palmas.)",
        "O projeto chega ao plenário amanhã.",
        ")",
    ]
    assert set(split_sentences(chunk)) <= set(segments)
    assert "".join(segments).replace(" ", "") == chunk.replace(" ", "")
    assert boundary_parts("   ") == []
    assert NO_JOIN.segments(chunk) == segments


def test_join_rule_merges_abbreviation_cuts_and_short_parts() -> None:
    assert JOIN.segments("Tem a palavra o Sr. Fulano de Tal, que fala agora.") == [
        "Tem a palavra o Sr. Fulano de Tal, que fala agora."
    ]
    assert JOIN.segments(
        "Muito obrigado. Isso nos preocupa. O projeto chega amanhã ao plenário."
    ) == [
        "Muito obrigado. Isso nos preocupa.",
        "O projeto chega amanhã ao plenário.",
    ]
    assert JOIN.segments("O projeto chega ao plenário amanhã. Obrigado. (Palmas.)") == [
        "O projeto chega ao plenário amanhã. Obrigado. (Palmas.)"
    ]
    assert JOIN.segments("Sim. Não.") == ["Sim. Não."]
    assert JOIN.segments("Eu gostaria que V.Exa. respondesse a pergunta do relator.") == [
        "Eu gostaria que V.Exa. respondesse a pergunta do relator."
    ]
    assert JOIN.segments("") == []


def test_join_rule_keeps_every_word_once() -> None:
    chunk = (
        "Dra. Ana. Obrigada. A proposta reduz o prazo de análise. Não. É isso. "
        "Fim da audiência pública."
    )
    units = JOIN.segments(chunk)
    assert " ".join(units) == chunk
    assert all(len(unit.split()) >= 4 for unit in units)


def test_translate_chunk_joins_parts_and_translates_each_once(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    chunk = "Não. O orçamento da saúde precisa crescer. Não. Fim da fala )"
    fake = FakeTranslator()
    translate_missing(fake, store, store.segmenter.segments(chunk))
    sent = [text for batch in fake.calls for text in batch]
    expected = ["Não.", "O orçamento da saúde precisa crescer.", "Fim da fala )"]
    assert sorted(sent) == sorted(expected)
    english = store.translate_chunk(chunk)
    assert english == (
        "EN<Não.> EN<O orçamento da saúde precisa crescer.> EN<Não.> EN<Fim da fala )>"
    )
    assert store.translate_chunk("") == ""
    translate_missing(fake, store, ["Sim.", "Tudo certo por aqui."])
    pairs = store.chunk_translations("Sim. Tudo certo por aqui.")
    assert pairs[0] == ("Sim.", store.lookup("Sim."))


def test_verbatim_parts_are_copied_and_never_sent(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    fake = FakeTranslator()
    translate_missing(fake, store, [")", "...", "Uma frase qualquer."])
    assert [text for batch in fake.calls for text in batch] == ["Uma frase qualquer."]
    assert store.lookup(")") == ")"
    assert store.translate_chunk("Uma frase qualquer. )") == "EN<Uma frase qualquer.> )"


def test_join_segments_uses_one_space_and_skips_empty_outputs() -> None:
    assert join_segments(["A.", "", "  B. ", "C."]) == "A. B. C."


def test_resume_after_interruption_translates_only_the_rest(tmp_path: Path) -> None:
    texts = [f"Frase número {index} da audiência pública." for index in range(7)]
    first = FakeTranslator(fail_after_batches=2)
    with pytest.raises(Interrupted):
        translate_missing(first, store_at(tmp_path), texts)
    resumed_store = store_at(tmp_path)
    assert len(resumed_store.entries) == 4
    second = FakeTranslator()
    result = translate_missing(second, resumed_store, texts)
    assert result["translated"] == 3
    assert sum(len(batch) for batch in second.calls) == 3
    done = {text for batch in first.calls + second.calls for text in batch}
    assert done == set(texts)
    final = store_at(tmp_path, writable=False)
    assert all(final.lookup(text) == f"EN<{text}>" for text in texts)
    assert translate_missing(FakeTranslator(), store_at(tmp_path), texts)["translated"] == 0


def test_incomplete_last_line_is_ignored_read_only_and_cut_when_writable(tmp_path: Path) -> None:
    translate_missing(FakeTranslator(), store_at(tmp_path), ["Primeira frase completa aqui."])
    path = tmp_path / "cache.jsonl"
    complete = path.read_bytes()
    path.write_bytes(complete + b'{"key": "abc", "signature')
    reader = store_at(tmp_path, writable=False)
    assert reader.incomplete_tail_bytes > 0
    assert len(reader.entries) == 1
    assert path.read_bytes() != complete
    writer = store_at(tmp_path, writable=True)
    assert path.read_bytes() == complete
    translate_missing(FakeTranslator(), writer, ["Segunda frase completa aqui."])
    assert len(store_at(tmp_path, writable=False).entries) == 2


def test_malformed_complete_line_stops(tmp_path: Path) -> None:
    translate_missing(FakeTranslator(), store_at(tmp_path), ["Primeira frase completa aqui."])
    path = tmp_path / "cache.jsonl"
    path.write_text("not json\n" + path.read_text())
    with pytest.raises(SystemExit):
        store_at(tmp_path)


def test_read_only_store_refuses_append(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        translate_missing(FakeTranslator(), store_at(tmp_path, writable=False), ["Uma frase."])


def test_length_batches_without_budget_cuts_fixed_batches() -> None:
    assert translation.length_batches([3, 9, 5, 7, 1], batch_size=2) == [[1, 3], [2, 0], [4]]


def test_length_batches_budget_shrinks_batches_of_long_texts() -> None:
    lengths = [400, 30, 410, 20, 100, 30]
    batches = translation.length_batches(lengths, batch_size=4, token_budget=820)
    assert batches == [[2, 0], [4, 1, 5, 3]]
    assert all(len(batch) * lengths[batch[0]] <= 820 for batch in batches)


def test_length_batches_keeps_a_text_longer_than_the_budget_alone() -> None:
    assert translation.length_batches([900, 10, 10], batch_size=4, token_budget=100) == [
        [0],
        [1, 2],
    ]


def test_translate_missing_applies_the_token_budget(tmp_path: Path) -> None:
    texts = ["um dois três quatro cinco seis sete oito", "um dois", "um dois três"]
    fake = FakeTranslator(batch_size=3)
    translate_missing(fake, store_at(tmp_path), texts, token_budget=12)
    assert fake.calls == [[texts[0]], [texts[2], texts[1]]]
    records = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert {record["batch_token_budget"] for record in records} == {12}
    assert [record["batch_number"] for record in records] == [1, 2, 2]


def test_cache_records_hold_provenance(tmp_path: Path) -> None:
    translate_missing(FakeTranslator(), store_at(tmp_path), ["Uma frase com cinco palavras."])
    record = json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[0])
    assert record["signature_sha256"] == translation.signature_digest(SIGNATURE)
    assert {"device", "batch_size", "batch_number", "batch_seconds", "seconds"} <= set(record)


@dataclass(frozen=True)
class SplitConfig:
    default_splits: tuple[str, ...] = ("train", "validation")
    final_test_splits: tuple[str, ...] = ("test",)


def test_split_guard_refuses_test_without_flag() -> None:
    config = SplitConfig()
    assert resolve_splits(config, None, False) == ("train", "validation")
    assert resolve_splits(config, ["validation"], False) == ("validation",)
    with pytest.raises(SystemExit):
        resolve_splits(config, ["test"], False)
    with pytest.raises(SystemExit):
        resolve_splits(config, ["validation", "test"], False)
    with pytest.raises(SystemExit):
        resolve_splits(config, ["dev"], False)
    assert resolve_splits(config, None, True) == ("train", "validation", "test")
    assert resolve_splits(config, ["test"], True) == ("test",)


def test_max_new_tokens_rule() -> None:
    assert DECODING.max_new_tokens(10) == 36
    assert DECODING.max_new_tokens(512) == 1020


def test_degenerate_repetition() -> None:
    assert repeated_ngram("the the the the the the the", 4, 4)
    assert not repeated_ngram("the the the the the the", 4, 4)
    repeated = " ".join(["we need more money"] * 4)
    assert repeated_ngram(repeated, 4, 4)
    assert not repeated_ngram("we need more money for health and education", 4, 4)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "translation.toml"
NLLB_SIGNATURE_SHA256 = "b9d9a899bcef1b762b757e3a1419167fb44032042d8f637ef29f84df427981e9"


@pytest.fixture(scope="module")
def real_config():
    return translation.load_config(CONFIG)


def test_nllb_condition_keeps_its_signature_decoding_and_units(real_config) -> None:
    nllb = translation.selected_model(real_config)
    assert nllb.key == real_config.default_model == "nllb"
    signature = model_signature(nllb, real_config.decoding)
    assert translation.signature_digest(signature) == NLLB_SIGNATURE_SHA256
    assert (real_config.decoding.batch_size, real_config.decoding.batch_token_budget) == (16, 2048)
    assert real_config.segmenter.join_short_parts
    assert {"Sr.", "V.Exa.", "art."} <= real_config.segmenter.join_abbreviations


def test_second_condition_has_its_own_signature_and_cache_file(real_config) -> None:
    nllb = translation.selected_model(real_config, "nllb")
    m2m100 = translation.selected_model(real_config, "m2m100")
    assert (m2m100.name, m2m100.license, m2m100.src_token, m2m100.tgt_token) == (
        "facebook/m2m100_418M",
        "MIT",
        "__pt__",
        "__en__",
    )
    digests = {
        translation.signature_digest(model_signature(spec, real_config.decoding))
        for spec in (nllb, m2m100)
    }
    assert len(digests) == 2
    paths = {translation.cache_path(real_config.cache_dir, spec) for spec in (nllb, m2m100)}
    assert len(paths) == 2
    with pytest.raises(SystemExit, match="unknown translation model"):
        translation.selected_model(real_config, "fallback")


@dataclass
class FakeTokenizer:
    source_id: int
    target_id: int
    eos_token_id: int = 2
    unk_token_id: int = 3
    pad_token_id: int = 1

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"__pt__": self.source_id, "__en__": self.target_id}.get(token, self.unk_token_id)

    def num_special_tokens_to_add(self) -> int:
        return 2

    def __call__(self, text: str) -> dict:
        return {"input_ids": [self.source_id, *range(10, 10 + len(text.split())), 2]}


M2M100_LIKE = replace(MODEL, src_lang="pt", tgt_lang="en", src_token="__pt__", tgt_token="__en__")


def test_check_tokenizer_accepts_the_m2m100_layout() -> None:
    info = translation.check_tokenizer(FakeTokenizer(128075, 128022), M2M100_LIKE)
    assert (info["source_token_id"], info["target_token_id"]) == (128075, 128022)
    assert info["special_tokens_per_input"] == 2


def test_check_tokenizer_refuses_unknown_language_codes() -> None:
    with pytest.raises(SystemExit, match="unknown"):
        translation.check_tokenizer(FakeTokenizer(3, 128022), M2M100_LIKE)


def test_output_length_skips_decoder_start_and_forced_target_token() -> None:
    fake = translation.Seq2SeqTranslator(
        spec=M2M100_LIKE,
        decoding=DECODING,
        tokenizer=FakeTokenizer(128075, 128022),
        model=None,
        device="cpu",
        signature=SIGNATURE,
        batch_size=2,
        target_token_id=128022,
        special_tokens=2,
        info={},
    )
    assert fake.output_length([2, 128022, 50, 51, 52, 2, 1, 1]) == (3, True)
    assert fake.output_length([2, 128022, 50, 51, 52]) == (3, False)


def test_spot_check_order_is_a_seeded_permutation() -> None:
    order = translation.spot_check_order(80, 20260924)
    assert sorted(order) == list(range(80))
    assert order == translation.spot_check_order(80, 20260924)
    assert order != list(range(80))


def test_existing_judgments_counts_filled_cells(tmp_path: Path, real_config) -> None:
    spot = real_config.spot_check
    path = tmp_path / "sheet.csv"
    rows = [{"item_id": "T01", "texto_pt": "a", "traducao_en": "b"}]
    translation.write_spot_check_csv(path, spot, rows)
    assert translation.existing_judgments(path, spot) == 0
    translation.write_spot_check_csv(path, spot, [{**rows[0], "fluencia": "3"}])
    assert translation.existing_judgments(path, spot) == 1


def condition_translator(config, key: str) -> FakeTranslator:
    spec = config.models[key]
    fake = FakeTranslator(signature=model_signature(spec, config.decoding), batch_size=16)
    fake.info = {"name": spec.name}
    return fake


def spot_args(tmp_path: Path, model: str, cached_only: bool = False):
    from argparse import Namespace

    return Namespace(model=model, cached_only=cached_only, cache_dir=tmp_path, device="cpu")


SOURCES = [
    "O orçamento da saúde precisa crescer neste ano.",
    "A comissão aprovou o requerimento da audiência.",
    "Os convidados falaram sobre a reforma tributária.",
]


def fill_condition(config, tmp_path: Path, key: str, texts: list[str]) -> Path:
    store = translation.open_store(config, key, writable=True, cache_dir=tmp_path)
    translate_missing(condition_translator(config, key), store, texts)
    return store.path


def test_spot_check_writes_only_the_selected_model_cache(
    tmp_path: Path, real_config, monkeypatch
) -> None:
    nllb_path = fill_condition(real_config, tmp_path, "nllb", SOURCES)
    nllb_path.write_bytes(nllb_path.read_bytes() + b'{"key": "cut')
    before = nllb_path.read_bytes()
    loads = []

    def fake_load(spec, decoding, device):
        loads.append(spec.key)
        return condition_translator(real_config, spec.key)

    monkeypatch.setattr(translation, "load_translator", fake_load)
    stores, runs, _ = translation.spot_check_stores(
        spot_args(tmp_path, "m2m100"), real_config, SOURCES
    )
    assert loads == ["m2m100"]
    assert nllb_path.read_bytes() == before
    assert not stores["nllb"].writable and stores["m2m100"].writable
    assert runs["m2m100"]["translated"] == 3 and runs["nllb"]["store_opened"] == "read-only"
    again, runs, _ = translation.spot_check_stores(
        spot_args(tmp_path, "m2m100"), real_config, SOURCES
    )
    assert loads == ["m2m100"] and not again["m2m100"].writable
    assert runs["m2m100"]["already_cached"] == 3


def test_spot_check_stops_when_another_model_lacks_a_unit(
    tmp_path: Path, real_config, monkeypatch
) -> None:
    nllb_path = fill_condition(real_config, tmp_path, "nllb", SOURCES[:2])
    before = nllb_path.read_bytes()
    monkeypatch.setattr(translation, "load_translator", lambda *args: pytest.fail("loaded"))
    with pytest.raises(SystemExit, match="1 of 3 spot-check units have no nllb translation"):
        translation.spot_check_stores(spot_args(tmp_path, "m2m100"), real_config, SOURCES)
    m2m100 = translation.cache_path(tmp_path, real_config.models["m2m100"])
    assert not m2m100.exists()
    assert nllb_path.read_bytes() == before
    fill_condition(real_config, tmp_path, "nllb", SOURCES)
    with pytest.raises(SystemExit, match="3 of 3 spot-check units have no m2m100 translation"):
        translation.spot_check_stores(spot_args(tmp_path, "nllb", True), real_config, SOURCES)


def test_lookup_failure_names_the_condition_of_the_store(tmp_path: Path, real_config) -> None:
    store = translation.open_store(real_config, "m2m100", cache_dir=tmp_path)
    assert store.condition == "m2m100" and not store.writable
    with pytest.raises(MissingTranslationError, match="translate --model m2m100 for the splits"):
        store.lookup("Uma frase que nunca foi traduzida.")


def test_spot_check_display_changes_only_typographic_quotes(real_config) -> None:
    spot = real_config.spot_check
    text = "It\u2019s the group\u2019s \u201cplan\u201d \u2014 a \u2018test\u2019, it's done."
    assert spot.display(text) == "It's the group's \"plan\" \u2014 a 'test', it's done."
    assert spot.display("No quotes here.") == "No quotes here."
    raw = {**real_config.source["translation"]["spot_check"]}
    raw["display_normalization"] = {"\u2019": "ab"}
    with pytest.raises(SystemExit, match="one character to one ASCII one"):
        translation.parse_spot_check(raw)


def test_blinding_check_lists_rows_told_apart_by_a_character() -> None:
    items = [
        {"item_id": "T01", "unit": "U01", "model": "nllb"},
        {"item_id": "T02", "unit": "U01", "model": "m2m100"},
        {"item_id": "T03", "unit": "U02", "model": "nllb"},
        {"item_id": "T04", "unit": "U02", "model": "m2m100"},
    ]
    sheet = [
        {"item_id": "T01", "traducao_en": "He said it, clearly."},
        {"item_id": "T02", "traducao_en": "He said it \u2014 clearly."},
        {"item_id": "T03", "traducao_en": "Yes, it is."},
        {"item_id": "T04", "traducao_en": "Yes, it is."},
    ]
    check = translation.blinding_check(sheet, items)
    assert check["characters_of_one_model"] == {"U+2014": {"m2m100": ["T02"]}}
    assert check["identifiable_items"] == ["T02"]
    assert check["identifiable_units"] == ["U01"]
    assert check["items_of_identifiable_units"] == ["T01", "T02"]
