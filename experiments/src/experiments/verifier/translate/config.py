"""Translation configuration: models, decoding, segmentation and the spot check."""

import math
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.common.splits import SPLIT_NAMES
from experiments.common.transcript import (
    SENTENCE_BOUNDARY_PATTERN,
    is_sentence,
    normalize_whitespace,
)

Record = dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    role: str
    name: str
    revision: str
    src_lang: str
    tgt_lang: str
    src_token: str
    tgt_token: str
    dtype: str
    license: str
    key: str = ""


@dataclass(frozen=True)
class DecodingSpec:
    num_beams: int
    do_sample: bool
    length_penalty: float
    early_stopping: bool
    no_repeat_ngram_size: int
    repetition_penalty: float
    max_input_tokens: int
    max_new_tokens_ratio: float
    max_new_tokens_margin: int
    max_new_tokens_cap: int
    batch_size: int
    batch_token_budget: int = 0

    def max_new_tokens(self, longest_input_tokens: int) -> int:
        budget = math.ceil(self.max_new_tokens_ratio * longest_input_tokens)
        return min(self.max_new_tokens_cap, budget + self.max_new_tokens_margin)

    def generation_kwargs(self) -> Record:
        return {
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class Segmenter:
    join_abbreviations: frozenset[str]
    join_short_parts: bool

    def needs_join(self, text: str) -> bool:
        if text.split()[-1] in self.join_abbreviations:
            return True
        return self.join_short_parts and not is_sentence(text)

    def segments(self, chunk: str) -> list[str]:
        units: list[str] = []
        pending = ""
        for part in boundary_parts(chunk):
            text = f"{pending} {part}" if pending else part
            if self.needs_join(text):
                pending = text
                continue
            units.append(text)
            pending = ""
        if pending and units:
            units[-1] = f"{units[-1]} {pending}"
        elif pending:
            units.append(pending)
        return units

    def describe(self) -> Record:
        return {
            "join_abbreviations": sorted(self.join_abbreviations),
            "join_short_parts": self.join_short_parts,
        }


@dataclass(frozen=True)
class SpotCheckSpec:
    name: str
    output_dir: Path
    split: str
    size: int
    seed: int
    item_id_prefix: str
    delimiter: str
    encoding: str
    columns: tuple[str, ...]
    judgment_columns: tuple[str, ...]
    readme: str
    models: tuple[str, ...] = ()
    order_seed: int = 0
    display_normalization: tuple[tuple[str, str], ...] = ()

    def display(self, text: str) -> str:
        return text.translate({ord(glyph): shown for glyph, shown in self.display_normalization})


@dataclass(frozen=True)
class TranslationConfig:
    nli_config_path: Path
    lds_sha256: str
    nli_path: Path
    nli_sha256: str
    benchmark_path: Path
    benchmark_report_path: Path
    manifest_path: Path
    default_splits: tuple[str, ...]
    final_test_splits: tuple[str, ...]
    probe_premises: tuple[str, ...]
    probe_hypotheses: tuple[str, ...]
    models: dict[str, ModelSpec]
    default_model: str
    decoding: DecodingSpec
    segmenter: Segmenter
    degenerate_ngram: int
    degenerate_min_count: int
    cache_dir: Path
    output_dir: Path
    spot_check: SpotCheckSpec
    seed: int
    device: str
    hf_hub_offline: bool
    progress_every: int
    source: Record = field(default_factory=dict)


def parse_model(key: str, raw: Record) -> ModelSpec:
    return ModelSpec(
        key=key,
        role=raw["role"],
        name=raw["name"],
        revision=raw["revision"],
        src_lang=raw["src_lang"],
        tgt_lang=raw["tgt_lang"],
        src_token=raw["src_token"],
        tgt_token=raw["tgt_token"],
        dtype=raw["dtype"],
        license=raw["license"],
    )


def parse_decoding(raw: Record) -> DecodingSpec:
    decoding = DecodingSpec(
        num_beams=raw["num_beams"],
        do_sample=raw["do_sample"],
        length_penalty=float(raw["length_penalty"]),
        early_stopping=raw["early_stopping"],
        no_repeat_ngram_size=raw["no_repeat_ngram_size"],
        repetition_penalty=float(raw["repetition_penalty"]),
        max_input_tokens=raw["max_input_tokens"],
        max_new_tokens_ratio=float(raw["max_new_tokens_ratio"]),
        max_new_tokens_margin=raw["max_new_tokens_margin"],
        max_new_tokens_cap=raw["max_new_tokens_cap"],
        batch_size=raw["batch_size"],
        batch_token_budget=raw.get("batch_token_budget", 0),
    )
    if decoding.do_sample:
        raise SystemExit("decoding.do_sample must be false: the translation must be deterministic")
    if decoding.num_beams < 1 or decoding.batch_size < 1 or decoding.max_input_tokens < 3:
        raise SystemExit("decoding.num_beams, batch_size and max_input_tokens must be positive")
    if decoding.batch_token_budget < 0:
        raise SystemExit("decoding.batch_token_budget must be zero or positive")
    return decoding


def parse_spot_check(raw: Record) -> SpotCheckSpec:
    spot = SpotCheckSpec(
        name=raw["name"],
        output_dir=Path(raw["output_dir"]),
        split=raw["split"],
        size=raw["size"],
        seed=raw["seed"],
        item_id_prefix=raw["item_id_prefix"],
        delimiter=raw["csv_delimiter"],
        encoding=raw["csv_encoding"],
        columns=tuple(raw["columns"]),
        judgment_columns=tuple(raw["judgment_columns"]),
        readme=raw["readme"].lstrip("\n"),
        models=tuple(raw["models"]),
        order_seed=raw["order_seed"],
        display_normalization=tuple(sorted(raw.get("display_normalization", {}).items())),
    )
    if any(
        len(glyph) != 1 or len(shown) != 1 or not shown.isascii()
        for glyph, shown in spot.display_normalization
    ):
        raise SystemExit("spot_check.display_normalization maps one character to one ASCII one")
    if not set(spot.judgment_columns) <= set(spot.columns):
        raise SystemExit("spot_check.judgment_columns must be among spot_check.columns")
    if not spot.models or len(set(spot.models)) != len(spot.models):
        raise SystemExit("spot_check.models must name distinct translation conditions")
    return spot


def parse_models(raw: Record) -> tuple[dict[str, ModelSpec], str]:
    conditions = tuple(raw["conditions"])
    missing = [key for key in conditions if not isinstance(raw.get(key), dict)]
    if not conditions or missing or len(set(conditions)) != len(conditions):
        raise SystemExit(f"models.conditions must name distinct [models.<key>] tables {missing}")
    models = {key: parse_model(key, raw[key]) for key in conditions}
    if len({(spec.name, spec.revision) for spec in models.values()}) != len(models):
        raise SystemExit("models: two conditions name the same model and revision")
    if raw["default"] not in models:
        raise SystemExit(f"models.default must be one of {list(models)}")
    return models, raw["default"]


def check_split_names(default: tuple[str, ...], final_test: tuple[str, ...]) -> None:
    named = [*default, *final_test]
    if not set(named) <= set(SPLIT_NAMES) or len(set(named)) != len(named):
        raise SystemExit("splits.default and splits.final_test must be distinct known split names")
    if "test" in default:
        raise SystemExit("test may only appear in splits.final_test")


def load_config(config_path: Path) -> TranslationConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    nli_config_path = Path(raw["inputs"]["nli_verifier_config"])
    with open(nli_config_path, "rb") as f:
        nli = tomllib.load(f)
    if nli["premise"]["text_source"] != "nli_chunks":
        raise SystemExit(f"{nli_config_path}: premise.text_source must be nli_chunks")
    default = tuple(raw["splits"]["default"])
    final_test = tuple(raw["splits"]["final_test"])
    check_split_names(default, final_test)
    if final_test != tuple(nli["splits"]["final_test"]):
        raise SystemExit("splits.final_test differs from the E3 config")
    if not set(default) <= {*nli["splits"]["fit"], *nli["splits"]["evaluate"]}:
        raise SystemExit("splits.default must be among the E3 fit and evaluate splits")
    probes = nli["label_probes"]
    if len(probes["premises"]) != len(probes["hypotheses"]):
        raise SystemExit("label_probes.premises and hypotheses must have the same length")
    report = raw["report"]
    models, default_model = parse_models(raw["models"])
    spot_check = parse_spot_check(raw["spot_check"])
    if not set(spot_check.models) <= set(models):
        raise SystemExit(f"spot_check.models must be among {list(models)}")
    return TranslationConfig(
        nli_config_path=nli_config_path,
        lds_sha256=nli["dataset"]["lds_sha256"],
        nli_path=Path(nli["dataset"]["nli_path"]),
        nli_sha256=nli["dataset"]["nli_sha256"],
        benchmark_path=Path(nli["benchmark"]["path"]),
        benchmark_report_path=Path(nli["benchmark"]["report_path"]),
        manifest_path=Path(nli["splits"]["manifest_path"]),
        default_splits=default,
        final_test_splits=final_test,
        probe_premises=tuple(probes["premises"]),
        probe_hypotheses=tuple(probes["hypotheses"]),
        models=models,
        default_model=default_model,
        decoding=parse_decoding(raw["decoding"]),
        segmenter=Segmenter(
            join_abbreviations=frozenset(raw["units"]["join_abbreviations"]),
            join_short_parts=bool(raw["units"]["join_short_parts"]),
        ),
        degenerate_ngram=report["degenerate_ngram"],
        degenerate_min_count=report["degenerate_min_count"],
        cache_dir=Path(raw["cache"]["dir"]),
        output_dir=Path(report["output_dir"]),
        spot_check=spot_check,
        seed=raw["run"]["seed"],
        device=raw["run"]["device"],
        hf_hub_offline=bool(raw["run"]["hf_hub_offline"]),
        progress_every=raw["run"]["progress_every_batches"],
        source={"translation": raw, "nli_verifier": {"path": str(nli_config_path)}},
    )


def boundary_parts(chunk: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_PATTERN.split(normalize_whitespace(chunk))
    return [part for part in (normalize_whitespace(p) for p in parts) if part]


def join_segments(translations: Iterable[str]) -> str:
    return " ".join(text for text in (t.strip() for t in translations) if text)
