import argparse
import csv
import functools
import hashlib
import json
import math
import os
import platform
import re
import time
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import huggingface_hub
import numpy as np
import tokenizers
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedConfig

from utils import (
    build_nli_benchmark,
    build_udvs,
    cache_lock,
    calibrate_threshold,
    dataset_io,
    hub_offline,
    udv_pipeline,
)
from utils.build_nli_benchmark import iter_opinions
from utils.build_udvs import seed_everything, select_device
from utils.cache_lock import CacheLockedError, acquire_writer_lock
from utils.calibrate_threshold import load_split_lookup
from utils.dataset_io import load_gated_jsonl, load_jsonl, module_path, sha256_of_file, write_json
from utils.hub_offline import enforce_offline, offline_state, pinned_weights_file
from utils.udv_pipeline import (
    SENTENCE_BOUNDARY_PATTERN,
    is_sentence,
    normalize_whitespace,
    split_sentences,
)

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
DEVICES = ("auto", "cpu", "mps", "cuda")
CONTENT_PATTERN = re.compile(r"\w")
WORD_PATTERN = re.compile(r"\w+")
FETCH_PATTERNS = ["*.json", "*.model", "README.md", "pytorch_model.bin", "model.safetensors"]
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
FLAG_NAMES = (
    "empty",
    "identical",
    "degenerate",
    "degenerate_not_in_source",
    "input_truncated",
    "hit_max_new_tokens",
    "long_output",
    "doubled_length",
)
MAX_LISTED_KEYS = 200


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


@dataclass(frozen=True)
class TranslationOutput:
    text: str
    input_tokens: int
    model_input_tokens: int
    output_tokens: int
    input_truncated: bool
    hit_max_new_tokens: bool


class Translator(Protocol):
    signature: Record
    device: str
    batch_size: int

    def model_input_tokens(self, texts: list[str]) -> list[int]: ...

    def translate(self, texts: list[str]) -> list[TranslationOutput]: ...


class MissingTranslationError(KeyError):
    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class OpinionUnit:
    unit_id: str
    hearing_id: int
    split: str
    opinion: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class TextSelection:
    opinions: list[str]
    chunks: list[str]
    segments: list[str]
    verbatim_segments: list[str]

    def model_texts(self) -> list[str]:
        return list(dict.fromkeys([*self.opinions, *self.segments]))


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


def model_signature(model: ModelSpec, decoding: DecodingSpec) -> Record:
    return {
        "model": model.name,
        "revision": model.revision,
        "src_lang": model.src_lang,
        "tgt_lang": model.tgt_lang,
        "dtype": model.dtype,
        **decoding.generation_kwargs(),
        "max_input_tokens": decoding.max_input_tokens,
        "max_new_tokens_ratio": decoding.max_new_tokens_ratio,
        "max_new_tokens_margin": decoding.max_new_tokens_margin,
        "max_new_tokens_cap": decoding.max_new_tokens_cap,
    }


def signature_digest(signature: Record) -> str:
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def text_key(digest: str, text: str) -> str:
    return hashlib.sha256(f"{digest}\x1e{text}".encode()).hexdigest()


def has_content(text: str) -> bool:
    return CONTENT_PATTERN.search(text) is not None


def boundary_parts(chunk: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_PATTERN.split(normalize_whitespace(chunk))
    return [part for part in (normalize_whitespace(p) for p in parts) if part]


def join_segments(translations: Iterable[str]) -> str:
    return " ".join(text for text in (t.strip() for t in translations) if text)


def cache_path(cache_dir: Path, model: ModelSpec) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model.name).strip("-").lower()
    return cache_dir / f"translations_{slug}_{model.revision[:12]}.jsonl"


def incomplete_tail(data: bytes) -> int:
    if not data or data.endswith(b"\n"):
        return 0
    return len(data) - (data.rfind(b"\n") + 1)


@dataclass
class TranslationStore:
    path: Path
    signature: Record
    digest: str
    segmenter: Segmenter
    entries: dict[str, Record] = field(default_factory=dict)
    other_signature_lines: int = 0
    incomplete_tail_bytes: int = 0
    writable: bool = False
    condition: str = ""

    @classmethod
    def open(
        cls,
        path: Path,
        signature: Record,
        segmenter: Segmenter,
        writable: bool = False,
        condition: str = "",
    ) -> "TranslationStore":
        store = cls(
            path,
            signature,
            signature_digest(signature),
            segmenter,
            writable=writable,
            condition=condition,
        )
        if writable:
            try:
                acquire_writer_lock(path)
            except CacheLockedError as error:
                raise SystemExit(str(error)) from error
        if not path.exists():
            return store
        data = path.read_bytes()
        tail = incomplete_tail(data)
        store.incomplete_tail_bytes = tail
        if tail and writable:
            with open(path, "r+b") as f:
                f.truncate(len(data) - tail)
        complete = data[: len(data) - tail].decode()
        for number, line in enumerate(complete.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}: line {number} is not valid JSON ({error})") from error
            if record.get("signature_sha256") != store.digest:
                store.other_signature_lines += 1
                continue
            store.entries[record["key"]] = record
        return store

    def key(self, text: str) -> str:
        return text_key(self.digest, normalize_whitespace(text))

    def record(self, text: str) -> Record | None:
        return self.entries.get(self.key(text))

    def contains(self, text: str) -> bool:
        normalized = normalize_whitespace(text)
        return not has_content(normalized) or self.key(normalized) in self.entries

    def missing(self, texts: Iterable[str]) -> list[str]:
        normalized = dict.fromkeys(normalize_whitespace(text) for text in texts)
        return [text for text in normalized if not self.contains(text)]

    def lookup(self, text: str) -> str:
        normalized = normalize_whitespace(text)
        if not has_content(normalized):
            return normalized
        key = self.key(normalized)
        record = self.entries.get(key)
        if record is None:
            raise MissingTranslationError(
                f"no translation of a {len(normalized)}-character text (key {key[:16]}) under "
                f"signature {self.digest[:16]} in {self.path}; run python -m utils.translation "
                f"translate --model {self.condition or '<the condition of this cache>'} for the "
                "splits that contain it"
            )
        return record["translation"]

    def translate_opinion(self, opinion: str) -> str:
        return self.lookup(opinion)

    def chunk_translations(self, chunk: str) -> list[tuple[str, str]]:
        return [(segment, self.lookup(segment)) for segment in self.segmenter.segments(chunk)]

    def translate_chunk(self, chunk: str) -> str:
        return join_segments(english for _, english in self.chunk_translations(chunk))

    def append(
        self, texts: list[str], outputs: list[TranslationOutput], provenance: Record
    ) -> None:
        if not self.writable:
            raise SystemExit(f"{self.path} was opened read-only")
        lines = []
        for text, output in zip(texts, outputs, strict=True):
            record = {
                "key": self.key(text),
                "signature_sha256": self.digest,
                "source": text,
                "translation": output.text,
                "input_tokens": output.input_tokens,
                "model_input_tokens": output.model_input_tokens,
                "output_tokens": output.output_tokens,
                "input_truncated": output.input_truncated,
                "hit_max_new_tokens": output.hit_max_new_tokens,
                **provenance,
                "seconds": provenance["batch_seconds"] / len(texts),
            }
            lines.append(json.dumps(record, ensure_ascii=False) + "\n")
            self.entries[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write("".join(lines))
            f.flush()
            os.fsync(f.fileno())

    def summary(self) -> Record:
        return {
            "path": str(self.path),
            "signature": self.signature,
            "signature_sha256": self.digest,
            "segmentation": self.segmenter.describe(),
            "entries": len(self.entries),
            "entries_by_device": dict(
                sorted(Counter(r["device"] for r in self.entries.values()).items())
            ),
            "other_signature_lines": self.other_signature_lines,
            "incomplete_tail_bytes": self.incomplete_tail_bytes,
        }


def selected_model(config: TranslationConfig, key: str | None = None) -> ModelSpec:
    chosen = config.default_model if key is None else key
    if chosen not in config.models:
        raise SystemExit(f"unknown translation model {chosen!r}; declared: {list(config.models)}")
    return config.models[chosen]


def open_store(
    config: TranslationConfig,
    model_key: str | None = None,
    writable: bool = False,
    cache_dir: Path | None = None,
) -> TranslationStore:
    model = selected_model(config, model_key)
    path = cache_path(cache_dir if cache_dir is not None else config.cache_dir, model)
    signature = model_signature(model, config.decoding)
    return TranslationStore.open(path, signature, config.segmenter, writable, model.key)


def model_record(config: TranslationConfig, spec: ModelSpec) -> Record:
    return {
        "condition": spec.key,
        "role": spec.role,
        "name": spec.name,
        "revision": spec.revision,
        "license": spec.license,
        "signature_sha256": signature_digest(model_signature(spec, config.decoding)),
    }


def english_probe_pairs(config: TranslationConfig, store: TranslationStore) -> list[Record]:
    return [
        {
            "premise": premise,
            "hypothesis": hypothesis,
            "premise_en": store.translate_chunk(premise),
            "hypothesis_en": store.translate_opinion(hypothesis),
        }
        for premise, hypothesis in zip(config.probe_premises, config.probe_hypotheses, strict=True)
    ]


def print_progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0
    remaining = (total - done) / rate / 60 if rate else float("nan")
    print(f"  {done}/{total} texts, {rate:.2f}/s, about {remaining:.1f} min left", flush=True)


def length_batches(lengths: list[int], batch_size: int, token_budget: int = 0) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    for index in order:
        current = batches[-1] if batches else None
        if (
            current is not None
            and len(current) < batch_size
            and (not token_budget or (len(current) + 1) * lengths[current[0]] <= token_budget)
        ):
            current.append(index)
        else:
            batches.append([index])
    return batches


def translate_missing(
    translator: Translator,
    store: TranslationStore,
    texts: Iterable[str],
    progress_every: int = 0,
    token_budget: int = 0,
) -> Record:
    if signature_digest(translator.signature) != store.digest:
        raise SystemExit("the translator signature differs from the store signature")
    missing = store.missing(texts)
    lengths = translator.model_input_tokens(missing) if missing else []
    size = translator.batch_size
    started = time.perf_counter()
    batches = 0
    done = 0
    for batches, indices in enumerate(length_batches(lengths, size, token_budget), start=1):
        batch = [missing[index] for index in indices]
        batch_started = time.perf_counter()
        outputs = translator.translate(batch)
        provenance = {
            "device": translator.device,
            "batch_size": size,
            "batch_token_budget": token_budget,
            "batch_number": batches,
            "batch_texts": len(batch),
            "batch_seconds": time.perf_counter() - batch_started,
            "created_at": now(),
        }
        store.append(batch, outputs, provenance)
        done += len(batch)
        if progress_every and batches % progress_every == 0:
            print_progress(done, len(missing), started)
    seconds = time.perf_counter() - started
    return {
        "translated": len(missing),
        "batches": batches,
        "seconds": round(seconds, 2),
        "texts_per_second": round(len(missing) / seconds, 4) if seconds and missing else None,
        "model_input_tokens": int(sum(lengths)),
        "model_input_tokens_per_second": (
            round(sum(lengths) / seconds, 2) if seconds and missing else None
        ),
    }


@dataclass
class Seq2SeqTranslator:
    spec: ModelSpec
    decoding: DecodingSpec
    tokenizer: Any
    model: Any
    device: str
    signature: Record
    batch_size: int
    target_token_id: int
    special_tokens: int
    info: Record

    def model_input_tokens(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(texts, truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def output_length(self, sequence: list[int]) -> tuple[int, bool]:
        body = sequence[1:]
        eos = self.tokenizer.eos_token_id
        finished = eos in body
        end = body.index(eos) if finished else len(body)
        forced = 1 if body and body[0] == self.target_token_id else 0
        return end - forced, finished

    def translate(self, texts: list[str]) -> list[TranslationOutput]:
        full = self.model_input_tokens(texts)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.decoding.max_input_tokens,
            padding=True,
            return_tensors="pt",
        )
        longest = int(encoded["attention_mask"].sum(dim=1).max())
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded.to(self.device),
                forced_bos_token_id=self.target_token_id,
                max_new_tokens=self.decoding.max_new_tokens(longest),
                **self.decoding.generation_kwargs(),
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs = []
        for count, sequence, text in zip(full, generated.cpu().tolist(), decoded, strict=True):
            length, finished = self.output_length(sequence)
            outputs.append(
                TranslationOutput(
                    text=normalize_whitespace(text),
                    input_tokens=count - self.special_tokens,
                    model_input_tokens=count,
                    output_tokens=length,
                    input_truncated=count > self.decoding.max_input_tokens,
                    hit_max_new_tokens=not finished,
                )
            )
        return outputs


def load_tokenizer(spec: ModelSpec) -> Any:
    return AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, src_lang=spec.src_lang, local_files_only=True
    )


def check_tokenizer(tokenizer: Any, spec: ModelSpec) -> Record:
    source_id = tokenizer.convert_tokens_to_ids(spec.src_token)
    target_id = tokenizer.convert_tokens_to_ids(spec.tgt_token)
    if tokenizer.unk_token_id in (source_id, target_id):
        raise SystemExit(f"{spec.name}: {spec.src_token} or {spec.tgt_token} is unknown")
    probe = tokenizer("Bom dia a todos.")["input_ids"]
    if probe[0] != source_id or probe[-1] != tokenizer.eos_token_id:
        raise SystemExit(f"{spec.name}: inputs do not start with {spec.src_token} and end in </s>")
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "source_token_id": source_id,
        "target_token_id": target_id,
        "special_tokens_per_input": tokenizer.num_special_tokens_to_add(),
        "input_layout": "[src_token] tokens [</s>], checked on a probe sentence",
    }


def resolved_commit(spec: ModelSpec) -> str | None:
    config_dict, _ = PreTrainedConfig.get_config_dict(
        spec.name, revision=spec.revision, local_files_only=True
    )
    return config_dict.get("_commit_hash")


def load_translator(spec: ModelSpec, decoding: DecodingSpec, device: str) -> Seq2SeqTranslator:
    started = time.perf_counter()
    commit = resolved_commit(spec)
    if commit != spec.revision:
        raise SystemExit(f"{spec.name}: resolved {commit} != {spec.revision}")
    weights = pinned_weights_file(spec.name, spec.revision)
    if not weights["snapshot_is_pinned_revision"]:
        raise SystemExit(
            f"{spec.name}: weights file {weights['path']} is not in the pinned snapshot"
        )
    tokenizer = load_tokenizer(spec)
    tokenizer_info = check_tokenizer(tokenizer, spec)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        spec.name,
        revision=spec.revision,
        dtype=getattr(torch, spec.dtype),
        local_files_only=True,
        use_safetensors=weights["format"] == "safetensors",
    )
    model.to(device).eval()
    info = {
        "role": spec.role,
        "name": spec.name,
        "revision": spec.revision,
        "resolved_commit": commit,
        "license": spec.license,
        "weights_file": weights,
        "tokenizer": tokenizer_info,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "torch_threads": torch.get_num_threads(),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    return Seq2SeqTranslator(
        spec=spec,
        decoding=decoding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        signature=model_signature(spec, decoding),
        batch_size=decoding.batch_size,
        target_token_id=tokenizer_info["target_token_id"],
        special_tokens=tokenizer_info["special_tokens_per_input"],
        info=info,
    )


def resolve_splits(
    config: TranslationConfig, requested: list[str] | None, final_test: bool
) -> tuple[str, ...]:
    allowed = (*config.default_splits, *(config.final_test_splits if final_test else ()))
    chosen = tuple(dict.fromkeys(requested)) if requested else allowed
    unknown = [split for split in chosen if split not in SPLIT_NAMES]
    if unknown:
        raise SystemExit(f"unknown splits {unknown}; known: {list(SPLIT_NAMES)}")
    refused = [split for split in chosen if split in config.final_test_splits and not final_test]
    if refused:
        raise SystemExit(f"splits {refused} are refused unless --final-test is passed")
    outside = [split for split in chosen if split not in allowed]
    if outside:
        raise SystemExit(f"splits {outside} are outside {list(allowed)}")
    return chosen


def split_records(splits: tuple[str, ...], final_test: bool) -> Record:
    return {"splits": list(splits), "final_test_flag": final_test, "test_read": "test" in splits}


def check_benchmark_file(config: TranslationConfig) -> Record:
    with open(config.benchmark_report_path) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    actual = sha256_of_file(config.benchmark_path)
    if actual != recorded:
        raise SystemExit(f"{config.benchmark_path}: sha256 {actual} != its report {recorded}")
    return {"path": str(config.benchmark_path), "sha256": actual}


def benchmark_rows(
    config: TranslationConfig, split_of: dict[int, str], splits: tuple[str, ...], limit: int | None
) -> list[Record]:
    rows = []
    taken: Counter[str] = Counter()
    for row in load_jsonl(config.benchmark_path):
        manifest_split = split_of.get(row["hearing_id"])
        if manifest_split != row["split"]:
            raise SystemExit(f"{row['id']}: split {row['split']!r} != manifest {manifest_split!r}")
        if row["split"] not in splits:
            continue
        if limit is not None and taken[row["split"]] >= limit:
            continue
        taken[row["split"]] += 1
        rows.append(row)
    return rows


def nli_chunks(config: TranslationConfig, rows: list[Record]) -> dict[str, list[str]]:
    wanted = {row["id"]: row for row in rows}
    chunks: dict[str, list[str]] = {}
    for hearing, person_index, _, opinion_index, opinion in iter_opinions(
        load_gated_jsonl(config.nli_path, config.nli_sha256)
    ):
        row_id = f"nli-{hearing['id']}-{person_index}-{opinion_index}"
        if row_id not in wanted:
            continue
        if opinion["opiniao"] != wanted[row_id]["opinion"]:
            raise SystemExit(f"{row_id}: the NLI opinion differs from the benchmark row")
        chunks[row_id] = list(opinion["chunks_proximos"])
    missing = sorted(set(wanted) - set(chunks))
    if missing:
        raise SystemExit(f"benchmark rows missing from the NLI file: {missing[:10]}")
    return chunks


def load_units(
    config: TranslationConfig, splits: tuple[str, ...], limit: int | None
) -> tuple[list[OpinionUnit], Record]:
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    benchmark = check_benchmark_file(config)
    rows = benchmark_rows(config, split_of, splits, limit)
    chunks = nli_chunks(config, rows)
    units = [
        OpinionUnit(
            unit_id=row["id"],
            hearing_id=row["hearing_id"],
            split=row["split"],
            opinion=normalize_whitespace(row["opinion"]),
            chunks=tuple(normalize_whitespace(text) for text in chunks[row["id"]]),
        )
        for row in rows
    ]
    context = {
        "sources": {
            "nli": {"path": str(config.nli_path), "sha256": config.nli_sha256},
            "benchmark": benchmark,
            "splits": split_source,
            "nli_verifier_config": {
                "path": str(config.nli_config_path),
                "sha256": sha256_of_file(config.nli_config_path),
            },
        },
        "subset": (
            None
            if limit is None
            else {"limit_per_split": limit, "rule": "first rows of each split in file order"}
        ),
    }
    return units, context


def select_texts(
    opinions: Iterable[str], chunks: Iterable[str], segmenter: Segmenter
) -> TextSelection:
    distinct_chunks = list(dict.fromkeys(chunk for chunk in chunks if chunk))
    segments = list(
        dict.fromkeys(s for chunk in distinct_chunks for s in segmenter.segments(chunk))
    )
    return TextSelection(
        opinions=list(dict.fromkeys(o for o in opinions if has_content(o))),
        chunks=distinct_chunks,
        segments=[segment for segment in segments if has_content(segment)],
        verbatim_segments=[segment for segment in segments if not has_content(segment)],
    )


def unit_selection(units: list[OpinionUnit], config: TranslationConfig) -> TextSelection:
    return select_texts(
        [*(unit.opinion for unit in units), *map(normalize_whitespace, config.probe_hypotheses)],
        [
            *(chunk for unit in units for chunk in unit.chunks),
            *map(normalize_whitespace, config.probe_premises),
        ],
        config.segmenter,
    )


def quantiles(values: Iterable[float], digits: int = 4) -> Record:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), digits),
        **{f"q{int(q * 100):02d}": round(float(np.quantile(array, q)), digits) for q in QUANTILES},
    }


def repeated_ngram(text: str, n: int, min_count: int) -> bool:
    words = WORD_PATTERN.findall(text.lower())
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(count >= min_count for count in grams.values())


def output_flags(source: str, record: Record, config: TranslationConfig) -> Record:
    translation = record["translation"]
    n, count = config.degenerate_ngram, config.degenerate_min_count
    degenerate = repeated_ngram(translation, n, count)
    decoding = config.decoding
    budget = decoding.max_new_tokens_ratio * record["input_tokens"] + decoding.max_new_tokens_margin
    return {
        "empty": not translation,
        "identical": translation == source,
        "degenerate": degenerate,
        "degenerate_not_in_source": degenerate and not repeated_ngram(source, n, count),
        "input_truncated": bool(record["input_truncated"]),
        "hit_max_new_tokens": bool(record["hit_max_new_tokens"]),
        "long_output": record["output_tokens"] > budget,
        "doubled_length": record["output_tokens"] > 2 * record["input_tokens"],
    }


def segmentation_check(chunks: list[str]) -> Record:
    dropped_words = total_words = chunks_with_drop = fallback = 0
    coverage = []
    for chunk in chunks:
        sentences = split_sentences(chunk)
        if not sentences:
            fallback += 1
            sentences = [chunk]
        words = len(chunk.split())
        kept = min(sum(len(sentence.split()) for sentence in sentences), words)
        dropped_words += words - kept
        total_words += words
        chunks_with_drop += kept < words
        coverage.append(kept / words)
    return {
        "method": (
            "for each distinct non-empty chunk, the words of the sentences split_sentences returns "
            "(the whole chunk when it returns none) against the words of the chunk; this is what "
            "joining only split_sentences sentences would drop, while the translation units keep "
            "every word"
        ),
        "distinct_chunks": len(chunks),
        "chunks_without_sentence": fallback,
        "chunks_losing_words": chunks_with_drop,
        "words": total_words,
        "words_dropped": dropped_words,
        "words_dropped_fraction": round(dropped_words / total_words, 6) if total_words else None,
        "chunk_word_coverage": quantiles(coverage),
        "distinct_split_sentences": len(
            {s for chunk in chunks for s in (split_sentences(chunk) or [chunk])}
        ),
    }


def selection_counts(
    units: list[OpinionUnit], selection: TextSelection, segmenter: Segmenter
) -> Record:
    segments_per_chunk = [len(segmenter.segments(chunk)) for chunk in selection.chunks]
    parts_per_chunk = [boundary_parts(chunk) for chunk in selection.chunks]
    distinct_parts = list(dict.fromkeys(part for parts in parts_per_chunk for part in parts))
    sentences = [segment for segment in selection.segments if is_sentence(segment)]
    chunk_slots = [chunk for unit in units for chunk in unit.chunks]
    return {
        "opinions": len(units),
        "opinions_by_split": dict(sorted(Counter(unit.split for unit in units).items())),
        "hearings": len({unit.hearing_id for unit in units}),
        "chunk_slots": len(chunk_slots),
        "empty_chunk_slots": sum(1 for chunk in chunk_slots if not chunk),
        "distinct_opinions_with_probes": len(selection.opinions),
        "distinct_chunks_with_probes": len(selection.chunks),
        "distinct_segments": len(selection.segments),
        "distinct_segments_accepted_by_is_sentence": len(sentences),
        "distinct_segments_shorter_than_a_sentence": len(selection.segments) - len(sentences),
        "distinct_verbatim_segments": len(selection.verbatim_segments),
        "distinct_boundary_parts": len(distinct_parts),
        "distinct_boundary_parts_ending_in_join_abbreviation": sum(
            1 for part in distinct_parts if part.split()[-1] in segmenter.join_abbreviations
        ),
        "distinct_boundary_parts_rejected_by_is_sentence": sum(
            1 for part in distinct_parts if not is_sentence(part)
        ),
        "join_abbreviation_last_token_counts": dict(
            sorted(
                Counter(
                    part.split()[-1]
                    for parts in parts_per_chunk
                    for part in parts[:-1]
                    if part.split()[-1] in segmenter.join_abbreviations
                ).items()
            )
        ),
        "chunks_with_joined_parts": sum(
            1
            for parts, count in zip(parts_per_chunk, segments_per_chunk, strict=True)
            if len(parts) != count
        ),
        "segments_per_chunk": quantiles(segments_per_chunk),
        "distinct_model_texts": len(selection.model_texts()),
    }


def token_plan(tokenizer: Any, texts: list[str], max_input: int, special: int) -> Record:
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]] if texts else []
    return {
        "texts": len(texts),
        "model_input_tokens": int(sum(lengths)),
        "model_input_tokens_distribution": quantiles(lengths, 1),
        "content_tokens_distribution": quantiles([length - special for length in lengths], 1),
        "over_max_input_tokens": sum(1 for length in lengths if length > max_input),
    }


def record_statistics(
    config: TranslationConfig, texts: list[str], store: TranslationStore
) -> Record:
    records = [(text, store.record(text)) for text in texts]
    present = [(text, record) for text, record in records if record is not None]
    flags: dict[str, list[str]] = {name: [] for name in FLAG_NAMES}
    for text, record in present:
        for name, value in output_flags(text, record, config).items():
            if value:
                flags[name].append(record["key"])
    return {
        "texts": len(texts),
        "translated": len(present),
        "missing": len(texts) - len(present),
        "flags": {name: len(keys) for name, keys in flags.items()},
        "flagged_keys": {name: keys[:MAX_LISTED_KEYS] for name, keys in flags.items() if keys},
        "input_tokens": quantiles((r["input_tokens"] for _, r in present), 1),
        "output_tokens": quantiles((r["output_tokens"] for _, r in present), 1),
        "token_ratio": quantiles(
            r["output_tokens"] / r["input_tokens"] for _, r in present if r["input_tokens"]
        ),
        "character_ratio": quantiles(len(r["translation"]) / len(t) for t, r in present),
        "devices": dict(sorted(Counter(r["device"] for _, r in present).items())),
    }


def chunk_statistics(chunks: list[str], store: TranslationStore) -> Record:
    ratios = []
    seconds = []
    incomplete = 0
    for chunk in chunks:
        segments = [s for s in dict.fromkeys(store.segmenter.segments(chunk)) if has_content(s)]
        records = [store.record(segment) for segment in segments]
        if any(record is None for record in records):
            incomplete += 1
            continue
        ratios.append(len(store.translate_chunk(chunk)) / len(chunk))
        seconds.append(sum(record["seconds"] for record in records if record is not None))
    return {
        "distinct_chunks": len(chunks),
        "chunks_with_missing_segments": incomplete,
        "character_ratio": quantiles(ratios),
        "seconds_per_chunk": quantiles(seconds),
        "seconds_per_chunk_rule": (
            "sum over the distinct parts of a chunk of the per-text seconds (batch seconds / "
            "batch texts) of their cache records; a part shared by chunks counts in each"
        ),
    }


def timing_by_device(records: list[Record]) -> Record:
    timing: Record = {}
    for device in sorted({record["device"] for record in records}):
        members = [record for record in records if record["device"] == device]
        seconds = sum(record["seconds"] for record in members)
        tokens = sum(record["model_input_tokens"] for record in members)
        timing[device] = {
            "records": len(members),
            "seconds": round(seconds, 2),
            "model_input_tokens": tokens,
            "seconds_per_text": round(seconds / len(members), 4),
            "seconds_per_model_input_token": round(seconds / tokens, 6) if tokens else None,
        }
    return timing


def cache_timing(selection: TextSelection, store: TranslationStore) -> Record:
    def present(texts: list[str]) -> list[Record]:
        return [record for text in texts if (record := store.record(text)) is not None]

    return {
        "rule": (
            "per-text seconds are batch seconds / batch texts, summed over the cache records of "
            "the selected texts; a text translated by an earlier run counts with its own timing"
        ),
        "all_texts": timing_by_device(present(selection.model_texts())),
        "opinions": timing_by_device(present(selection.opinions)),
        "segments": timing_by_device(present(selection.segments)),
    }


def estimate_from_plan(plan_path: Path, timing: Record) -> Record:
    with open(plan_path) as f:
        plan = json.load(f)
    tokens = plan["tokens"]["all"]["model_input_tokens"]
    texts = plan["counts"]["distinct_model_texts"]
    estimates = {}
    for device, measured in timing["all_texts"].items():
        per_token = measured["seconds_per_model_input_token"]
        estimates[device] = {
            "from_tokens_hours": round(per_token * tokens / 3600, 2) if per_token else None,
            "from_texts_hours": round(measured["seconds_per_text"] * texts / 3600, 2),
            "measured_records": measured["records"],
        }
    return {
        "plan": {"path": str(plan_path), "sha256": sha256_of_file(plan_path)},
        "plan_splits": plan["splits"],
        "plan_distinct_model_texts": texts,
        "plan_model_input_tokens": tokens,
        "by_device": estimates,
        "devices_without_measurement": [d for d in ("cpu", "mps") if d not in estimates],
        "note": (
            "an extrapolation from the measured records, not a measurement; the time of a device "
            "without records is unknown until measured"
        ),
    }


@functools.cache
def code_hashes() -> Record:
    modules = (
        udv_pipeline,
        build_udvs,
        build_nli_benchmark,
        cache_lock,
        calibrate_threshold,
        dataset_io,
        hub_offline,
    )
    hashes = {f"utils/{module_path(m).name}": sha256_of_file(module_path(m)) for m in modules}
    return {"utils/translation.py": sha256_of_file(Path(__file__)), **hashes}


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "load_average": [round(value, 2) for value in os.getloadavg()],
        "platform": platform.platform(),
        "hub_offline": offline_state(),
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_directory(config: TranslationConfig, args: argparse.Namespace) -> Path:
    base = args.output_dir if args.output_dir is not None else config.output_dir
    return base / args.run_name


def texts_digest(texts: list[str]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def build_report(
    args: argparse.Namespace,
    config: TranslationConfig,
    spec: ModelSpec,
    units: list[OpinionUnit],
    context: Record,
    store: TranslationStore,
    run: Record,
    splits: tuple[str, ...],
) -> Record:
    selection = unit_selection(units, config)
    probe_chunks = {normalize_whitespace(p) for p in config.probe_premises}
    probes_ready = not store.missing(
        [*config.probe_hypotheses, *(s for p in probe_chunks for s in store.segmenter.segments(p))]
    )
    timing = cache_timing(selection, store)
    report = {
        "experiment": "translation",
        "command": args.command,
        "run_name": args.run_name,
        "created_at": now(),
        "splits": split_records(splits, args.final_test),
        "model": model_record(config, spec),
        "segmentation": config.segmenter.describe(),
        "counts": selection_counts(units, selection, config.segmenter),
        "coverage": {
            "model_texts_missing": len(store.missing(selection.model_texts())),
            "complete": not store.missing(selection.model_texts()),
        },
        "opinions": record_statistics(config, selection.opinions, store),
        "segments": record_statistics(config, selection.segments, store),
        "chunks": chunk_statistics(
            [chunk for chunk in selection.chunks if chunk not in probe_chunks], store
        ),
        "timing": {"this_run": run, "cache": timing},
        "label_probes": english_probe_pairs(config, store) if probes_ready else None,
        "store": store.summary(),
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    if getattr(args, "estimate_from_plan", None) is not None:
        report["estimate"] = estimate_from_plan(args.estimate_from_plan, timing)
    return report


def print_report(report: Record) -> None:
    for kind in ("opinions", "segments"):
        part = report[kind]
        print(
            f"{kind}: {part['translated']}/{part['texts']} translated, flags {part['flags']}, "
            f"token ratio median {part['token_ratio'].get('q50')}",
            flush=True,
        )
    print(f"timing {json.dumps(report['timing'])}", flush=True)
    if "estimate" in report:
        print(f"estimate {json.dumps(report['estimate']['by_device'])}", flush=True)


def command_fetch(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    path = snapshot_download(spec.name, revision=spec.revision, allow_patterns=FETCH_PATTERNS)
    print(f"{spec.key}: {spec.name}@{spec.revision} -> {path}", flush=True)


def command_plan(args: argparse.Namespace, config: TranslationConfig) -> None:
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    selection = unit_selection(units, config)
    spec = selected_model(config, args.model)
    tokenizer = load_tokenizer(spec)
    tokenizer_info = check_tokenizer(tokenizer, spec)
    special = tokenizer_info["special_tokens_per_input"]
    limit = config.decoding.max_input_tokens
    store = open_store(config, spec.key, cache_dir=args.cache_dir)
    probe_chunks = {normalize_whitespace(p) for p in config.probe_premises}
    report = {
        "experiment": "translation",
        "command": "plan",
        "run_name": args.run_name,
        "created_at": now(),
        "splits": split_records(splits, args.final_test),
        "model": {**model_record(config, spec), "tokenizer": tokenizer_info},
        "segmentation": config.segmenter.describe(),
        "counts": selection_counts(units, selection, config.segmenter),
        "model_texts": {
            "distinct": len(selection.model_texts()),
            "sha256": texts_digest(selection.model_texts()),
            "rule": (
                "sha256 of the JSON list of the distinct model texts (opinions, then chunk units, "
                "each once, in first-occurrence order); it depends on the units only, not on the "
                "model, so two conditions with the same sha256 translate the same texts"
            ),
        },
        "tokens": {
            "opinions": token_plan(tokenizer, selection.opinions, limit, special),
            "segments": token_plan(tokenizer, selection.segments, limit, special),
            "all": token_plan(tokenizer, selection.model_texts(), limit, special),
        },
        "segmentation_check": segmentation_check(
            [chunk for chunk in selection.chunks if chunk not in probe_chunks]
        ),
        "cache": {
            **store.summary(),
            "model_texts_missing": len(store.missing(selection.model_texts())),
        },
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    path = run_directory(config, args) / "plan.json"
    write_json(report, path)
    print(json.dumps({key: report[key] for key in ("counts", "tokens")}, indent=2))
    print(json.dumps(report["segmentation_check"], indent=2))
    print(f"-> {path}", flush=True)


def translate_selection(
    args: argparse.Namespace,
    config: TranslationConfig,
    store: TranslationStore,
    spec: ModelSpec,
    texts: list[str],
) -> tuple[Record, Record | None]:
    missing = store.missing(texts)
    run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
    if not missing:
        return {**run, "translated": 0, "model_loaded": False}, None
    if store.digest != signature_digest(model_signature(spec, config.decoding)):
        raise SystemExit(f"{store.path} was opened for another model than {spec.key}")
    device = select_device(args.device if args.device is not None else config.device)
    translator = load_translator(spec, config.decoding, device)
    print(f"{spec.name} on {device}: {len(missing)} texts to translate", flush=True)
    result = translate_missing(
        translator, store, missing, config.progress_every, config.decoding.batch_token_budget
    )
    return {**run, **result, "model_loaded": True}, translator.info


def command_translate(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    selection = unit_selection(units, config)
    store = open_store(config, spec.key, writable=True, cache_dir=args.cache_dir)
    run, model_info = translate_selection(args, config, store, spec, selection.model_texts())
    report = build_report(args, config, spec, units, context, store, run, splits)
    report["model_info"] = model_info
    path = run_directory(config, args) / "report.json"
    write_json(report, path)
    print_report(report)
    print(f"-> {path}", flush=True)


def command_report(args: argparse.Namespace, config: TranslationConfig) -> None:
    spec = selected_model(config, args.model)
    splits = resolve_splits(config, args.splits, args.final_test)
    units, context = load_units(config, splits, args.limit)
    store = open_store(config, spec.key, cache_dir=args.cache_dir)
    report = build_report(
        args, config, spec, units, context, store, {"model_loaded": False}, splits
    )
    path = run_directory(config, args) / "report.json"
    write_json(report, path)
    print_report(report)
    print(f"-> {path}", flush=True)


def spot_check_population(
    units: list[OpinionUnit], segmenter: Segmenter
) -> tuple[list[str], dict[str, list[str]]]:
    occurrences: dict[str, list[str]] = {}
    for unit in units:
        for chunk in unit.chunks:
            for segment in segmenter.segments(chunk):
                if is_sentence(segment) and has_content(segment):
                    ids = occurrences.setdefault(segment, [])
                    if unit.unit_id not in ids:
                        ids.append(unit.unit_id)
    return list(occurrences), occurrences


def existing_judgments(path: Path, spot: SpotCheckSpec) -> int:
    if not path.exists():
        return 0
    with open(path, encoding=spot.encoding, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=spot.delimiter))
    return sum(
        1 for row in rows for column in spot.judgment_columns if (row.get(column) or "").strip()
    )


def write_spot_check_csv(path: Path, spot: SpotCheckSpec, rows: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=spot.encoding, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(spot.columns), delimiter=spot.delimiter, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in spot.columns})


def spot_check_draws(
    units: list[OpinionUnit], config: TranslationConfig
) -> tuple[list[int], list[str], int, dict[str, list[str]]]:
    spot = config.spot_check
    population, occurrences = spot_check_population(units, config.segmenter)
    if len(population) < spot.size:
        raise SystemExit(f"population of {len(population)} sentences is smaller than {spot.size}")
    rng = np.random.default_rng(spot.seed)
    draws = [int(index) for index in rng.choice(len(population), size=spot.size, replace=False)]
    return draws, [population[index] for index in draws], len(population), occurrences


def spot_check_order(rows: int, seed: int) -> list[int]:
    return [int(index) for index in np.random.default_rng(seed).permutation(rows)]


def blinding_check(sheet: list[Record], items: list[Record]) -> Record:
    model_of = {item["item_id"]: item["model"] for item in items}
    unit_of = {item["item_id"]: item["unit"] for item in items}
    rows: dict[str, dict[str, set[str]]] = {}
    for row in sheet:
        for character in row["traducao_en"]:
            if not character.isalnum() and not character.isspace():
                by_model = rows.setdefault(character, {})
                by_model.setdefault(model_of[row["item_id"]], set()).add(row["item_id"])
    single = {
        f"U+{ord(character):04X}": {model: sorted(ids) for model, ids in by_model.items()}
        for character, by_model in sorted(rows.items())
        if len(by_model) == 1
    }
    identifiable = sorted(
        {item_id for by_model in single.values() for ids in by_model.values() for item_id in ids}
    )
    units = sorted({unit_of[item_id] for item_id in identifiable})
    return {
        "rule": (
            "every non-alphanumeric, non-space character of the shown translations that occurs in "
            "the rows of one model only; such a row can be told apart by that character, and so "
            "can the other row of its unit, which shows the same texto_pt"
        ),
        "characters_of_one_model": single,
        "identifiable_items": identifiable,
        "identifiable_units": units,
        "items_of_identifiable_units": sorted(i for i, u in unit_of.items() if u in units),
    }


def spot_check_stores(
    args: argparse.Namespace, config: TranslationConfig, sources: list[str]
) -> tuple[dict[str, TranslationStore], dict[str, Record], dict[str, Record | None]]:
    spot = config.spot_check
    writer = None if args.cached_only else selected_model(config, args.model).key
    if writer is not None and writer not in spot.models:
        raise SystemExit(f"--model {writer} is not among spot_check.models {list(spot.models)}")
    stores = {key: open_store(config, key, cache_dir=args.cache_dir) for key in spot.models}
    for key, store in stores.items():
        missing = store.missing(sources)
        if missing and key != writer:
            raise SystemExit(
                f"{len(missing)} of {len(sources)} spot-check units have no {key} translation in "
                f"{store.path} (signature {store.digest[:16]}); spot-check translates only the "
                f"model of --model and opens every other cache read-only, so nothing was written; "
                f"run spot-check --model {key} for it first"
            )
    runs: dict[str, Record] = {}
    infos: dict[str, Record | None] = {}
    for key in spot.models:
        store = stores[key]
        missing = store.missing(sources)
        if key == writer and missing:
            stores[key] = open_store(config, key, writable=True, cache_dir=args.cache_dir)
            runs[key], infos[key] = translate_selection(
                args, config, stores[key], config.models[key], sources
            )
            continue
        runs[key] = {
            "requested": len(sources),
            "already_cached": len(sources) - len(missing),
            "translated": 0,
            "model_loaded": False,
            "store_opened": "read-only",
        }
        infos[key] = None
    return stores, runs, infos


def command_spot_check(args: argparse.Namespace, config: TranslationConfig) -> None:
    spot = config.spot_check
    splits = resolve_splits(config, [spot.split], False)
    units, context = load_units(config, splits, None)
    draws, sources, population_size, occurrences = spot_check_draws(units, config)
    out_dir = spot.output_dir / spot.name
    csv_path = out_dir / "spot_check.csv"
    filled = existing_judgments(csv_path, spot)
    if filled:
        raise SystemExit(f"{csv_path} has {filled} judgment cells filled in; move it away first")
    replaced = {
        "path": str(csv_path),
        "existed": csv_path.exists(),
        "sha256": sha256_of_file(csv_path) if csv_path.exists() else None,
        "filled_judgment_cells": filled,
        "judgment_columns": list(spot.judgment_columns),
    }
    stores, runs, infos = spot_check_stores(args, config, sources)
    built = [(unit, key) for unit in range(len(sources)) for key in spot.models]
    order = spot_check_order(len(built), spot.order_seed)
    width = len(str(len(built)))
    unit_width = len(str(len(sources)))
    sheet, items = [], []
    item_of: dict[tuple[int, str], str] = {}
    for number, index in enumerate(order, start=1):
        unit, key = built[index]
        source = sources[unit]
        record = stores[key].record(source)
        if record is None:
            raise SystemExit(f"unit {unit + 1} has no {key} translation after translate")
        item_id = f"{spot.item_id_prefix}{number:0{width}d}"
        item_of[(unit, key)] = item_id
        shown = spot.display(record["translation"])
        sheet.append({"item_id": item_id, "texto_pt": source, "traducao_en": shown})
        items.append(
            {
                "item_id": item_id,
                "unit": f"U{unit + 1:0{unit_width}d}",
                "built_row": index,
                "model": key,
                "model_name": config.models[key].name,
                "model_revision": config.models[key].revision,
                "signature_sha256": stores[key].digest,
                "population_index": draws[unit],
                "cache_key": record["key"],
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "translation_sha256": hashlib.sha256(record["translation"].encode()).hexdigest(),
                "display_normalized": shown != record["translation"],
                "sheet_translation_sha256": hashlib.sha256(shown.encode()).hexdigest(),
                "opinion_ids": occurrences[source],
                "hearing_ids": sorted({int(i.split("-")[1]) for i in occurrences[source]}),
                "input_tokens": record["input_tokens"],
                "output_tokens": record["output_tokens"],
                "device": record["device"],
                "flags": output_flags(source, record, config),
            }
        )
    distinct = [
        {stores[key].entries[stores[key].key(source)]["translation"] for key in spot.models}
        for source in sources
    ]
    unit_records = [
        {
            "unit": f"U{unit + 1:0{unit_width}d}",
            "population_index": draws[unit],
            "source_sha256": hashlib.sha256(sources[unit].encode()).hexdigest(),
            "items": {key: item_of[(unit, key)] for key in spot.models},
            "translations_identical": len(distinct[unit]) == 1,
        }
        for unit in range(len(sources))
    ]
    write_spot_check_csv(csv_path, spot, sheet)
    (out_dir / "README.md").write_text(spot.readme)
    raw = config.source["translation"]["spot_check"]
    key_report = {
        "experiment": "translation_spot_check",
        "name": spot.name,
        "created_at": now(),
        "supersedes": raw.get("supersedes"),
        "splits": split_records(splits, False),
        "population": {"size": population_size, "rule": raw["population"]},
        "draw": {"seed": spot.seed, "size": spot.size, "rule": raw["draw_rule"], "indices": draws},
        "order": {"seed": spot.order_seed, "rule": raw["order_rule"], "permutation": order},
        "display": {
            "normalization": {
                f"U+{ord(glyph):04X}": shown for glyph, shown in spot.display_normalization
            },
            "rule": raw.get("display_rule"),
            "rows_changed": sum(1 for item in items if item["display_normalized"]),
        },
        "blinding_check": blinding_check(sheet, items),
        "replaced_sheet": replaced,
        "models": {key: model_record(config, config.models[key]) for key in spot.models},
        "counts": {
            "units": len(sources),
            "rows": len(built),
            "units_with_identical_translations": sum(
                1 for record in unit_records if record["translations_identical"]
            ),
            "flags_by_model": {
                key: {
                    name: sum(1 for item in items if item["model"] == key and item["flags"][name])
                    for name in FLAG_NAMES
                }
                for key in spot.models
            },
        },
        "units": unit_records,
        "items": items,
        "sheet": {"path": str(csv_path), "sha256": sha256_of_file(csv_path)},
        "runs": runs,
        "model_info": infos,
        "stores": {key: store.summary() for key, store in stores.items()},
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    write_json(key_report, out_dir / "spot_check_key.json")
    print(
        f"{len(sources)} of {population_size} units x {len(spot.models)} models = {len(built)} "
        f"rows -> {csv_path}",
        flush=True,
    )


def add_selection_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--run-name", required=True)
    command.add_argument("--output-dir", type=Path, default=None)
    command.add_argument("--splits", nargs="+", default=None)
    command.add_argument("--limit", type=int, default=None, help="first N opinions per split")
    command.add_argument(
        "--final-test",
        action="store_true",
        help="also read the final_test splits; never used for choices",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portuguese to English machine translation of the E3 premises and "
        "hypotheses (benchmark opinions and chunk parts), with a resumable cache."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/translation.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "plan", "translate", "report", "spot-check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--model",
            default=None,
            help="translation condition of [models] (default: models.default, nllb)",
        )
        if name == "fetch":
            continue
        command.add_argument("--cache-dir", type=Path, default=None)
        if name in ("plan", "translate", "report"):
            add_selection_arguments(command)
        if name in ("translate", "spot-check"):
            command.add_argument("--device", choices=DEVICES, default=None)
        if name in ("translate", "report"):
            command.add_argument("--estimate-from-plan", type=Path, default=None)
        if name == "spot-check":
            command.add_argument("--cached-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    code_hashes()
    args = parse_args()
    config = load_config(args.config)
    selected_model(config, args.model)
    if args.command != "fetch" and config.hf_hub_offline:
        enforce_offline()
    seed_everything(config.seed)
    transformers.logging.set_verbosity_error()
    commands = {
        "fetch": command_fetch,
        "plan": command_plan,
        "translate": command_translate,
        "report": command_report,
        "spot-check": command_spot_check,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
