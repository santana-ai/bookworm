"""Append-only translation cache per model, keyed by the source text and the model signature."""

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from experiments.common.cache_lock import CacheLockedError, acquire_writer_lock
from experiments.common.transcript import (
    normalize_whitespace,
)
from experiments.verifier.runtime import incomplete_tail
from experiments.verifier.translate.config import (
    DecodingSpec,
    ModelSpec,
    Segmenter,
    TranslationConfig,
    join_segments,
)

Record = dict[str, Any]

REVISION_PREFIX = 12


CONTENT_PATTERN = re.compile(r"\w")


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


def cache_path(cache_dir: Path, model: ModelSpec) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model.name).strip("-").lower()
    return cache_dir / f"translations_{slug}_{model.revision[:REVISION_PREFIX]}.jsonl"


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
                f"signature {self.digest[:16]} in {self.path}; run python -m "
                "experiments.verifier.translation translate --model "
                f"{self.condition or '<the condition of this cache>'} for the "
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
