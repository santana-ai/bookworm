"""English translations of UDV texts, read from the translation cache of the verifier run."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bookworm.data.io import JsonObject, required_field, required_text
from bookworm.errors import ConfigError
from bookworm.transcript.sentences import SENTENCE_BOUNDARY_PATTERN, is_sentence
from bookworm.transcript.text import normalize_whitespace

CONTENT_PATTERN = re.compile(r"\w")
TRANSLATION_KEY_SEPARATOR = "\x1e"
TRANSLATION_MODEL_KEYS = ("name", "revision", "license")


def has_content(text: str) -> bool:
    return CONTENT_PATTERN.search(text) is not None


def translation_key(signature_sha256: str, text: str) -> str:
    return hashlib.sha256(
        f"{signature_sha256}{TRANSLATION_KEY_SEPARATOR}{text}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class Segmentation:
    """Sentence segmentation the translator was fed, so a chunk is split the same way."""

    join_abbreviations: frozenset[str]
    join_short_parts: bool

    def needs_join(self, text: str) -> bool:
        if text.split()[-1] in self.join_abbreviations:
            return True
        return self.join_short_parts and not is_sentence(text)

    def parts(self, chunk: str) -> list[str]:
        normalized = (
            normalize_whitespace(raw)
            for raw in SENTENCE_BOUNDARY_PATTERN.split(normalize_whitespace(chunk))
        )
        return [part for part in normalized if part]

    def segments(self, chunk: str) -> list[str]:
        units: list[str] = []
        pending = ""
        for part in self.parts(chunk):
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


@dataclass(frozen=True)
class Translations:
    signature_sha256: str
    segmentation: Segmentation
    entries: Mapping[str, str]
    path: Path

    def lookup(self, text: str) -> str:
        normalized = normalize_whitespace(text)
        if not has_content(normalized):
            return normalized
        found = self.entries.get(translation_key(self.signature_sha256, normalized))
        if found is None:
            raise ConfigError(
                f"{self.path}: no translation of a {len(normalized)}-character text under "
                f"signature {self.signature_sha256[:16]}"
            )
        return found

    def translate_opinion(self, opinion: str) -> str:
        return self.lookup(opinion)

    def translate_chunk(self, chunk: str) -> str:
        translated = (self.lookup(segment).strip() for segment in self.segmentation.segments(chunk))
        return " ".join(text for text in translated if text)


@dataclass(frozen=True)
class TranslationSource:
    path: Path
    signature_sha256: str
    segmentation: Segmentation
    model: JsonObject


def wanted_translations(signature_sha256: str, texts: Iterable[str]) -> dict[str, str]:
    return {
        translation_key(signature_sha256, text): text
        for text in (normalize_whitespace(raw) for raw in texts)
        if has_content(text)
    }


def cached_translations(
    path: Path, signature_sha256: str, wanted: dict[str, str]
) -> dict[str, str]:
    entries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key") if isinstance(row, dict) else None
            if key not in wanted or row.get("signature_sha256") != signature_sha256:
                continue
            if row.get("source") != wanted[key] or not isinstance(row.get("translation"), str):
                raise ConfigError(f"{path}: entry {key[:16]} does not hold the text of its key")
            entries[key] = row["translation"]
    return entries


def load_translations(
    path: Path,
    signature_sha256: str,
    segmentation: Segmentation,
    texts: Iterable[str],
) -> Translations:
    """Read the cached translation of every text; raises ``ConfigError`` if one is missing."""
    wanted = wanted_translations(signature_sha256, texts)
    if not path.is_file():
        raise ConfigError(f"{path}: translation cache not found")
    entries = cached_translations(path, signature_sha256, wanted)
    missing = len(wanted) - len(entries)
    if missing:
        raise ConfigError(
            f"{path}: {missing} of {len(wanted)} texts have no translation under signature "
            f"{signature_sha256[:16]}"
        )
    return Translations(signature_sha256, segmentation, entries, path)


def translation_source(report: JsonObject, origin: str) -> TranslationSource:
    store = required_field(report, ("translation", "store"), origin)
    abbreviations = required_field(store, ("segmentation", "join_abbreviations"), origin)
    join_short = required_field(store, ("segmentation", "join_short_parts"), origin)
    if not isinstance(abbreviations, list) or not isinstance(join_short, bool):
        raise ConfigError(f"{origin}: translation.store.segmentation is malformed")
    model = {
        key: required_text(report, ("translation", "model", key), origin)
        for key in TRANSLATION_MODEL_KEYS
    }
    return TranslationSource(
        path=Path(required_text(store, ("path",), origin)),
        signature_sha256=required_text(store, ("signature_sha256",), origin),
        segmentation=Segmentation(frozenset(str(item) for item in abbreviations), join_short),
        model=model,
    )
