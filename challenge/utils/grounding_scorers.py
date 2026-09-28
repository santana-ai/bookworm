import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.stats import rankdata

Record = dict[str, Any]
Pair = tuple[str, str]

KINDS = (
    "minicheck_t5",
    "minicheck_encoder",
    "factcg",
    "alignscore",
    "hhem",
    "reranker",
    "llm_judge",
    "granite_guardian",
)
PROBABILITY_KINDS = tuple(kind for kind in KINDS if kind != "reranker")
MINICHECK_LABEL_IDS = (3, 209)
ALIGNSCORE_IGNORED_KEYS = ("embeddings.position_ids",)
HHEM_PROMPT = (
    "<pad> Determine if the hypothesis is true given the premise?\n\n"
    "Premise: {text1}\n\nHypothesis: {text2}"
)
HHEM_CARD_PAIRS = (
    ("The capital of France is Berlin.", "The capital of France is Paris."),
    ("I am in California", "I am in United States."),
    ("I am in United States", "I am in California."),
    (
        "A person on a horse jumps over a broken down airplane.",
        "A person is outdoors, on a horse.",
    ),
    (
        "A boy is jumping on skateboard in the middle of a red bridge.",
        "The boy skates down the sidewalk on a red bridge",
    ),
    (
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "A blond man wearing a brown shirt is reading a book.",
    ),
    ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."),
)
HHEM_CARD_SCORES = (
    0.011061512865126133,
    0.6473632454872131,
    0.1290171593427658,
    0.8969419002532959,
    0.18462494015693665,
    0.005031010136008263,
    0.05432349815964699,
)
HHEM_CARD_TOLERANCE = 0.005


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    kind: str
    name: str
    revision: str
    language: str
    max_length: int
    concatenated: bool
    batch_size: int
    dtype: str
    raw: Record = field(default_factory=dict)

    @property
    def probability(self) -> bool:
        return self.kind in PROBABILITY_KINDS


@dataclass(frozen=True)
class PairScore:
    value: float
    tokens: int
    truncated: bool
    extra: Record = field(default_factory=dict)

    def record(self) -> Record:
        return {
            "value": self.value,
            "tokens": self.tokens,
            "truncated": self.truncated,
            **self.extra,
        }


class PairScorer(Protocol):
    signature: str
    info: Record

    def score(self, pairs: Sequence[Pair]) -> list[PairScore]: ...


def parse_candidate(key: str, raw: Record) -> CandidateSpec:
    if raw["kind"] not in KINDS:
        raise SystemExit(f"{key}: unknown kind {raw['kind']!r}")
    if raw["language"] not in ("pt", "en"):
        raise SystemExit(f"{key}: language must be pt or en")
    return CandidateSpec(
        key=key,
        kind=raw["kind"],
        name=raw["name"],
        revision=raw["revision"],
        language=raw["language"],
        max_length=int(raw["max_length"]),
        concatenated=bool(raw["concatenated"]),
        batch_size=int(raw["batch_size"]),
        dtype=raw["dtype"],
        raw=raw,
    )


def with_model(spec: CandidateSpec, name: str, revision: str) -> CandidateSpec:
    return CandidateSpec(
        spec.key,
        spec.kind,
        name,
        revision,
        spec.language,
        spec.max_length,
        spec.concatenated,
        spec.batch_size,
        spec.dtype,
        spec.raw,
    )


def spec_signature(spec: CandidateSpec, device: str) -> str:
    payload = {
        "kind": spec.kind,
        "name": spec.name,
        "revision": spec.revision,
        "max_length": spec.max_length,
        "dtype": spec.dtype,
        "device": device,
        "prompt": spec.raw.get("prompt"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pair_key(signature: str, premise: str, hypothesis: str) -> str:
    text = json.dumps([signature, premise, hypothesis], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


class ScoreCache:
    def __init__(self, path: Path, signature: str) -> None:
        self.path = path
        self.signature = signature
        self.entries: dict[str, Record] = {}
        self.hits = 0
        self.computed = 0
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if row["signature"] == signature:
                            self.entries[row["key"]] = row["score"]

    def get(self, key: str) -> Record | None:
        return self.entries.get(key)

    def append(self, keys: list[str], scores: list[PairScore]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for key, score in zip(keys, scores, strict=True):
                record = score.record()
                self.entries[key] = record
                row = {"signature": self.signature, "key": key, "score": record}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.computed += len(keys)


def cached_scores(
    scorer: PairScorer,
    pairs: Sequence[Pair],
    cache: ScoreCache | None,
    chunk: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> list[Record]:
    keys = [pair_key(scorer.signature, premise, hypothesis) for premise, hypothesis in pairs]
    results: dict[str, Record] = {}
    todo: dict[str, Pair] = {}
    for key, pair in zip(keys, pairs, strict=True):
        found = cache.get(key) if cache is not None else None
        if found is not None:
            results[key] = found
        elif key not in todo:
            todo[key] = pair
    if cache is not None:
        cache.hits += len(pairs) - len(todo)
    pending = list(todo.items())
    for start in range(0, len(pending), chunk):
        part = pending[start : start + chunk]
        scores = scorer.score([pair for _, pair in part])
        if len(scores) != len(part):
            raise SystemExit(f"scorer returned {len(scores)} scores for {len(part)} pairs")
        part_keys = [key for key, _ in part]
        if cache is not None:
            cache.append(part_keys, scores)
        for key, score in zip(part_keys, scores, strict=True):
            results[key] = score.record()
        if progress is not None:
            progress(min(start + chunk, len(pending)), len(pending))
    return [results[key] for key in keys]


def length_order(pairs: Sequence[Pair]) -> list[int]:
    return sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))


def batched(order: list[int], size: int) -> list[list[int]]:
    return [order[start : start + size] for start in range(0, len(order), size)]


def two_way_probability(logits: np.ndarray, positive: int, negative: int) -> np.ndarray:
    pair = logits[:, [negative, positive]].astype(np.float64)
    pair -= pair.max(axis=1, keepdims=True)
    exp = np.exp(pair)
    return exp[:, 1] / exp.sum(axis=1)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values) / len(values)
