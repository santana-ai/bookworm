import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from experiments.common.cache_lock import CacheLockedError, acquire_writer_lock

Record = dict[str, Any]

QUESTION_TYPES = ("choice", "noul", "score")
NOUL_KEYS = ("false", "true")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_ALIASES = ("jev-latest", "jev-preview")
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
PROBABILITY_TOLERANCE = 2e-3
LAYA_ROUNDING_TOLERANCE = 1.5e-4
ERROR_SNIPPET = 300
LAYA_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
LAYA_REQUIRED = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)


class DecisionModelError(RuntimeError):
    pass


class MissingApiKeyError(DecisionModelError):
    pass


class ReplayMissError(DecisionModelError):
    pass


class JevRequestError(DecisionModelError):
    pass


class TransportError(DecisionModelError):
    pass


@dataclass(frozen=True)
class DecisionQuestion:
    key: str
    type: str
    instructions: str
    option_names: tuple[str, ...]
    option_texts: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if self.type not in QUESTION_TYPES:
            raise DecisionModelError(f"question {self.key!r}: type must be one of {QUESTION_TYPES}")
        if not self.instructions.strip():
            raise DecisionModelError(f"question {self.key!r}: empty instructions")
        if len(self.option_names) != len(self.option_texts):
            raise DecisionModelError(f"question {self.key!r}: option names and texts differ")
        if len(set(self.option_names)) != len(self.option_names):
            raise DecisionModelError(f"question {self.key!r}: option names must be distinct")
        if self.type == "noul" and self.option_names != NOUL_KEYS:
            raise DecisionModelError(f"question {self.key!r}: a noul answers {NOUL_KEYS}")
        if self.type == "score":
            levels = len(self.option_names)
            if self.option_names != tuple(str(i) for i in range(levels)):
                raise DecisionModelError(f"question {self.key!r}: score levels are named 0..n-1")
            if not 2 <= levels <= MAX_SCORE_LEVELS or not all(self.option_texts):
                raise DecisionModelError(
                    f"question {self.key!r}: a score needs 2 to {MAX_SCORE_LEVELS} described levels"
                )
        if self.type == "choice" and not 1 <= len(self.option_names) <= MAX_CHOICE_OPTIONS:
            raise DecisionModelError(
                f"question {self.key!r}: a choice needs 1 to {MAX_CHOICE_OPTIONS} options"
            )

    def payload(self) -> Record:
        if self.type == "choice":
            criteria = dict(zip(self.option_names, self.option_texts, strict=True))
            return {"type": "choice", "instructions": self.instructions, "criteria": criteria}
        if self.type == "score":
            return {
                "type": "score",
                "instructions": self.instructions,
                "criteria": list(self.option_texts),
            }
        payload: Record = {"type": "noul", "instructions": self.instructions}
        described = {
            name: text
            for name, text in zip(self.option_names, self.option_texts, strict=True)
            if text
        }
        if described:
            payload["criteria"] = described
        return payload

    def payload_json(self) -> str:
        return json.dumps(self.payload(), ensure_ascii=False, separators=(",", ":"))


def choice_question(
    key: str, instructions: str, options: Sequence[tuple[str, str | None]]
) -> DecisionQuestion:
    return DecisionQuestion(
        key,
        "choice",
        instructions,
        tuple(name for name, _ in options),
        tuple(text for _, text in options),
    )


def noul_question(
    key: str, instructions: str, true_text: str | None = None, false_text: str | None = None
) -> DecisionQuestion:
    return DecisionQuestion(key, "noul", instructions, NOUL_KEYS, (false_text, true_text))


def score_question(key: str, instructions: str, levels: Sequence[str]) -> DecisionQuestion:
    return DecisionQuestion(
        key, "score", instructions, tuple(str(i) for i in range(len(levels))), tuple(levels)
    )


@dataclass(frozen=True)
class DecisionAnswer:
    question: str
    type: str
    probabilities: dict[str, float]
    input_tokens: int | None = None
    state_tokens: int | None = None
    fitted_tokens: int | None = None
    truncated: bool = False
    overflow: bool = False
    premise_chars: int | None = None
    premise_chars_kept: int | None = None

    @property
    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    def expected_level(self) -> float:
        return float(sum(int(name) * value for name, value in self.probabilities.items()))

    def record(self) -> Record:
        record: Record = {"probabilities": dict(self.probabilities), "choice": self.choice}
        if self.type == "score":
            record["expected_level"] = self.expected_level()
        return record


def normalized_probabilities(question: DecisionQuestion, values: Mapping[str, float]) -> dict:
    if set(values) != set(question.option_names):
        raise DecisionModelError(
            f"question {question.key!r}: answer options {sorted(values)} != "
            f"{sorted(question.option_names)}"
        )
    ordered = {name: float(values[name]) for name in question.option_names}
    total = sum(ordered.values())
    if any(not math.isfinite(v) or v < 0 for v in ordered.values()) or abs(total - 1) > (
        PROBABILITY_TOLERANCE
    ):
        raise DecisionModelError(f"question {question.key!r}: probabilities {ordered} sum {total}")
    return ordered


def parse_answer(
    question: DecisionQuestion, raw: Record, input_tokens: int | None
) -> DecisionAnswer:
    if raw.get("type") != question.type:
        raise DecisionModelError(
            f"question {question.key!r}: answer type {raw.get('type')!r} != {question.type!r}"
        )
    if question.type == "noul":
        value = float(raw["noul"])
        if not 0.0 <= value <= 1.0:
            raise DecisionModelError(f"question {question.key!r}: noul {value} outside [0, 1]")
        probabilities = {"false": 1.0 - value, "true": value}
    else:
        probabilities = normalized_probabilities(question, raw["probabilities"])
    return DecisionAnswer(question.key, question.type, probabilities, input_tokens=input_tokens)


class DecisionModel(Protocol):
    name: str
    revision: str

    def describe(self) -> Record: ...

    def predict_batch(
        self, states: Sequence[Record], questions: Sequence[DecisionQuestion]
    ) -> list[dict[str, DecisionAnswer]]: ...


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def serialize_state(state: Record) -> str:
    return json.dumps(state, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fake_answer(state: Record, question: DecisionQuestion) -> dict[str, float]:
    seed = sha256_text(f"{question.key}\x1e{serialize_state(state)}")
    raw = [int(seed[8 * i : 8 * i + 8], 16) + 1 for i in range(len(question.option_names))]
    total = sum(raw)
    return {name: value / total for name, value in zip(question.option_names, raw, strict=True)}


@dataclass
class FakeDecisionModel:
    name: str = "fake/decision"
    revision: str = "0" * 40
    answer_fn: Callable[[Record, DecisionQuestion], Mapping[str, float]] = fake_answer
    calls: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)

    def describe(self) -> Record:
        return {
            "kind": "fake",
            "name": self.name,
            "revision": self.revision,
            "calls": len(self.calls),
        }

    def predict_batch(
        self, states: Sequence[Record], questions: Sequence[DecisionQuestion]
    ) -> list[dict[str, DecisionAnswer]]:
        self.calls.append((len(states), tuple(q.key for q in questions)))
        return [
            {
                q.key: DecisionAnswer(
                    q.key, q.type, normalized_probabilities(q, self.answer_fn(state, q))
                )
                for q in questions
            }
            for state in states
        ]


def incomplete_tail(data: bytes) -> int:
    if not data or data.endswith(b"\n"):
        return 0
    return len(data) - (data.rfind(b"\n") + 1)


def read_jsonl_lines(path: Path) -> tuple[list[Record], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    tail = incomplete_tail(data)
    records = []
    for number, line in enumerate(data[: len(data) - tail].decode().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise DecisionModelError(
                f"{path}: line {number} is not valid JSON ({error})"
            ) from error
    return records, tail


def append_jsonl(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        f.flush()
        os.fsync(f.fileno())


@dataclass
class AnswerCache:
    path: Path
    signature: str
    entries: dict[str, Record] = field(default_factory=dict)
    other_signature_lines: int = 0
    incomplete_tail_bytes: int = 0

    @classmethod
    def open(cls, path: Path, signature: str) -> "AnswerCache":
        try:
            acquire_writer_lock(path)
        except CacheLockedError as error:
            raise DecisionModelError(str(error)) from error
        cache = cls(path, signature)
        records, cache.incomplete_tail_bytes = read_jsonl_lines(path)
        if cache.incomplete_tail_bytes:
            data = path.read_bytes()
            with open(path, "r+b") as f:
                f.truncate(len(data) - cache.incomplete_tail_bytes)
        for record in records:
            if record.get("signature") != signature:
                cache.other_signature_lines += 1
                continue
            cache.entries.setdefault(record["key"], record)
        return cache

    def key(self, question: DecisionQuestion, state: Record) -> str:
        return sha256_text(
            f"{self.signature}\x1e{question.payload_json()}\x1e{serialize_state(state)}"
        )

    def append(self, records: Sequence[Record]) -> None:
        stamped = [{"signature": self.signature, **record} for record in records]
        append_jsonl(self.path, stamped)
        for record in stamped:
            self.entries.setdefault(record["key"], record)


@dataclass(frozen=True)
class FittedState:
    state: Record
    state_tokens: int
    fitted_tokens: int
    truncated: bool
    overflow: bool
    chars: int
    chars_kept: int


def cut_points(text: str) -> list[int]:
    points = {0, len(text)}
    points.update(index for index, character in enumerate(text) if character.isspace())
    return sorted(points)


def fit_state(state: Record, key: str, room: int, count: Callable[[str], int]) -> FittedState:
    text = state[key]
    full = count(serialize_state(state))
    if full <= room:
        return FittedState(state, full, full, False, False, len(text), len(text))

    def tokens_at(cut: int) -> int:
        return count(serialize_state({**state, key: text[:cut].rstrip()}))

    points = cut_points(text)
    low, high = 0, len(points) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if tokens_at(points[middle]) <= room:
            low = middle
        else:
            high = middle - 1
    while low > 0 and tokens_at(points[low]) > room:
        low -= 1
    kept = text[: points[low]].rstrip()
    fitted = {**state, key: kept}
    fitted_tokens = count(serialize_state(fitted))
    return FittedState(
        fitted, full, fitted_tokens, True, fitted_tokens > room, len(text), len(kept)
    )


def softmax_rows(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def library_values(question: DecisionQuestion, answer: Record) -> dict[str, float]:
    if question.type == "noul":
        return {"true": float(answer["noul"])}
    return {name: float(value) for name, value in answer["probabilities"].items()}


def check_library_rounding(
    question: DecisionQuestion, probabilities: dict[str, float], answer: Record
) -> float:
    reported = library_values(question, answer)
    gaps = [abs(probabilities[name] - value) for name, value in reported.items()]
    worst = max(gaps)
    if worst > LAYA_ROUNDING_TOLERANCE:
        raise DecisionModelError(
            f"question {question.key!r}: probabilities from the captured logits differ from the "
            f"library answer by {worst:.2e}; the logit capture is not aligned with the answers"
        )
    return worst


@dataclass(frozen=True)
class LayaSpec:
    repo_id: str
    revision: str
    subfolder: str
    device: str
    batch_size: int
    max_length: int
    head_max_length: int
    truncate_key: str
    cache_dir: Path

    @property
    def label(self) -> str:
        return self.subfolder or "root"


def laya_checkpoint_dir(repo_id: str, revision: str, subfolder: str) -> Path:
    from huggingface_hub import try_to_load_from_cache

    prefix = f"{subfolder}/" if subfolder else ""
    found = try_to_load_from_cache(repo_id, f"{prefix}rl_agent_config.json", revision=revision)
    if not isinstance(found, str):
        raise DecisionModelError(
            f"{repo_id}@{revision} {subfolder or 'root'}: checkpoint not in the local Hugging Face "
            "cache; run python -m experiments.verifier.nli_experiments fetch"
        )
    directory = Path(found).parent
    snapshot = directory.parent if subfolder else directory
    if snapshot.name != revision:
        raise DecisionModelError(f"{directory} is not in the pinned snapshot {revision}")
    missing = [name for name in LAYA_REQUIRED if not (directory / name).exists()]
    if missing:
        raise DecisionModelError(f"{directory}: missing {missing}; rerun fetch")
    return directory


def fetch_laya(repo_id: str, revision: str, subfolder: str) -> Path:
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    snapshot_download(
        repo_id, revision=revision, allow_patterns=[prefix + name for name in LAYA_FILES]
    )
    return laya_checkpoint_dir(repo_id, revision, subfolder)


def state_token_count(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text.replace(tokenizer.mask_token, " "), add_special_tokens=False)["input_ids"]
    return len(ids)


def question_room(
    tokenizer: Any, question: DecisionQuestion, max_length: int, head_max_length: int
) -> int:
    from laya.agent import Agent
    from laya.common import build_sequence

    internal = Agent._to_internal(question.payload())
    ids, markers = build_sequence(tokenizer, "", internal, max_length, head_max_length)
    if len(markers) != len(question.option_names):
        raise DecisionModelError(
            f"question {question.key!r}: options exceed head_max_len {head_max_length}"
        )
    return max_length - len(ids)


class LayaTokenizer:
    def __init__(self, repo_id: str, revision: str, subfolder: str) -> None:
        from transformers import AutoTokenizer

        self.directory = laya_checkpoint_dir(repo_id, revision, subfolder)
        self.cfg = json.loads((self.directory / "rl_agent_config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory / "tokenizer"))
        self.max_length = int(self.cfg["max_len"])
        self.head_max_length = int(self.cfg["head_max_len"])
        self.counts: dict[str, int] = {}

    def count(self, text: str) -> int:
        cached = self.counts.get(text)
        if cached is None:
            cached = state_token_count(self.tokenizer, text)
            self.counts[text] = cached
        return cached

    def room(self, question: DecisionQuestion) -> int:
        return question_room(self.tokenizer, question, self.max_length, self.head_max_length)


def file_record(path: Path) -> Record:
    return {
        "path": str(path),
        "is_symlink": path.is_symlink(),
        "blob": path.resolve().name,
        "bytes": path.stat().st_size,
    }


class LayaDecisionModel:
    def __init__(self, spec: LayaSpec, package_version: str | None = None) -> None:
        import laya

        self.spec = spec
        self.name = spec.repo_id if not spec.subfolder else f"{spec.repo_id}/{spec.subfolder}"
        self.revision = spec.revision
        self.package_version = package_version or laya.__version__
        self.directory = laya_checkpoint_dir(spec.repo_id, spec.revision, spec.subfolder)
        tokenizer_config = self.directory / "tokenizer" / "tokenizer_config.json"
        before = file_record(tokenizer_config)
        started = time.perf_counter()
        self.agent = laya.load(str(self.directory), device=spec.device)
        self.load_seconds = time.perf_counter() - started
        after = file_record(tokenizer_config)
        if before != after:
            raise DecisionModelError(
                f"laya rewrote {tokenizer_config} while loading; the pinned snapshot changed"
            )
        cfg = self.agent.cfg
        if (cfg.get("max_len"), cfg.get("head_max_len")) != (spec.max_length, spec.head_max_length):
            raise DecisionModelError(
                f"{self.name}: checkpoint max_len {cfg.get('max_len')} and head_max_len "
                f"{cfg.get('head_max_len')} != configured {spec.max_length}, {spec.head_max_length}"
            )
        if self.agent.device.type != spec.device:
            raise DecisionModelError(
                f"{self.name}: loaded on {self.agent.device}, not {spec.device}"
            )
        self.signature = (
            f"laya@{self.package_version}@{spec.repo_id}@{spec.revision}@{spec.label}@"
            f"{spec.device}@float32@{spec.max_length}@{spec.head_max_length}@{spec.truncate_key}"
        )
        path = spec.cache_dir / f"laya_{spec.label}_{spec.revision[:12]}_{spec.device}.jsonl"
        self.cache = AnswerCache.open(path, self.signature)
        self.entries_at_start = len(self.cache.entries)
        self.captured: list[Any] = []
        self.agent.model.register_forward_hook(self._capture)
        self.token_counts: dict[str, int] = {}
        self.counters: Record = {
            "requests": 0,
            "computed": 0,
            "cache_hits": 0,
            "forward_seconds": 0.0,
            "computed_input_tokens": 0,
            "batches": 0,
            "max_rounding_gap": 0.0,
        }
        self.tokenizer_config = before

    def _capture(self, module: Any, inputs: Any, output: Any) -> None:
        self.captured.append(output[0].detach().float().cpu().numpy().astype(np.float64))

    def count_tokens(self, text: str) -> int:
        cached = self.token_counts.get(text)
        if cached is None:
            cached = state_token_count(self.agent.tok, text)
            self.token_counts[text] = cached
        return cached

    def room(self, question: DecisionQuestion) -> int:
        return question_room(
            self.agent.tok, question, self.spec.max_length, self.spec.head_max_length
        )

    def temperature(self, question: DecisionQuestion) -> float:
        from laya.common import QTYPES, temp_bucket

        qtype = QTYPES[question.type]
        bucket = temp_bucket(qtype, len(question.option_names))
        return float(self.agent.temperature_by_options.get(bucket, self.agent.temperature[qtype]))

    def describe(self) -> Record:
        cfg = self.agent.cfg
        return {
            "kind": "laya",
            "name": self.name,
            "revision": self.revision,
            "subfolder": self.spec.subfolder,
            "laya_version": self.package_version,
            "checkpoint_dir": str(self.directory),
            "weights_file": file_record(self.directory / "model.safetensors"),
            "tokenizer_config": self.tokenizer_config,
            "encoder": cfg.get("encoder"),
            "max_len": cfg.get("max_len"),
            "head_max_len": cfg.get("head_max_len"),
            "temperature_applied": list(self.agent.temperature),
            "temperature_by_options_applied": dict(self.agent.temperature_by_options),
            "device": str(self.agent.device),
            "dtype": str(self.agent.dtype),
            "parameters": int(sum(p.numel() for p in self.agent.model.parameters())),
            "load_seconds": round(self.load_seconds, 1),
            "batch_size": self.spec.batch_size,
            "truncation": (
                f"when the serialized state does not fit max_len minus the question prefix, "
                f"{self.spec.truncate_key!r} is cut at the last whitespace that fits, so the "
                "other keys stay whole"
            ),
            "answer_cache": {
                "path": str(self.cache.path),
                "signature": self.signature,
                "entries_at_start": self.entries_at_start,
                "other_signature_lines": self.cache.other_signature_lines,
            },
            "counters": {
                **self.counters,
                "forward_seconds": round(self.counters["forward_seconds"], 2),
            },
        }

    def forward(self, question: DecisionQuestion, states: list[Record]) -> tuple[np.ndarray, list]:
        self.captured.clear()
        results = self.agent.predict_batch(states, {question.key: question.payload()})
        if self.agent.device.type != self.spec.device:
            raise DecisionModelError(
                f"{self.name}: laya moved the model from {self.spec.device} to "
                f"{self.agent.device} during a forward pass (out of memory); nothing from this "
                f"batch was cached; free memory or score with --device {self.agent.device.type}, "
                "which writes to its own cache file"
            )
        if len(self.captured) != 1 or self.captured[0].shape[0] != len(states):
            raise DecisionModelError(
                f"question {question.key!r}: captured {len(self.captured)} forward outputs for "
                f"one batch of {len(states)} states"
            )
        logits = self.captured[0][:, : len(question.option_names)]
        self.captured.clear()
        return logits, results

    def compute(
        self, question: DecisionQuestion, fitted: list[FittedState], keys: list[str]
    ) -> None:
        first: dict[str, int] = {}
        for index, key in enumerate(keys):
            if key not in self.cache.entries:
                first.setdefault(key, index)
        order = sorted(first.values(), key=lambda i: -fitted[i].fitted_tokens)
        temperature = self.temperature(question)
        for start in range(0, len(order), self.spec.batch_size):
            rows = order[start : start + self.spec.batch_size]
            began = time.perf_counter()
            logits, results = self.forward(question, [fitted[i].state for i in rows])
            seconds = time.perf_counter() - began
            probabilities = softmax_rows(logits, temperature)
            records = []
            for position, index in enumerate(rows):
                values = dict(zip(question.option_names, probabilities[position], strict=True))
                answer = results[position]["answers"][question.key]
                gap = check_library_rounding(question, values, answer)
                self.counters["max_rounding_gap"] = max(self.counters["max_rounding_gap"], gap)
                item = fitted[index]
                tokens = int(results[position]["usage"]["input_tokens"])
                records.append(
                    {
                        "key": keys[index],
                        "question": question.key,
                        "logits": logits[position].tolist(),
                        "temperature": temperature,
                        "input_tokens": tokens,
                        "state_tokens": item.state_tokens,
                        "fitted_tokens": item.fitted_tokens,
                        "truncated": item.truncated,
                        "overflow": item.overflow,
                        "premise_chars": item.chars,
                        "premise_chars_kept": item.chars_kept,
                        "batch_size": len(rows),
                        "seconds": seconds / len(rows),
                    }
                )
                self.counters["computed_input_tokens"] += tokens
            self.cache.append(records)
            self.counters["forward_seconds"] += seconds
            self.counters["batches"] += 1
            self.counters["computed"] += len(rows)

    def answer(self, question: DecisionQuestion, record: Record) -> DecisionAnswer:
        logits = np.array([record["logits"]], dtype=np.float64)
        values = softmax_rows(logits, float(record["temperature"]))[0]
        return DecisionAnswer(
            question.key,
            question.type,
            dict(zip(question.option_names, (float(v) for v in values), strict=True)),
            input_tokens=record["input_tokens"],
            state_tokens=record["state_tokens"],
            fitted_tokens=record["fitted_tokens"],
            truncated=record["truncated"],
            overflow=record["overflow"],
            premise_chars=record["premise_chars"],
            premise_chars_kept=record["premise_chars_kept"],
        )

    def predict_batch(
        self, states: Sequence[Record], questions: Sequence[DecisionQuestion]
    ) -> list[dict[str, DecisionAnswer]]:
        results: list[dict[str, DecisionAnswer]] = [{} for _ in states]
        for question in questions:
            self.agent._check_question(question.key, question.payload())
            room = self.room(question)
            fitted = [
                fit_state(dict(state), self.spec.truncate_key, room, self.count_tokens)
                for state in states
            ]
            keys = [self.cache.key(question, dict(state)) for state in states]
            computed_before = self.counters["computed"]
            self.compute(question, fitted, keys)
            new = self.counters["computed"] - computed_before
            self.counters["requests"] += len(states)
            self.counters["cache_hits"] += len(states) - new
            for result, key in zip(results, keys, strict=True):
                result[question.key] = self.answer(question, self.cache.entries[key])
        self.token_counts.clear()
        return results


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class Transport(Protocol):
    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse: ...


def urllib_transport(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                response.status,
                {name.lower(): value for name, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return HttpResponse(
            error.code,
            {name.lower(): value for name, value in (error.headers or {}).items()},
            error.read() or b"",
        )
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        raise TransportError(f"{type(error).__name__}: {error}") from error


@dataclass(frozen=True)
class JevSpec:
    model_version: str
    endpoint: str
    api_key_env: str
    timeout_seconds: float
    max_retries: int
    backoff_initial_seconds: float
    backoff_max_seconds: float
    requests_per_minute: float
    max_concurrency: int
    price_usd_per_million_input_tokens: float
    cache_dir: Path

    @property
    def pinned(self) -> bool:
        return self.model_version not in JEV_ALIASES

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"jev_{self.model_version}.jsonl"


def missing_key_message(env: str) -> str:
    return (
        f"the environment variable {env} is not set, so the Jev scorers cannot call the TypeSafe "
        f"API; export {env} with an API key, or score with --decision-mode replay to answer only "
        "from recorded responses"
    )


def jev_request_body(
    model_version: str, state: Record, questions: Sequence[DecisionQuestion]
) -> bytes:
    body = {
        "state": state,
        "model": model_version,
        "questions": {question.key: question.payload() for question in questions},
    }
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def jev_request_key(model_version: str, body: bytes) -> str:
    return hashlib.sha256(model_version.encode() + b"\x1e" + body).hexdigest()


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def backoff_seconds(spec: JevSpec, attempt: int) -> float:
    return min(spec.backoff_max_seconds, spec.backoff_initial_seconds * 2**attempt)


def parse_jev_response(
    spec: JevSpec, questions: Sequence[DecisionQuestion], response: Record
) -> dict[str, DecisionAnswer]:
    answered_by = response.get("model")
    if spec.pinned and answered_by != spec.model_version:
        raise JevRequestError(
            f"answered by model {answered_by!r}, not the pinned {spec.model_version!r}"
        )
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != {question.key for question in questions}:
        raise JevRequestError(
            f"response answers {sorted(answers or {})} do not match the questions"
        )
    usage = response.get("usage") or {}
    tokens = usage.get("input_tokens")
    return {
        question.key: parse_answer(question, answers[question.key], tokens)
        for question in questions
    }


@dataclass
class ReplayCache:
    path: Path
    entries: dict[str, Record] = field(default_factory=dict)
    lines: int = 0
    failed_lines: int = 0
    incomplete_tail_bytes: int = 0

    @classmethod
    def open(cls, path: Path, writable: bool = False) -> "ReplayCache":
        cache = cls(path)
        records, cache.incomplete_tail_bytes = read_jsonl_lines(path)
        if writable and cache.incomplete_tail_bytes:
            data = path.read_bytes()
            with open(path, "r+b") as f:
                f.truncate(len(data) - cache.incomplete_tail_bytes)
        for record in records:
            cache.lines += 1
            if record.get("status") != 200:
                cache.failed_lines += 1
                continue
            cache.entries.setdefault(record["key"], record)
        return cache

    def summary(self) -> Record:
        return {
            "path": str(self.path),
            "lines": self.lines,
            "replayable": len(self.entries),
            "failed_attempt_lines": self.failed_lines,
            "incomplete_tail_bytes": self.incomplete_tail_bytes,
        }


class RateLimiter:
    def __init__(
        self,
        per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.next_start = 0.0

    def wait(self) -> None:
        with self.lock:
            current = self.clock()
            start = max(current, self.next_start)
            self.next_start = start + self.interval
        if start > current:
            self.sleep(start - current)


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def body_text(body: bytes, secret: str) -> str:
    return redact(body.decode(errors="replace"), secret)[:ERROR_SNIPPET]


def decode_json(body: bytes) -> Record | None:
    try:
        value = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class JevDecisionModel:
    def __init__(
        self,
        spec: JevSpec,
        transport: Transport = urllib_transport,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        source = os.environ if environ is None else environ
        secret = source.get(spec.api_key_env, "")
        if not secret:
            raise MissingApiKeyError(missing_key_message(spec.api_key_env))
        self._secret = secret
        self.spec = spec
        self.name = "typesafe/jev"
        self.revision = spec.model_version
        self.transport = transport
        self.clock = clock
        self.sleep = sleep
        self.limiter = RateLimiter(spec.requests_per_minute, clock, sleep)
        self.cache = ReplayCache.open(spec.cache_path, writable=True)
        self.entries_at_start = len(self.cache.entries)
        self.lock = threading.Lock()
        self.counters: Record = {
            "requests": 0,
            "sent": 0,
            "attempts": 0,
            "retries": 0,
            "cache_hits": 0,
            "billed_input_tokens": 0,
            "seconds": 0.0,
        }

    def __repr__(self) -> str:
        return f"JevDecisionModel({self.spec.model_version})"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._secret}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def record(self, key: str, body: bytes, attempt: int, outcome: Record) -> Record:
        line = {
            "key": key,
            "model_version": self.spec.model_version,
            "endpoint": self.spec.endpoint,
            "attempt": attempt,
            "created_at": now(),
            "request": json.loads(body.decode()),
            **outcome,
        }
        with self.lock:
            append_jsonl(self.cache.path, [line])
            self.counters["attempts"] += 1
            if line.get("status") == 200:
                self.cache.entries.setdefault(key, line)
        return line

    def request(self, key: str, body: bytes, questions: Sequence[DecisionQuestion]) -> None:
        last = "no attempt made"
        for attempt in range(self.spec.max_retries + 1):
            self.limiter.wait()
            started = self.clock()
            try:
                response = self.transport(
                    self.spec.endpoint, body, self.headers(), self.spec.timeout_seconds
                )
            except TransportError as error:
                latency = self.clock() - started
                last = redact(str(error), self._secret)[:ERROR_SNIPPET]
                self.record(
                    key, body, attempt, {"status": None, "error": last, "latency_seconds": latency}
                )
                delay = backoff_seconds(self.spec, attempt)
            else:
                latency = self.clock() - started
                parsed = decode_json(response.body)
                outcome: Record = {
                    "status": response.status,
                    "latency_seconds": latency,
                    "retry_after": response.headers.get("retry-after"),
                }
                if parsed is not None:
                    outcome["response"] = parsed
                else:
                    outcome["response_text"] = body_text(response.body, self._secret)
                if response.status == 200:
                    if parsed is None:
                        self.record(key, body, attempt, {**outcome, "status": "invalid_json"})
                        raise JevRequestError("HTTP 200 with a body that is not a JSON object")
                    self.record(key, body, attempt, outcome)
                    parse_jev_response(self.spec, questions, parsed)
                    with self.lock:
                        self.counters["sent"] += 1
                        self.counters["retries"] += attempt
                        self.counters["seconds"] += latency
                        usage = parsed.get("usage") or {}
                        self.counters["billed_input_tokens"] += int(usage.get("input_tokens") or 0)
                    return
                self.record(key, body, attempt, outcome)
                last = f"HTTP {response.status}: {body_text(response.body, self._secret)}"
                if response.status not in RETRY_STATUSES:
                    raise JevRequestError(last)
                waited = retry_after_seconds(response.headers)
                delay = waited if waited is not None else backoff_seconds(self.spec, attempt)
            if attempt < self.spec.max_retries:
                self.sleep(delay)
        raise JevRequestError(f"gave up after {self.spec.max_retries + 1} attempts; last: {last}")

    def predict_batch(
        self, states: Sequence[Record], questions: Sequence[DecisionQuestion]
    ) -> list[dict[str, DecisionAnswer]]:
        bodies = [jev_request_body(self.spec.model_version, dict(s), questions) for s in states]
        keys = [jev_request_key(self.spec.model_version, body) for body in bodies]
        todo = {
            key: body
            for key, body in zip(keys, bodies, strict=True)
            if key not in self.cache.entries
        }
        self.counters["requests"] += len(keys)
        self.counters["cache_hits"] += len(keys) - sum(1 for key in keys if key in todo)
        if todo:
            with ThreadPoolExecutor(max_workers=self.spec.max_concurrency) as pool:
                futures = [
                    pool.submit(self.request, key, body, questions) for key, body in todo.items()
                ]
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                failed = [future for future in done if future.exception() is not None]
                if failed:
                    for future in futures:
                        future.cancel()
                    raise failed[0].exception() or JevRequestError("request failed")
        return [
            parse_jev_response(self.spec, questions, self.cache.entries[key]["response"])
            for key in keys
        ]

    def describe(self) -> Record:
        tokens = self.counters["billed_input_tokens"]
        return {
            "kind": "jev",
            "mode": "live",
            "name": self.name,
            "revision": self.revision,
            "model_version": self.spec.model_version,
            "pinned": self.spec.pinned,
            "endpoint": self.spec.endpoint,
            "api_key_env": self.spec.api_key_env,
            "api_key_set": True,
            "rate_limit": {
                "requests_per_minute": self.spec.requests_per_minute,
                "max_concurrency": self.spec.max_concurrency,
                "max_retries": self.spec.max_retries,
                "backoff_initial_seconds": self.spec.backoff_initial_seconds,
                "backoff_max_seconds": self.spec.backoff_max_seconds,
                "timeout_seconds": self.spec.timeout_seconds,
            },
            "replay_cache": {**self.cache.summary(), "replayable_at_start": self.entries_at_start},
            "counters": {**self.counters, "seconds": round(self.counters["seconds"], 2)},
            "billed_cost_usd": round(
                tokens * self.spec.price_usd_per_million_input_tokens / 1e6, 4
            ),
        }


class ReplayDecisionModel:
    def __init__(self, spec: JevSpec) -> None:
        self.spec = spec
        self.name = "typesafe/jev"
        self.revision = spec.model_version
        self.cache = ReplayCache.open(spec.cache_path)
        self.counters: Record = {"requests": 0, "cache_hits": 0}

    def predict_batch(
        self, states: Sequence[Record], questions: Sequence[DecisionQuestion]
    ) -> list[dict[str, DecisionAnswer]]:
        keys = [
            jev_request_key(
                self.spec.model_version,
                jev_request_body(self.spec.model_version, dict(s), questions),
            )
            for s in states
        ]
        missing = [key for key in dict.fromkeys(keys) if key not in self.cache.entries]
        if missing:
            raise ReplayMissError(
                f"{len(missing)} of {len(set(keys))} distinct Jev requests have no recorded "
                f"HTTP 200 response in {self.cache.path} for {self.spec.model_version}; score "
                f"live with {self.spec.api_key_env} set to record them"
            )
        self.counters["requests"] += len(keys)
        self.counters["cache_hits"] += len(keys)
        return [
            parse_jev_response(self.spec, questions, self.cache.entries[key]["response"])
            for key in keys
        ]

    def describe(self) -> Record:
        return {
            "kind": "jev",
            "mode": "replay",
            "name": self.name,
            "revision": self.revision,
            "model_version": self.spec.model_version,
            "pinned": self.spec.pinned,
            "endpoint": self.spec.endpoint,
            "api_key_env": self.spec.api_key_env,
            "replay_cache": self.cache.summary(),
            "counters": dict(self.counters),
        }
