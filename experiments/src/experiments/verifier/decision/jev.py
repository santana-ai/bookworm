"""Remote JEV decision model with rate limiting, retries and a replay cache."""

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
from pathlib import Path
from typing import Any, Protocol

from experiments.common.reporting import utc_timestamp
from experiments.verifier.decision.answer_cache import append_jsonl, read_jsonl_lines
from experiments.verifier.decision.questions import (
    DecisionAnswer,
    DecisionQuestion,
    JevRequestError,
    MissingApiKeyError,
    ReplayMissError,
    TransportError,
    parse_answer,
)

Record = dict[str, Any]
DEFAULT_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_ALIASES = ("jev-latest", "jev-preview")
HTTP_OK = 200
SECONDS_PER_MINUTE = 60.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
ERROR_SNIPPET = 300


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
            if record.get("status") != HTTP_OK:
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
        self.interval = SECONDS_PER_MINUTE / per_minute if per_minute > 0 else 0.0
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
            "created_at": utc_timestamp(),
            "request": json.loads(body.decode()),
            **outcome,
        }
        with self.lock:
            append_jsonl(self.cache.path, [line])
            self.counters["attempts"] += 1
            if line.get("status") == HTTP_OK:
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
                if response.status == HTTP_OK:
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
