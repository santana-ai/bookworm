"""Local Laya decision model: checkpoint fetch, state fitting and batched scoring."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.verifier.decision.answer_cache import AnswerCache
from experiments.verifier.decision.questions import (
    DecisionAnswer,
    DecisionModelError,
    DecisionQuestion,
    serialize_state,
)

Record = dict[str, Any]
LAYA_ROUNDING_TOLERANCE = 1.5e-4
REVISION_PREFIX = 12
LAYA_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
LAYA_REQUIRED = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)


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
        path = (
            spec.cache_dir
            / f"laya_{spec.label}_{spec.revision[:REVISION_PREFIX]}_{spec.device}.jsonl"
        )
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
