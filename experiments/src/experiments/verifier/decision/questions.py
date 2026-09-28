"""Decision questions, their normalized answers and a deterministic fake model."""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Record = dict[str, Any]

QUESTION_TYPES = ("choice", "noul", "score")
NOUL_KEYS = ("false", "true")
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
PROBABILITY_TOLERANCE = 2e-3


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
