"""Backend-neutral pieces of a chat model call, shared by the transformers backend in
`experiments.actors.backend` and the MLX backend in `experiments.mlx.backend`."""

import re
from dataclasses import dataclass
from typing import Any, Protocol

Record = dict[str, Any]

LETTERS = ("A", "B", "C", "D")
THINK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Generation:
    text: str
    output_tokens: int
    truncated: bool


class ChatClient(Protocol):
    """Writes one profile from a system and a user prompt."""

    def chat(self, system: str, user: str) -> ChatResult: ...


class SimulationBackend(Protocol):
    """Scores the option letters and generates greedy text for the actor simulation."""

    letter_ids: list[int]

    def letter_logprobs(self, messages: list[Record]) -> list[float]: ...

    def generate(self, messages: list[Record], max_new_tokens: int) -> Generation: ...


def system_user_messages(system: str, user: str) -> list[Record]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_output_budget(output_tokens: int, max_output_tokens: int) -> None:
    """A profile that fills the whole token budget was cut, so it is an error."""
    if output_tokens >= max_output_tokens:
        raise RuntimeError(f"generation reached max_output_tokens={max_output_tokens}")


def strip_think(content: str) -> str:
    """The response without its leading think block, which must be closed."""
    if "<think>" in content and "</think>" not in content:
        raise RuntimeError("response contains an unterminated think block")
    return THINK_PATTERN.sub("", content).strip()


def letter_token_ids(tokenizer: Any) -> list[int]:
    """Token id of each option letter; each letter must be a single token."""
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != letter:
            raise SystemExit(
                f"letter {letter!r} is not a single token in this tokenizer: {encoded}"
            )
        ids.append(encoded[0])
    return ids
