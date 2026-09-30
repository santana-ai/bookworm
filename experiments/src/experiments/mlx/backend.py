"""MLX backend: an engine that reuses the KV cache of the longest prompt prefix shared with the
previous call, and on top of it the same two classes as `experiments.actors.backend`, the
profile chat client and the simulation model."""

import copy
import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from experiments.actors.chat import (
    ChatResult,
    Generation,
    check_output_budget,
    letter_token_ids,
    strip_think,
    system_user_messages,
)

if TYPE_CHECKING:
    from experiments.actors.generate_profiles import ProfilesConfig

Record = dict[str, Any]
Sampler = Callable[[mx.array], mx.array]

MIN_SNAPSHOT_TOKENS = 64
DEFAULT_PREFILL_STEP = 2048


@dataclass(frozen=True)
class BackendOptions:
    template_kwargs: tuple[tuple[str, Any], ...] = ()
    prefix_cache: bool = True
    prefill_step: int = DEFAULT_PREFILL_STEP


def common_prefix(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, keepdims=True)


def set_wired_limit() -> None:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])


_locations: dict[str, Path] = {}


def register_model(name: str, local_dir: Path) -> None:
    """Load the model `name` from `local_dir` instead of the Hugging Face cache."""
    _locations[name] = local_dir


class MLXEngine:
    def __init__(self, name: str, options: BackendOptions) -> None:
        set_wired_limit()
        self.name = name
        self.options = options
        self.model, self.tokenizer = load(str(_locations.get(name, name)))
        self.eos_ids = set(self.tokenizer.eos_token_ids)
        self._previous: list[int] = []
        self._snapshot: tuple[list[int], list[Any]] | None = None

    def encode(self, messages: list[Record]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **dict(self.options.template_kwargs),
            )
        )

    def _forward(self, tokens: list[int], cache: list[Any]) -> mx.array:
        return self.model(mx.array(tokens)[None], cache=cache)

    def _consume(self, tokens: list[int], cache: list[Any]) -> None:
        step = self.options.prefill_step
        for start in range(0, len(tokens), step):
            self._forward(tokens[start : start + step], cache)
            mx.eval([entry.state for entry in cache])

    def last_logits(self, tokens: list[int], cache: list[Any]) -> mx.array:
        """Logits after the last token, prefilling the others in steps."""
        self._consume(tokens[:-1], cache)
        return self._forward(tokens[-1:], cache)[0, -1].astype(mx.float32)

    def reset_prefix(self) -> None:
        self._previous = []
        self._snapshot = None

    def _restore_snapshot(self, tokens: list[int]) -> tuple[int, list[Any]]:
        """A copy of the saved prefix cache when that prefix starts `tokens`, or a new cache."""
        if self._snapshot is not None:
            snapshot_tokens, snapshot_cache = self._snapshot
            if len(snapshot_tokens) < len(tokens) and tokens[: len(snapshot_tokens)] == (
                snapshot_tokens
            ):
                return len(snapshot_tokens), copy.deepcopy(snapshot_cache)
        return 0, make_prompt_cache(self.model)

    def prefill(self, tokens: list[int]) -> tuple[mx.array, list[Any]]:
        """Logits after `tokens` and the cache that holds them. With the prefix cache, the part
        shared with the previous prompt is saved as the new snapshot when it is long enough."""
        if not self.options.prefix_cache:
            fresh_cache = make_prompt_cache(self.model)
            return self.last_logits(tokens, fresh_cache), fresh_cache
        start, cache = self._restore_snapshot(tokens)
        shared = min(common_prefix(tokens, self._previous), len(tokens) - 1)
        if shared - start >= MIN_SNAPSHOT_TOKENS:
            self._consume(tokens[start:shared], cache)
            self._snapshot = (tokens[:shared], copy.deepcopy(cache))
            start = shared
        self._previous = tokens
        return self.last_logits(tokens[start:], cache), cache

    def step(self, token: int, cache: list[Any]) -> mx.array:
        return self._forward([token], cache)[0, -1].astype(mx.float32)

    def generate_tokens(
        self, tokens: list[int], max_new_tokens: int, sampler: Sampler | None = None
    ) -> list[int]:
        """Greedy decoding, or sampling with `sampler`, until an end token or the budget."""
        logits, cache = self.prefill(tokens)
        generated: list[int] = []
        while len(generated) < max_new_tokens:
            token = next_token(logits, sampler)
            generated.append(token)
            if token in self.eos_ids or len(generated) >= max_new_tokens:
                break
            logits = self.step(token, cache)
        mx.clear_cache()
        return generated

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def next_token(logits: mx.array, sampler: Sampler | None) -> int:
    scores = log_softmax(logits)
    if sampler is None:
        return int(mx.argmax(scores).item())
    return int(sampler(scores[None]).item())


_engines: dict[tuple[str, BackendOptions], MLXEngine] = {}


def engine_for(name: str, options: BackendOptions) -> MLXEngine:
    """One engine per model and options, so the profile and simulation stages share it."""
    key = (name, options)
    if key not in _engines:
        _engines[key] = MLXEngine(name, options)
    return _engines[key]


def release_engines() -> None:
    _engines.clear()
    gc.collect()
    mx.clear_cache()


class MLXChatClient:
    def __init__(self, config: "ProfilesConfig", options: BackendOptions) -> None:
        self._config = config
        self._engine = engine_for(config.model, options)

    def chat(self, system: str, user: str) -> ChatResult:
        config = self._config
        engine = self._engine
        tokens = engine.encode(system_user_messages(system, user))
        if config.seed is not None:
            mx.random.seed(config.seed)
        sampler = make_sampler(temp=config.temperature, top_p=config.top_p)
        generated = engine.generate_tokens(tokens, config.max_output_tokens, sampler=sampler)
        check_output_budget(len(generated), config.max_output_tokens)
        return ChatResult(
            text=strip_think(engine.decode(generated)),
            input_tokens=len(tokens),
            output_tokens=len(generated),
        )


class MLXSimulationModel:
    def __init__(self, name: str, device_map: str, options: BackendOptions) -> None:
        self._engine = engine_for(name, options)
        self.tokenizer = self._engine.tokenizer
        self.model = self._engine.model
        self.letter_ids = letter_token_ids(self.tokenizer)

    def letter_logprobs(self, messages: list[Record]) -> list[float]:
        logits, _ = self._engine.prefill(self._engine.encode(messages))
        return log_softmax(logits)[mx.array(self.letter_ids)].tolist()

    def generate(self, messages: list[Record], max_new_tokens: int) -> Generation:
        engine = self._engine
        generated = engine.generate_tokens(engine.encode(messages), max_new_tokens)
        return Generation(
            text=engine.decode(generated).strip(),
            output_tokens=len(generated),
            truncated=len(generated) >= max_new_tokens,
        )
