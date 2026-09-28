import copy
import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from utils.actor_simulation import Generation, letter_token_ids
from utils.generate_actor_profiles import THINK_PATTERN, ChatResult, ProfilesConfig

Record = dict[str, Any]
Sampler = Callable[[mx.array], mx.array]

MIN_SNAPSHOT_TOKENS = 64


@dataclass(frozen=True)
class BackendOptions:
    template_kwargs: tuple[tuple[str, Any], ...] = ()
    prefix_cache: bool = True
    prefill_step: int = 2048


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

    def _last_logits(self, tokens: list[int], cache: list[Any]) -> mx.array:
        self._consume(tokens[:-1], cache)
        return self._forward(tokens[-1:], cache)[0, -1].astype(mx.float32)

    def reset_prefix(self) -> None:
        self._previous = []
        self._snapshot = None

    def prefill(self, tokens: list[int]) -> tuple[mx.array, list[Any]]:
        if not self.options.prefix_cache:
            cache = make_prompt_cache(self.model)
            return self._last_logits(tokens, cache), cache
        start = 0
        cache: list[Any] | None = None
        if self._snapshot is not None:
            snapshot_tokens, snapshot_cache = self._snapshot
            if len(snapshot_tokens) < len(tokens) and tokens[: len(snapshot_tokens)] == (
                snapshot_tokens
            ):
                cache = copy.deepcopy(snapshot_cache)
                start = len(snapshot_tokens)
        if cache is None:
            cache = make_prompt_cache(self.model)
        shared = min(common_prefix(tokens, self._previous), len(tokens) - 1)
        if shared - start >= MIN_SNAPSHOT_TOKENS:
            self._consume(tokens[start:shared], cache)
            self._snapshot = (tokens[:shared], copy.deepcopy(cache))
            start = shared
        self._previous = tokens
        return self._last_logits(tokens[start:], cache), cache

    def step(self, token: int, cache: list[Any]) -> mx.array:
        return self._forward([token], cache)[0, -1].astype(mx.float32)

    def generate_tokens(
        self, tokens: list[int], max_new_tokens: int, sampler: Sampler | None = None
    ) -> list[int]:
        logits, cache = self.prefill(tokens)
        generated: list[int] = []
        while len(generated) < max_new_tokens:
            scores = log_softmax(logits)
            if sampler is None:
                token = int(mx.argmax(scores).item())
            else:
                token = int(sampler(scores[None]).item())
            generated.append(token)
            if token in self.eos_ids or len(generated) >= max_new_tokens:
                break
            logits = self.step(token, cache)
        mx.clear_cache()
        return generated

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


_engines: dict[tuple[str, BackendOptions], MLXEngine] = {}


def engine_for(name: str, options: BackendOptions) -> MLXEngine:
    key = (name, options)
    if key not in _engines:
        _engines[key] = MLXEngine(name, options)
    return _engines[key]


def release_engines() -> None:
    _engines.clear()
    gc.collect()
    mx.clear_cache()


class MLXChatClient:
    def __init__(self, config: ProfilesConfig, options: BackendOptions) -> None:
        self._config = config
        self._engine = engine_for(config.model, options)

    def chat(self, system: str, user: str) -> ChatResult:
        engine = self._engine
        tokens = engine.encode(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        if self._config.seed is not None:
            mx.random.seed(self._config.seed)
        sampler = make_sampler(temp=self._config.temperature, top_p=self._config.top_p)
        generated = engine.generate_tokens(tokens, self._config.max_output_tokens, sampler=sampler)
        if len(generated) >= self._config.max_output_tokens:
            raise RuntimeError(
                f"generation reached max_output_tokens={self._config.max_output_tokens}"
            )
        content = engine.decode(generated)
        if "<think>" in content and "</think>" not in content:
            raise RuntimeError("response contains an unterminated think block")
        return ChatResult(
            text=THINK_PATTERN.sub("", content).strip(),
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
