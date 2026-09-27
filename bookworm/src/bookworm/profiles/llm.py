import re
from dataclasses import dataclass
from typing import Protocol

from bookworm.errors import ConfigError
from bookworm.profiles.config import ModelSettings

THINK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


class ChatClient(Protocol):
    def chat(self, system: str, user: str) -> ChatResult: ...


def finish_generation(
    content: str, input_tokens: int, output_tokens: int, max_output_tokens: int
) -> ChatResult:
    if output_tokens >= max_output_tokens:
        raise GenerationError(f"generation reached max_output_tokens={max_output_tokens}")
    if THINK_OPEN in content and THINK_CLOSE not in content:
        raise GenerationError("response contains an unterminated think block")
    return ChatResult(
        text=THINK_PATTERN.sub("", content).strip(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def load_transformers_client(settings: ModelSettings) -> ChatClient:
    if not settings.name:
        raise ConfigError("set [model].name in the config, or pass --model")
    try:
        from bookworm.profiles import transformers_client
    except ModuleNotFoundError as error:
        raise ConfigError(
            f"profile generation needs the optional 'profiles' extra: {error}"
        ) from error
    return transformers_client.TransformersChatClient(settings)
