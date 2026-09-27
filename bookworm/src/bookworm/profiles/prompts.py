import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template, TemplateError

from bookworm.errors import ConfigError

PROMPT_VERSION_LENGTH = 12


@dataclass(frozen=True)
class PromptTurn:
    role: str
    text: str


@dataclass(frozen=True)
class PromptHearing:
    date: str
    assunto: str
    turns: tuple[PromptTurn, ...]


@dataclass(frozen=True)
class PromptSet:
    version: str
    system_profile: str
    user_profile: Template

    def render_user(self, actor_label: str, hearings: Sequence[PromptHearing]) -> str:
        return self.user_profile.render(actor_label=actor_label, hearings=list(hearings))


def prompt_version(prompts_dir: Path, file_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(file_names):
        digest.update(name.encode())
        digest.update((prompts_dir / name).read_bytes())
    return digest.hexdigest()[:PROMPT_VERSION_LENGTH]


def prompt_environment(prompts_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(prompts_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )


def load_prompts(prompts_dir: Path, system_file: str, user_file: str) -> PromptSet:
    for name in (system_file, user_file):
        if not (prompts_dir / name).is_file():
            raise ConfigError(f"{prompts_dir / name}: prompt file not found")
    try:
        user_profile = prompt_environment(prompts_dir).get_template(user_file)
    except TemplateError as error:
        raise ConfigError(f"{prompts_dir / user_file}: invalid template: {error}") from error
    return PromptSet(
        version=prompt_version(prompts_dir, (system_file, user_file)),
        system_profile=(prompts_dir / system_file).read_text(encoding="utf-8"),
        user_profile=user_profile,
    )
