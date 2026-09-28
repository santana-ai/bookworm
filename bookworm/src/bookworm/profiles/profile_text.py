"""Profile text on the demo pages: sections and claims, actor slugs and trimmed passages."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

UNTITLED_SECTION = "Sem título"
DEFAULT_SLUG = "ator"
PASSAGE_MAX_CHARS = 480
PASSAGE_ELLIPSIS = " […]"
PASSAGE_TRAILING_CHARACTERS = " ,;:"
SECTION_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
BULLET_PATTERN = re.compile(r"^\s{0,1}[-*]\s+(.+?)\s*$")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ProfileSection:
    title: str
    claims: tuple[str, ...]


def add_profile_line(sections: list[tuple[str, list[str]]], raw: str) -> None:
    heading = SECTION_PATTERN.match(raw.strip())
    if heading is not None:
        sections.append((heading.group(1).strip(), []))
        return
    if not sections:
        sections.append((UNTITLED_SECTION, []))
    claims = sections[-1][1]
    bullet = BULLET_PATTERN.match(raw)
    if bullet is not None:
        claims.append(bullet.group(1))
    elif not claims:
        claims.append(raw.strip())
    else:
        claims[-1] = f"{claims[-1]} {raw.strip()}"


def parse_profile(text: str) -> list[ProfileSection]:
    """Markdown sections of a profile, each bullet a claim; a wrapped line joins its bullet."""
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        if raw.strip():
            add_profile_line(sections, raw)
    return [ProfileSection(title, tuple(claims)) for title, claims in sections if claims]


def actor_slug(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return SLUG_PATTERN.sub("-", folded).strip("-") or DEFAULT_SLUG


def assign_slugs(names: Iterable[str]) -> dict[str, str]:
    """A distinct slug per name, numbered from ``-2`` on collisions, in name order."""
    slugs: dict[str, str] = {}
    taken: set[str] = set()
    for name in sorted(set(names)):
        base = actor_slug(name)
        slug = base
        suffix = 2
        while slug in taken:
            slug = f"{base}-{suffix}"
            suffix += 1
        taken.add(slug)
        slugs[name] = slug
    return slugs


def trim_passage(text: str, max_chars: int = PASSAGE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    room = text[:max_chars]
    boundary = room.rfind(" ")
    kept = room[:boundary] if boundary > max_chars // 2 else room
    return kept.rstrip(PASSAGE_TRAILING_CHARACTERS) + PASSAGE_ELLIPSIS
