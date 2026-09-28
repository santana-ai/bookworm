"""Whitespace, accent and name normalization."""

import re
import unicodedata

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def normalize_name(name: str) -> str:
    return strip_accents(name).upper().strip()


def prefix_pattern(prefix: str) -> re.Pattern[str]:
    first, rest = prefix[0], prefix[1:]
    lower, upper = first.lower(), first.upper()
    if lower == upper:
        head = rf"(?<!\w){re.escape(first)}"
    else:
        head = rf"(?:(?<!\w){re.escape(lower)}|{re.escape(upper)})"
    return re.compile(rf"{head}(?i:{re.escape(rest)})(?!\w)")
