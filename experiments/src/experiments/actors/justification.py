"""Check of the justification the model writes for a simulated speech: every sentence must be
backed by cited profile items, examples or excerpts, with a literal copy of each cited text."""

import json
from typing import Any, cast

from experiments.actors.simulation import Material, profile_ids, turn_sentences
from experiments.common.transcript import (
    TRUSTED_PREFIX_WORDS,
    find_quote_match,
    is_trusted_quote,
    normalize_whitespace,
)

Record = dict[str, Any]

QUOTE_MARKS = '"“”'
TRAILING_PUNCTUATION = ".!?…;:,\"'”) "
CHECK_RULE = (
    "JSON objects are read anywhere in the justification (one per line is asked, but objects"
    " spread over several lines or inside code fences are read too; fragments that do not parse"
    " are counted in parse_failures). A sentence of the simulated speech is grounded when at"
    " least one object's 'frase' contains it (whitespace-normalized, ignoring trailing"
    " punctuation and leading quote marks) and every such object has a non-empty 'apoio', only"
    " ids that exist in the material of the call, and, for each cited example or excerpt, a"
    " 'copias' entry that,"
    " without enclosing quote marks, has at least 6 words and whose first 6 to 10 words are"
    " found in it by the UDV quote-prefix rule (otherwise copy_missing, copy_too_short or"
    " copy_not_found). This shows that the cited item exists and that the start of the copy is"
    " literal, not that the item supports the sentence."
)


def copy_problem(copy: Any, source_id: str, source: str) -> str | None:
    if not isinstance(copy, str):
        return f"copy_missing:{source_id}"
    text = normalize_whitespace(copy).strip(QUOTE_MARKS).strip()
    if len(text.split()) < TRUSTED_PREFIX_WORDS:
        return f"copy_too_short:{source_id}"
    if not is_trusted_quote(find_quote_match(text, source)):
        return f"copy_not_found:{source_id}"
    return None


def check_line(item: Record, known_ids: set[str], sources: dict[str, str]) -> Record:
    """The problems of one justification object: missing support, unknown ids, bad copies."""
    support = item.get("apoio")
    support = (
        [value for value in support if isinstance(value, str)] if isinstance(support, list) else []
    )
    copies = cast(Record, item.get("copias")) if isinstance(item.get("copias"), dict) else {}
    problems = [] if support else ["empty_support"]
    for source_id in support:
        if source_id not in known_ids:
            problems.append(f"unknown_id:{source_id}")
        elif source_id in sources:
            problem = copy_problem(copies.get(source_id), source_id, sources[source_id])
            if problem is not None:
                problems.append(problem)
    return {
        "frase": normalize_whitespace(item["frase"]),
        "apoio": support,
        "copias": copies,
        "problems": problems,
    }


def json_objects(text: str) -> tuple[list[Any], int]:
    """Every JSON value that starts at a '{' of the text, and the count of those that fail."""
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    failures = 0
    position = 0
    while (start := text.find("{", position)) != -1:
        try:
            item, position = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            failures += 1
            position = start + 1
            continue
        objects.append(item)
    return objects, failures


def sentence_key(text: str) -> str:
    return normalize_whitespace(text).rstrip(TRAILING_PUNCTUATION).lstrip(QUOTE_MARKS)


def cited_sources(material: Material) -> dict[str, str]:
    """The texts that a justification can cite by id: examples E1.. and excerpts T1.."""
    sources = {f"E{i}": example.text for i, example in enumerate(material.examples, start=1)}
    sources |= {f"T{i}": excerpt.text for i, excerpt in enumerate(material.excerpts, start=1)}
    return sources


def check_justification(speech: str, justification: str, material: Material) -> Record:
    sources = cited_sources(material)
    known_ids = profile_ids(material.profile) | set(sources)
    objects, parse_failures = json_objects(justification)
    lines: list[Record] = []
    invalid: list[Any] = []
    for item in objects:
        if isinstance(item, dict) and isinstance(item.get("frase"), str) and item["frase"].strip():
            lines.append(check_line(item, known_ids, sources))
        else:
            invalid.append(item)
    sentences = []
    for sentence in turn_sentences(speech):
        key = sentence_key(sentence)
        matched = [line for line in lines if key and key in line["frase"]]
        sentences.append(
            {
                "sentence": sentence,
                "lines": len(matched),
                "grounded": bool(matched) and all(not line["problems"] for line in matched),
            }
        )
    speech_text = normalize_whitespace(speech)
    return {
        "sentences": sentences,
        "lines": lines,
        "invalid_items": invalid,
        "parse_failures": parse_failures,
        "lines_not_in_speech": sum(
            sentence_key(line["frase"]) not in speech_text for line in lines
        ),
        "n_sentences": len(sentences),
        "n_ungrounded": sum(not sentence["grounded"] for sentence in sentences),
    }
