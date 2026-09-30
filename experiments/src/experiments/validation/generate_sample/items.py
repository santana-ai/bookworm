"""The rows the annotator judges: an evidence passage with its context, or a speaker check."""

from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.transcript import (
    TURN_HEADER_PATTERN,
    extract_quotes,
    normalize_whitespace,
    quote_prefix_pattern,
    quote_prefixes,
    split_into_turns,
)
from experiments.validation.generate_sample.config import Stratum
from experiments.validation.generate_sample.draw import record_order

Record = dict[str, Any]

EVIDENCE_QUESTION = "trecho_sustenta"
SPEAKER_QUESTION = "pessoa_falou"
MIN_ITEM_ID_DIGITS = 3
TURN_START_MARK = "[início do turno]"
TURN_END_MARK = "[fim do turno]"
ELLIPSIS = "…"
NO_OFFSETS_LINK = "sem localização na transcrição; procure o trecho no arquivo da audiência"
SPEAKER_LINK = "sem trecho; consulte a transcrição inteira da audiência"


def header_text(match: Any) -> str:
    return normalize_whitespace(match.group(0)).rstrip(" -")


def hearing_view(hearing: Record) -> Record:
    transcript = hearing["transcricao"]
    matches = list(TURN_HEADER_PATTERN.finditer(transcript))
    return {
        "id": hearing["id"],
        "transcript": transcript,
        "turns": split_into_turns(transcript),
        "headers": [header_text(match) for match in matches],
        "opening_end": matches[0].start() if matches else len(transcript),
    }


def format_offset(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def context_window(
    transcript: str, turn: Record, start: int, end: int, chars: int
) -> tuple[str, str]:
    """Up to ``chars`` characters of the turn on each side of the passage, cut at a word."""
    left = max(turn["start_char"], start - chars)
    right = min(turn["end_char"], end + chars)
    before = normalize_whitespace(transcript[left:start])
    after = normalize_whitespace(transcript[end:right])
    if left > turn["start_char"]:
        before = f"{ELLIPSIS} " + before.partition(" ")[2]
    else:
        before = f"{TURN_START_MARK} {before}".strip()
    if right < turn["end_char"]:
        after = after.rpartition(" ")[0] + f" {ELLIPSIS}"
    else:
        after = f"{after} {TURN_END_MARK}".strip()
    return before, after


def quote_cue_in_trecho(statement: str, passage: str) -> bool:
    return any(
        quote_prefix_pattern(prefix).search(passage)
        for quote in extract_quotes(statement)
        for prefix, _ in quote_prefixes(quote)
    )


def evidence_location(record: Record, view: Record, chars: int) -> tuple[str, str, str]:
    """The context before and after the evidence and the link that locates it."""
    evidence = record["evidence"]
    start, end, turn_index = evidence["start_char"], evidence["end_char"], evidence["speaker_turn"]
    if start is None:
        return "", "", NO_OFFSETS_LINK
    transcript = view["transcript"]
    if normalize_whitespace(transcript[start:end]) != evidence["text"]:
        raise SystemExit(f"{record['id']}: offsets do not reproduce the evidence text")
    turn = view["turns"][turn_index]
    if not turn["start_char"] <= start < end <= turn["end_char"]:
        raise SystemExit(f"{record['id']}: evidence span is outside turn {turn_index}")
    before, after = context_window(transcript, turn, start, end, chars)
    link = (
        f"caracteres {format_offset(start)}–{format_offset(end)} do turno {turn_index}; "
        f"cabeçalho do turno: {view['headers'][turn_index]}"
    )
    return before, after, link


def display_row(
    hearing_id: int, question: str, actor: Record, statement: str, passage: str
) -> Record:
    return {
        "hearing_id": str(hearing_id),
        "pergunta": question,
        "participante": actor["name"],
        "cargo": actor["role"] or "",
        "afirmacao": statement,
        "trecho": passage,
    }


def evidence_item(record: Record, view: Record, stratum: Stratum, chars: int) -> Record:
    evidence = record["evidence"]
    before, after, link = evidence_location(record, view, chars)
    display = display_row(
        record["hearing_id"],
        stratum.question,
        record["actor"],
        record["proposition"],
        evidence["text"],
    )
    return {
        "question": stratum.question,
        "stratum": stratum.name,
        "udv_ids": [record["id"]],
        "tier": record["tier"],
        "support_type": evidence["support_type"],
        "score": evidence["score"],
        "quote_prefix": evidence["quote_prefix"],
        "quote_cue_in_trecho": quote_cue_in_trecho(record["proposition"], evidence["text"]),
        "speaker_turn": evidence["speaker_turn"],
        "start_char": evidence["start_char"],
        "end_char": evidence["end_char"],
        "display": {
            **display,
            "contexto_antes": before,
            "contexto_depois": after,
            "link": link,
        },
    }


def speaker_directory(view: Record) -> str:
    counts: dict[str, int] = {}
    for header in view["headers"]:
        counts[header] = counts.get(header, 0) + 1
    entries = "; ".join(f"{header} [{count}]" for header, count in counts.items())
    return f"Cabeçalhos de fala detectados na transcrição [número de turnos]: {entries}"


def joined_statements(members: list[Record]) -> str:
    if len(members) == 1:
        return str(members[0]["proposition"])
    return " ".join(
        f"[{number}] {member['proposition']}" for number, member in enumerate(members, 1)
    )


def speaker_item(
    hearing_id: int, person_index: int, members: list[Record], view: Record, stratum: Stratum
) -> Record:
    tiers = {member["tier"] for member in members}
    if len(tiers) != 1 or any(member["evidence"] is not None for member in members):
        raise SystemExit(f"hearing {hearing_id}, person {person_index}: mixed speaker records")
    display = display_row(
        hearing_id, stratum.question, members[0]["actor"], joined_statements(members), ""
    )
    return {
        "question": stratum.question,
        "stratum": stratum.name,
        "udv_ids": [member["id"] for member in members],
        "tier": tiers.pop(),
        "support_type": None,
        "score": None,
        "quote_prefix": None,
        "quote_cue_in_trecho": None,
        "person_index": person_index,
        "display": {
            **display,
            "contexto_antes": speaker_directory(view),
            "contexto_depois": "",
            "link": SPEAKER_LINK,
        },
    }


def speaker_items(
    records: list[Record], views: dict[int, Record], stratum: Stratum
) -> list[Record]:
    """One row per drawn person: every drawn UDV of that person is judged together."""
    groups: dict[tuple[int, int], list[Record]] = {}
    for record in records:
        hearing_id, person_index, _ = record_order(record)
        groups.setdefault((hearing_id, person_index), []).append(record)
    return [
        speaker_item(hearing_id, person_index, members, views[hearing_id], stratum)
        for (hearing_id, person_index), members in groups.items()
    ]


def build_items(
    drawn: dict[str, list[Record]],
    strata: tuple[Stratum, ...],
    views: dict[int, Record],
    chars: int,
) -> list[Record]:
    items: list[Record] = []
    for stratum in strata:
        records = drawn[stratum.name]
        if stratum.question == SPEAKER_QUESTION:
            items.extend(speaker_items(records, views, stratum))
        else:
            items.extend(
                evidence_item(record, views[record["hearing_id"]], stratum, chars)
                for record in records
            )
    return items


def assign_item_ids(
    items: list[Record], rng: np.random.Generator, prefix: str
) -> dict[str, Record]:
    """Shuffle the items and number them in the new order, so an id says nothing of a stratum."""
    order = rng.permutation(len(items))
    width = max(MIN_ITEM_ID_DIGITS, len(str(len(items))))
    return {
        f"{prefix}{position:0{width}d}": items[int(index)]
        for position, index in enumerate(order, start=1)
    }


def render_transcript(view: Record) -> str:
    lines = [
        f"Audiência {view['id']}: transcrição completa.",
        "Cada turno começa com [turno N | caracteres início–fim] seguido do cabeçalho de fala.",
        "",
    ]
    opening = view["transcript"][: view["opening_end"]].strip()
    if opening:
        lines += ["[antes do primeiro cabeçalho de fala]", opening, ""]
    for turn, header in zip(view["turns"], view["headers"], strict=True):
        span = f"{format_offset(turn['start_char'])}–{format_offset(turn['end_char'])}"
        lines += [f"[turno {turn['turn_index']} | caracteres {span}] {header}", turn["speech"], ""]
    return "\n".join(lines)


def write_transcripts(views: dict[int, Record], directory: Path) -> Record:
    directory.mkdir(parents=True, exist_ok=True)
    for hearing_id, view in sorted(views.items()):
        (directory / f"{hearing_id}.txt").write_text(render_transcript(view), encoding="utf-8")
    return {"dir": str(directory), "files": len(views), "hearing_ids": sorted(views)}
