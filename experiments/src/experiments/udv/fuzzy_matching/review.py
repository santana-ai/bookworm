"""The blind review sheets: items shown to the reviewer, the hidden key and its summary."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file

from experiments.common.reporting import file_record, utc_timestamp
from experiments.common.transcript import TURN_HEADER_PATTERN, split_into_turns
from experiments.udv.fuzzy_matching.config import QUOTE_METHODS, FuzzyConfig
from experiments.udv.fuzzy_matching.sampling import draw_rng, sorted_counts
from experiments.validation.generate_sample import (
    assign_item_ids,
    canonical_sha256,
    context_window,
    header_text,
)

Record = dict[str, Any]

DISPLAY_FIELDS = {
    "quotes": (
        "hearing_id",
        "participant",
        "role",
        "opinion",
        "quotes",
        "turn_header",
        "context_before",
        "candidate",
        "context_after",
    ),
    "names": (
        "hearing_id",
        "participant",
        "role",
        "opinions",
        "candidate_speaker",
        "party_info",
        "turn_count",
        "speech_excerpt",
    ),
}
FILL_FIELDS = ("judgment", "note")
SPEAKER_RANKS = ("best", "second")


@dataclass(frozen=True)
class HearingText:
    transcript: str
    turns: dict[int, Record]
    headers: tuple[str, ...]


def hearing_text(hearing: Record) -> HearingText:
    transcript = hearing["transcricao"]
    return HearingText(
        transcript=transcript,
        turns={turn["turn_index"]: turn for turn in split_into_turns(transcript)},
        headers=tuple(header_text(match) for match in TURN_HEADER_PATTERN.finditer(transcript)),
    )


def quote_candidate_key(result: Record) -> tuple[Any, ...]:
    return (
        result["turn_index"],
        result["sentence_start_char"],
        result["sentence_end_char"],
        result["sentence"] or result["aligned_text"],
    )


def quote_display(
    row: Record, result: Record, text: HearingText, chars: int
) -> tuple[Record, bool]:
    start, end = result["sentence_start_char"], result["sentence_end_char"]
    located = start is not None and end is not None and bool(result["sentence"])
    before, after = (
        context_window(text.transcript, text.turns[result["turn_index"]], start, end, chars)
        if located
        else ("", "")
    )
    display = {
        "hearing_id": row["hearing_id"],
        "participant": row["person"]["name"],
        "role": row["person"]["role"],
        "opinion": row["opinion"],
        "quotes": row["quotes"],
        "turn_header": text.headers[result["turn_index"]],
        "context_before": before,
        "candidate": result["sentence"] or result["aligned_text"],
        "context_after": after,
    }
    return display, located


def quote_proposal(row: Record, result: Record, method: str, level: int) -> Record:
    return {
        "method": method,
        "level": level,
        "score": result["score"],
        "quote_index": result["quote_index"],
        "prefix": result["prefix"],
        "aligned_text": result["aligned_text"],
        "agrees_with_encoder": result["agrees_with_encoder"],
        "best_elsewhere": row["elsewhere"][method][str(level)],
    }


def quote_candidates(row: Record, config: FuzzyConfig) -> dict[tuple[Any, ...], Record]:
    """The distinct candidate passages of one row, each with the proposals that point at it."""
    candidates: dict[tuple[Any, ...], Record] = {}
    for method in QUOTE_METHODS:
        for level in config.prefix_levels:
            result = row["fuzzy"][method][str(level)]
            if result is None or result["score"] < config.quote_review_min_score:
                continue
            entry = candidates.setdefault(
                quote_candidate_key(result), {"result": result, "proposals": []}
            )
            entry["proposals"].append(quote_proposal(row, result, method, level))
    return candidates


def quote_item(row: Record, entry: Record, text: HearingText, config: FuzzyConfig) -> Record:
    result = entry["result"]
    reference = row["reference"] or {}
    display, located = quote_display(row, result, text, config.review_context_chars)
    return {
        "display": display,
        "hidden": {
            "row_id": row["id"],
            "split": row["split"],
            "hearing_id": row["hearing_id"],
            "turn_index": result["turn_index"],
            "sentence": result["sentence"],
            "sentence_start_char": result["sentence_start_char"],
            "sentence_end_char": result["sentence_end_char"],
            "context_located": located,
            "proposals": entry["proposals"],
            "exact_match": row["exact_match"],
            "encoder_sentence": reference.get("text"),
            "encoder_tier": reference.get("tier"),
            "encoder_support_type": reference.get("support_type"),
        },
    }


def quote_review_items(
    rows: list[Record], texts: dict[int, HearingText], config: FuzzyConfig
) -> list[Record]:
    return [
        quote_item(row, entry, texts[row["hearing_id"]], config)
        for row in rows
        for entry in quote_candidates(row, config).values()
    ]


def speaker_candidates(results: dict[str, Record]) -> dict[str, Record]:
    """The distinct best and second speakers of one row, with each metric's view of them."""
    candidates: dict[str, Record] = {}
    for metric, result in results.items():
        for rank in SPEAKER_RANKS:
            entry = result[rank]
            if entry is None:
                continue
            candidate = candidates.setdefault(entry["speaker"], {"entry": entry, "metrics": {}})
            best_only = (
                {
                    "assigned_to": result["best_assigned_to"],
                    "quote_support": result["best_quote_support"],
                }
                if rank == "best"
                else {}
            )
            candidate["metrics"][metric] = {"rank": rank, "score": entry["score"], **best_only}
    return candidates


def name_item(row: Record, speaker: str, candidate: Record) -> Record:
    entry = candidate["entry"]
    return {
        "display": {
            "hearing_id": row["hearing_id"],
            "participant": row["name"],
            "role": row["role"],
            "opinions": row["opinions"],
            "candidate_speaker": entry["display"],
            "party_info": entry["party_info"],
            "turn_count": entry["turn_count"],
            "speech_excerpt": entry["speech_excerpt"],
        },
        "hidden": {
            "row_id": row["id"],
            "split": row["split"],
            "hearing_id": row["hearing_id"],
            "speaker": speaker,
            "turn_indices": entry["turn_indices"],
            "metrics": candidate["metrics"],
        },
    }


def name_review_items(rows: list[Record], config: FuzzyConfig) -> list[Record]:
    items: list[Record] = []
    for row in rows:
        results = {metric: result for metric, result in row["metrics"].items() if result}
        if (
            not results
            or max(result["best"]["score"] for result in results.values())
            < config.name_review_min_score
        ):
            continue
        items.extend(
            name_item(row, speaker, candidate)
            for speaker, candidate in speaker_candidates(results).items()
        )
    return items


def sheet_fields(kind: str) -> list[str]:
    return ["item_id", *DISPLAY_FIELDS[kind], *FILL_FIELDS]


def blind_sheet(kind: str, items: list[Record], config: FuzzyConfig) -> dict[str, Record]:
    """The items in a seeded random order under new ids, checked against the sheet design."""
    labelled = assign_item_ids(
        items,
        draw_rng(config.seed, *config.review_order_streams[kind]),
        config.review_item_prefixes[kind],
    )
    for item_id, item in labelled.items():
        if list(item["display"]) != list(DISPLAY_FIELDS[kind]):
            raise SystemExit(f"{kind} review item {item_id} shows fields outside the sheet design")
    return labelled


def sheet_rows(kind: str, labelled: dict[str, Record]) -> list[Record]:
    rows = [
        {"item_id": item_id, **item["display"], **{name: None for name in FILL_FIELDS}}
        for item_id, item in labelled.items()
    ]
    if any(list(row) != sheet_fields(kind) for row in rows):
        raise SystemExit(f"the {kind} review sheet has fields outside its design")
    return rows


def refuse_filled_sheet(path: Path) -> None:
    if not path.exists():
        return
    filled = [row.get("item_id") for row in load_jsonl(path) if row.get("judgment") is not None]
    if filled:
        raise SystemExit(
            f"{path} already holds {len(filled)} judgments (e.g. {filled[:3]}); a filled review "
            "sheet is never overwritten, so nothing was written"
        )


def review_key(
    kind: str,
    labelled: dict[str, Record],
    run_name: str,
    splits: tuple[str, ...],
    final_test: bool,
    rows_path: Path,
    sheet_path: Path,
    config: FuzzyConfig,
) -> Record:
    return {
        "role": f"fuzzy_{kind}_review_key",
        "run_name": run_name,
        "created_at": utc_timestamp(),
        "splits_used": list(splits),
        "final_test": final_test,
        "rows_file": file_record(rows_path),
        "sheet": {
            "path": str(sheet_path),
            "rows": len(labelled),
            "sha256_at_creation": sha256_of_file(sheet_path),
            "fields": sheet_fields(kind),
        },
        "display_fields": list(DISPLAY_FIELDS[kind]),
        "fill_fields": list(FILL_FIELDS),
        "labels": list(config.review_labels[kind]),
        "positive_label": config.review_positive[kind],
        "unsure_label": config.review_unsure[kind],
        "items": {
            item_id: {**item["hidden"], "display_sha256": canonical_sha256(item["display"])}
            for item_id, item in labelled.items()
        },
    }


def items_per_row(items: list[Record]) -> dict[int, int]:
    return sorted_counts(list(Counter(item["row_id"] for item in items).values()))


def review_summary(kind: str, key: Record, key_path: Path) -> Record:
    items = list(key["items"].values())
    summary: Record = {
        "sheet": {
            "path": key["sheet"]["path"],
            "rows": key["sheet"]["rows"],
            "sha256": key["sheet"]["sha256_at_creation"],
        },
        "key": file_record(key_path),
        "shown_fields": key["display_fields"],
        "fill_fields": key["fill_fields"],
        "hidden_fields": sorted({name for item in items for name in item} - {"display_sha256"}),
        "labels": key["labels"],
        "items_by_split": sorted_counts([item["split"] for item in items]),
        "distinct_rows": len({item["row_id"] for item in items}),
    }
    if kind == "quotes":
        summary["proposals"] = sum(len(item["proposals"]) for item in items)
        summary["items_without_context"] = sum(1 for item in items if not item["context_located"])
        summary["items_per_opinion"] = items_per_row(items)
    else:
        summary["items_per_participant"] = items_per_row(items)
    return summary
