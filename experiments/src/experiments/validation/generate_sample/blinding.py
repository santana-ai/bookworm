"""The blinding check of a sheet and the record of what it hides and what can still leak."""

from typing import Any

from experiments.common.udv_run import SUPPORT_TYPES, TIERS
from experiments.validation.generate_sample.config import Stratum, ValidationConfig
from experiments.validation.generate_sample.items import EVIDENCE_QUESTION
from experiments.validation.generate_sample.sheets import CSV_COLUMNS

Record = dict[str, Any]

HIDDEN_ITEM_FIELDS = (
    "udv_ids",
    "stratum",
    "tier",
    "support_type",
    "score",
    "quote_prefix",
    "quote_cue_in_trecho",
)
ROW_ORDER_RULE = "random permutation (seed stream row_order); item ids follow the row order"
KNOWN_LEAKS = (
    "a direct_quote item can be recognized when the afirmacao holds a quotation whose "
    "words reappear in trecho",
    "a semantic_with_short_quote item is recognizable in the same way: this support type "
    "is assigned only when a quote prefix of the afirmacao is found in the sentence that "
    "becomes trecho, so the quoted words reappear in trecho; semantic_similarity items "
    "(the semantic_match_high and semantic_match_weak strata) show the cue only by "
    "coincidence, so only high against weak stays blind",
    "pessoa_falou rows are recognizable by their question, which is asked only of the "
    "speaker_check stratum",
)
QUOTE_CUE_DEFINITION = (
    "quote_cue_in_trecho is true when a quotation extracted from afirmacao "
    "(bookworm.udv.quotes.extract_quotes) has a prefix of 10, 6, 4 or 3 words "
    "(bookworm.udv.quotes.quote_prefixes) that bookworm.udv.quotes.quote_prefix_pattern finds "
    "in trecho; it is stored per item in the key only and the precision report splits each "
    "evidence stratum by it"
)


def forbidden_token_hits(rows: list[Record], strata: tuple[Stratum, ...]) -> list[str]:
    tokens = {*TIERS, *SUPPORT_TYPES, *(stratum.name for stratum in strata)}
    return [
        f"{row['item_id']}.{column}"
        for row in rows
        for column in CSV_COLUMNS
        if any(token in row[column] for token in tokens)
    ]


def quote_cue_counts(items: dict[str, Record], strata: tuple[Stratum, ...]) -> Record:
    counts: Record = {}
    for stratum in strata:
        if stratum.question != EVIDENCE_QUESTION:
            continue
        flags = [
            bool(item["quote_cue_in_trecho"])
            for item in items.values()
            if item["stratum"] == stratum.name
        ]
        counts[stratum.name] = {"rows": len(flags), "with_cue": sum(flags)}
    return counts


def blinding_section(
    rows: list[Record], items: dict[str, Record], config: ValidationConfig
) -> Record:
    """Refuse a sheet that shows a tier, support type or stratum name, and describe its blinding."""
    hits = forbidden_token_hits(rows, config.strata)
    if hits:
        raise SystemExit(f"blinding check failed, tier or stratum names in the sheet: {hits[:10]}")
    return {
        "csv_columns": list(CSV_COLUMNS),
        "hidden_in_key_only": list(HIDDEN_ITEM_FIELDS),
        "row_order": ROW_ORDER_RULE,
        "forbidden_token_hits": 0,
        "known_leaks": list(KNOWN_LEAKS),
        "quote_cue": {
            "definition": QUOTE_CUE_DEFINITION,
            "rows_by_stratum": quote_cue_counts(items, config.strata),
        },
    }
