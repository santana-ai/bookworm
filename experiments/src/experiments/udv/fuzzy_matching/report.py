"""The experiment report: definitions, per-group quote and name summaries, provenance."""

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bookworm.data.io
import numpy as np
import rapidfuzz

from experiments.common import reporting, splits, udv_run
from experiments.common.provenance import code_section
from experiments.common.reporting import utc_timestamp
from experiments.common.transcript import CODE_SOURCES, TRUSTED_PREFIX_WORDS
from experiments.udv.fuzzy_matching.config import QUOTE_METHODS, FuzzyConfig
from experiments.udv.fuzzy_matching.name_summary import summarize_name_group
from experiments.udv.fuzzy_matching.quote_summary import summarize_quote_group
from experiments.udv.fuzzy_matching.quotes import QuoteCollection
from experiments.validation import generate_sample

Record = dict[str, Any]

PACKAGE_DIR = Path(__file__).resolve().parent
QUESTION = (
    "would approximate string matching recover quotes and participant names that the exact "
    "rules miss, and at what risk"
)
PRINTED_THRESHOLDS = (90.0, 100.0)
STATIC_DEFINITIONS = {
    "gain": (
        "a population opinion whose fuzzy score reaches the threshold at the first prefix level, "
        "in configured order, that reaches it; the sentence is the one enclosing the aligned span "
        "in that turn"
    ),
    "agrees_with_encoder": (
        "sentences_agree between the fuzzy sentence and the reference run's top-1 sentence, only "
        "for records whose reference tier is semantic; it is a weak precision proxy, because the "
        "encoder embeds the full opinion, quote included, so both signals see the quote text and "
        "are not independent"
    ),
    "null": (
        "quotes that should not occur in the person's speech, aligned against the same person's "
        "turns with the same scorer; the rate of draws at or above the threshold estimates how "
        "often a threshold is reached by chance; same_hearing_other_person is topic-matched and "
        "may include genuine repetitions"
    ),
    "expected_chance_matches": "null rate times the population opinions with an eligible quote",
    "positive_control": (
        "opinions with a trusted exact prefix, the exact prefix aligned with the same scorer; "
        "score 100 and agreement with the exact sentence are expected"
    ),
    "encoder_baseline": (
        "on the positive control, how often the production encoder's top-1 sentence, recomputed "
        "from the cached embeddings of the production run (no encoding here), agrees with the "
        "sentence of the trusted exact quote; it is the agreement rate the proxy reaches on quote "
        "sentences known by exact matching, so an agreement rate among fuzzy gains is read "
        "against it, not against 1"
    ),
    "elsewhere": (
        "the same prefixes aligned against every turn of the hearing not matched to the person, "
        "at the accepted level; a gain whose quote scores higher elsewhere may belong to another "
        "speaker (misattribution, a quote read aloud, or the person quoting someone)"
    ),
    "encoder_cache_consistency": (
        "for population opinions, whether the cached top-1 equals the reference run's evidence; "
        "with the production run as reference every compared pair must match"
    ),
    "bands": (
        "opinions accepted at a threshold and not at the next one, with the sentence of the lower "
        "threshold"
    ),
    "names_population": "participants with no matched turn under resolve_person_speech",
    "names_status": (
        "unique: best at or above the threshold and the best disjoint rival below it; ambiguous: "
        "a disjoint rival also at or above it; already_assigned: the best speaker shares turns "
        "with a participant the exact rules resolved; metrics_disagree (rule both): the two "
        "metrics pick different speakers; contested: two unresolved participants of one hearing "
        "would take the same turns"
    ),
    "impostor_control": (
        "participants the exact rules resolved, scored against the hearing's speakers that share "
        "no turn with their own; the rate at or above the threshold estimates how often a name "
        "reaches it on a different speaker"
    ),
    "quote_support": (
        "opinions of the participant whose quote has an exact prefix match "
        "(find_opinion_turn_quote_match) inside the candidate speaker's turns"
    ),
}


@dataclass(frozen=True)
class RunSources:
    run_name: str
    splits: tuple[str, ...]
    final_test: bool
    split_source: Record
    reference_source: Record
    encoder_source: Record


def definitions(config: FuzzyConfig) -> Record:
    quote_population = (
        "opinions of people with at least one matched turn, with at least one quote extracted by "
        "extract_quotes (bookworm.udv.quotes), whose exact match (find_opinion_turn_quote_match) "
        f"has no trusted prefix of {TRUSTED_PREFIX_WORDS}+ words"
    )
    eligible_quote = f"an extracted quote with {config.min_prefix_words}+ whitespace words"
    bootstrap = (
        f"{config.bootstrap_samples} resamples of whole hearings, "
        f"{config.confidence_level:.0%} percentile interval, fixed seed"
    )
    return {
        "quote_population": quote_population,
        "eligible_quote": eligible_quote,
        **STATIC_DEFINITIONS,
        "bootstrap": bootstrap,
    }


def group_members(splits_used: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {name: (name,) for name in splits_used} | {"all": splits_used}


def in_group(entries: list[Record], members: tuple[str, ...]) -> list[Record]:
    return [entry for entry in entries if entry["split"] in members]


def quote_groups(
    quotes: QuoteCollection, groups: dict[str, tuple[str, ...]], config: FuzzyConfig
) -> Record:
    return {
        name: summarize_quote_group(
            in_group(quotes.rows, members),
            in_group(quotes.funnel, members),
            in_group(quotes.controls, members),
            in_group(quotes.cache_status, members),
            config,
            name,
        )
        for name, members in groups.items()
    }


def name_groups(
    name_rows: list[Record],
    impostors: list[Record],
    cache_status: list[Record],
    groups: dict[str, tuple[str, ...]],
    config: FuzzyConfig,
) -> Record:
    return {
        name: summarize_name_group(
            in_group(name_rows, members),
            in_group(impostors, members),
            sorted(entry["hearing_id"] for entry in in_group(cache_status, members)),
            config,
            name,
        )
        for name, members in groups.items()
    }


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rapidfuzz": rapidfuzz.__version__,
        "platform": platform.platform(),
        "device": "cpu",
    }


def code_hashes() -> dict[str, str]:
    return code_section(
        PACKAGE_DIR,
        *CODE_SOURCES,
        udv_run,
        bookworm.data.io,
        generate_sample,
        splits,
        reporting,
    )


def build_report(
    sources: RunSources,
    quotes: QuoteCollection,
    name_rows: list[Record],
    impostors: list[Record],
    artifacts: Record,
    review: Record,
    elapsed_seconds: float,
    config: FuzzyConfig,
) -> Record:
    groups = group_members(sources.splits)
    return {
        "run_name": sources.run_name,
        "created_at": utc_timestamp(),
        "question": QUESTION,
        "splits_used": list(sources.splits),
        "final_test": sources.final_test,
        "sources": {
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "splits": sources.split_source,
            "reference_udv": sources.reference_source,
            "encoder_baseline": sources.encoder_source,
        },
        "definitions": definitions(config),
        "quotes": quote_groups(quotes, groups, config),
        "names": name_groups(name_rows, impostors, quotes.cache_status, groups, config),
        "artifacts": artifacts,
        "review": review,
        "code": code_hashes(),
        "timing": {"elapsed_seconds": round(elapsed_seconds, 1)},
        "environment": environment(),
        "config": config.source,
    }


def quote_cells(gains: Record, config: FuzzyConfig) -> list[str]:
    cells = []
    for threshold in (config.quote_thresholds[0], *PRINTED_THRESHOLDS):
        entry = gains.get(str(threshold))
        if entry is None:
            continue
        null = entry["null"]["other_hearing_same_split"]["rate"]["value"]
        cells.append(
            f"t={threshold:g}: gain={entry['opinions']} "
            f"agree={entry['agrees_with_encoder']}/{entry['with_encoder_reference']} "
            f"null={null}"
        )
    return cells


def name_cells(by_threshold: Record, config: FuzzyConfig) -> list[str]:
    cells = []
    for threshold in (config.name_thresholds[0], *PRINTED_THRESHOLDS):
        entry = by_threshold.get(str(threshold))
        if entry is not None:
            cells.append(
                f"t={threshold:g}: resolve={entry['would_resolve']} status={entry['by_status']}"
            )
    return cells


def print_quote_summary(group: str, summary: Record, config: FuzzyConfig) -> None:
    funnel = summary["funnel"]
    control = summary["positive_control"]
    cache = summary["encoder_cache_consistency"]
    print(
        f"quotes {group:10s} population={funnel['population_without_trusted_exact']} "
        f"eligible={funnel['population_with_eligible_quote']} "
        f"controls={control['opinions']} "
        f"encoder_baseline={control['encoder_baseline']['agrees']}/"
        f"{control['encoder_baseline']['with_encoder_cache']} "
        f"cached_hearings={cache['hearings_with_encoder_cache']}/{cache['hearings']}"
    )
    for method in QUOTE_METHODS:
        print(f"  {method:5s} " + " | ".join(quote_cells(summary["gains"][method], config)))


def print_summary(report: Record, config: FuzzyConfig) -> None:
    for group, summary in report["quotes"].items():
        print_quote_summary(group, summary, config)
    for group, summary in report["names"].items():
        print(
            f"names  {group:10s} unresolved={summary['unresolved_participants']} "
            f"opinions={summary['unresolved_opinions']}"
        )
        for rule, by_threshold in summary["by_rule"].items():
            print(f"  {rule:15s} " + " | ".join(name_cells(by_threshold, config)))
