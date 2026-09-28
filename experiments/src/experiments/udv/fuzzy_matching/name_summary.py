"""Summaries of the name rows: what each rule and threshold would resolve, impostor rates."""

from collections import Counter
from typing import Any

from experiments.udv.fuzzy_matching.config import COMBINED_NAME_RULE, NAME_METRICS, FuzzyConfig
from experiments.udv.fuzzy_matching.sampling import (
    HearingCounts,
    bootstrap_ratio,
    increment,
    score_histogram,
)

Record = dict[str, Any]
Decisions = list[tuple[Record, Record]]


def name_acceptance(row: Record, rule: str, threshold: float, config: FuzzyConfig) -> Record | None:
    metrics = config.name_metrics if rule == COMBINED_NAME_RULE else (rule,)
    results = [row["metrics"].get(metric) for metric in metrics]
    if any(result is None for result in results):
        return None
    if any(result["best"]["score"] < threshold for result in results):
        return None
    if len({result["best"]["speaker"] for result in results}) != 1:
        return {"status": "metrics_disagree", "result": results[0]}
    ambiguous = any(
        result["second"] is not None and result["second"]["score"] >= threshold
        for result in results
    )
    if ambiguous:
        return {"status": "ambiguous", "result": results[0]}
    if results[0]["best_assigned_to"]:
        return {"status": "already_assigned", "result": results[0]}
    return {"status": "unique", "result": results[0]}


def contested_ids(accepted: Decisions) -> set[str]:
    """Participants of one hearing whose accepted speakers would take the same turns."""
    contested: set[str] = set()
    for position, (row, decision) in enumerate(accepted):
        turns = set(decision["result"]["best"]["turn_indices"])
        for other_row, other_decision in accepted[position + 1 :]:
            if other_row["hearing_id"] != row["hearing_id"]:
                continue
            if turns & set(other_decision["result"]["best"]["turn_indices"]):
                contested.update({row["id"], other_row["id"]})
    return contested


def name_decisions(
    rows: list[Record], rule: str, threshold: float, config: FuzzyConfig
) -> Decisions:
    return [
        (row, decision)
        for row in rows
        if (decision := name_acceptance(row, rule, threshold, config)) is not None
    ]


def resolved_names(decisions: Decisions) -> tuple[Decisions, set[str]]:
    unique = [(row, decision) for row, decision in decisions if decision["status"] == "unique"]
    contested = contested_ids(unique)
    return [(row, decision) for row, decision in unique if row["id"] not in contested], contested


def quote_supported(resolved: Decisions, field: str) -> int:
    return sum(1 for _, decision in resolved if decision["result"]["best_quote_support"][field])


def names_at_threshold(
    rows: list[Record],
    impostors: list[Record],
    rule: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    decisions = name_decisions(rows, rule, threshold, config)
    resolved, contested = resolved_names(decisions)
    summary: Record = {
        "best_at_or_above": len(decisions),
        "by_status": dict(Counter(decision["status"] for _, decision in decisions)),
        "contested_among_unresolved": len(contested),
        "would_resolve": len(resolved),
        "would_resolve_opinions": sum(len(row["opinions"]) for row, _ in resolved),
        "would_resolve_with_trusted_quote_support": quote_supported(
            resolved, "with_trusted_prefix"
        ),
        "would_resolve_with_any_quote_prefix": quote_supported(resolved, "with_any_prefix"),
        "would_resolve_ids": [row["id"] for row, _ in resolved],
    }
    if rule in NAME_METRICS:
        summary["impostor_control"] = impostor_rates(
            impostors, rule, threshold, hearing_ids, config, group
        )
    return summary


def impostor_rates(
    impostors: list[Record],
    metric: str,
    threshold: float,
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    own_hits: HearingCounts = {}
    impostor_hits: HearingCounts = {}
    outranks: HearingCounts = {}
    totals: HearingCounts = {}
    for row in impostors:
        scores = row[metric]
        hearing_id = row["hearing_id"]
        increment(totals, hearing_id)
        own, other = scores["own_score"], scores["impostor_score"]
        if own is not None and own >= threshold:
            increment(own_hits, hearing_id)
        if other is not None and other >= threshold:
            increment(impostor_hits, hearing_id)
            if own is None or other >= own:
                increment(outranks, hearing_id)
    stream = f"{group}/names/{metric}/{threshold}"
    return {
        "resolved_participants": int(sum(totals.values())),
        "own_speaker_at_or_above": bootstrap_ratio(
            own_hits, totals, hearing_ids, config, f"{stream}/own"
        ),
        "impostor_at_or_above": bootstrap_ratio(
            impostor_hits, totals, hearing_ids, config, f"{stream}/other"
        ),
        "impostor_at_or_above_and_not_below_own": int(sum(outranks.values())),
    }


def top_impostors(impostors: list[Record], config: FuzzyConfig) -> Record:
    return {
        metric: [
            {
                "id": row["id"],
                "split": row["split"],
                "name": row["name"],
                "own_score": row[metric]["own_score"],
                "impostor_speaker": row[metric]["impostor_speaker"],
                "impostor_score": row[metric]["impostor_score"],
            }
            for row in sorted(
                (row for row in impostors if row[metric]["impostor_score"] is not None),
                key=lambda row: (-row[metric]["impostor_score"], row["hearing_id"], row["id"]),
            )[: config.impostor_examples]
        ]
        for metric in config.name_metrics
    }


def name_rules(config: FuzzyConfig) -> tuple[str, ...]:
    if len(config.name_metrics) > 1:
        return (*config.name_metrics, COMBINED_NAME_RULE)
    return config.name_metrics


def best_score_histograms(rows: list[Record], config: FuzzyConfig) -> Record:
    return {
        metric: score_histogram(
            [
                row["metrics"][metric]["best"]["score"]
                for row in rows
                if row["metrics"][metric] is not None
            ],
            config.score_bin_width,
        )
        for metric in config.name_metrics
    }


def summarize_name_group(
    rows: list[Record],
    impostors: list[Record],
    hearing_ids: list[int],
    config: FuzzyConfig,
    group: str,
) -> Record:
    return {
        "unresolved_participants": len(rows),
        "unresolved_opinions": sum(len(row["opinions"]) for row in rows),
        "hearings_with_unresolved": len({row["hearing_id"] for row in rows}),
        "without_any_speaker": sum(
            1 for row in rows if all(result is None for result in row["metrics"].values())
        ),
        "same_best_speaker_across_metrics": sum(
            1
            for row in rows
            if all(result is not None for result in row["metrics"].values())
            and len({result["best"]["speaker"] for result in row["metrics"].values()}) == 1
        ),
        "best_score_histograms": best_score_histograms(rows, config),
        "by_rule": {
            rule: {
                str(threshold): names_at_threshold(
                    rows, impostors, rule, threshold, hearing_ids, config, group
                )
                for threshold in config.name_thresholds
            }
            for rule in name_rules(config)
        },
        "top_impostors": top_impostors(impostors, config),
    }
