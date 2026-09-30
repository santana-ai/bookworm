"""Metrics with hearing-bootstrap intervals and the paired tests of E1 and E2 over query rows."""

from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl

from experiments.common.reporting import file_record, rounded_or_none
from experiments.common.stats import bootstrap_mean, holm, mcnemar_exact, sign_flip_test, stream_rng
from experiments.retrieval.data import UNIT_KINDS

Record = dict[str, Any]
RowGroups = dict[tuple[str, str], list[Record]]
Pair = tuple[Record, Record]

CHARS_DECIMALS = 1
COUNT_DECIMALS = 2
INTERVAL_KEYS = ("point", "low", "high")
DEFINITIONS: Record = {
    "rank": "1-based position of the first relevant unit in the retriever's order",
    "rank_optimistic": "1 + number of non-relevant units scored strictly above the best relevant",
    "rank_pessimistic": "1 + number of non-relevant units scored at or above the best relevant",
    "acc_at_k": "share of queries whose rank is at most k",
    "mrr": "mean of 1 / rank",
    "random_acc_at_1": "mean over queries of n_relevant / n_units, the acc@1 of a uniform pick",
    "acc_at_1_minus_random": "acc@1 minus random_acc_at_1, comparable across unit sizes",
    "top1_chars": "characters of the top-1 unit, the evidence the retriever would hand over",
    "ci": "percentile interval over bootstrap replicates that resample whole hearings",
    "mcnemar": (
        "exact two-sided binomial test on the queries where exactly one of the baseline and "
        "the other retriever has the relevant unit at rank 1; it treats queries as independent, "
        "so the hearing-bootstrap interval of the acc@1 difference is reported next to it"
    ),
    "mrr_test": "two-sided sign-flip permutation test that flips whole hearings",
    "lift": "per query: 1 if the rank-1 unit is relevant, else 0, minus n_relevant / n_units",
    "retrievers_within_unit": (
        "E1: each retriever against evaluation.baseline_retriever on the same unit kind and the "
        "same queries (exact McNemar on acc@1, hearing sign-flip on the reciprocal rank); "
        "retrievers are never tested across unit kinds"
    ),
    "units_within_retriever": (
        "E2: each unit kind against evaluation.baseline_unit for the same retriever and the same "
        "queries; the tested quantity is the paired difference of lift (hearing-bootstrap "
        "interval, hearing sign-flip test), reported next to the paired difference of top1_chars; "
        "raw acc@1 and MRR are shown but not tested across unit kinds, because a larger unit is "
        "relevant by chance more often (random_acc_at_1) and a query whose units are all "
        "relevant cannot be missed"
    ),
    "holm_family": (
        "the set of p-values adjusted together: E1|<bench>|<split group>|<unit>|<test> and "
        "E2|<bench>|<split group>|<retriever>|<test>; each adjusted p-value carries the name of "
        "its family and each family its size"
    ),
}
E1_TESTS = (("acc_at_1", "mcnemar"), ("mrr", "sign_flip"))
E2_TESTS = (("acc_at_1_minus_random", "sign_flip"),)


def interval(result: Record) -> Record:
    return {key: rounded_or_none(result[key]) for key in INTERVAL_KEYS}


def hearing_interval(
    values: np.ndarray, hearings: np.ndarray, label: str, evaluation: Record
) -> Record:
    """The rounded hearing-bootstrap interval of the mean of ``values``."""
    return interval(
        bootstrap_mean(
            values,
            hearings,
            evaluation["bootstrap_samples"],
            evaluation["confidence_level"],
            stream_rng(evaluation["seed"], label),
        )
    )


def hearing_sign_flip(
    differences: np.ndarray, hearings: np.ndarray, label: str, evaluation: Record
) -> Record:
    return sign_flip_test(
        differences,
        hearings,
        evaluation["permutation_samples"],
        stream_rng(evaluation["seed"], label),
    )


def row_values(rows: list[Record], name: str, dtype: Any = None) -> np.ndarray:
    return np.array([row[name] for row in rows], dtype=dtype)


def summarize_rows(rows: list[Record], label: str, evaluation: Record) -> Record:
    hearings = row_values(rows, "hearing_id")
    ranks = row_values(rows, "rank", np.float64)
    optimistic = row_values(rows, "rank_optimistic")
    pessimistic = row_values(rows, "rank_pessimistic")
    random_acc = row_values(rows, "random_acc_at_1")
    top1_chars = row_values(rows, "top1_chars", np.float64)
    metrics: Record = {
        f"acc_at_{k}": hearing_interval(
            (ranks <= k).astype(np.float64), hearings, f"{label}|acc{k}", evaluation
        )
        for k in evaluation["ks"]
    }
    metrics["mrr"] = hearing_interval(1.0 / ranks, hearings, f"{label}|mrr", evaluation)
    metrics["acc_at_1_minus_random"] = hearing_interval(
        (ranks <= 1) - random_acc, hearings, f"{label}|lift", evaluation
    )
    return {
        "queries": len(rows),
        "hearings": int(len(np.unique(hearings))),
        **metrics,
        "random_acc_at_1": rounded_or_none(random_acc.mean()),
        "acc_at_1_tie_bounds": [
            rounded_or_none((pessimistic <= 1).mean()),
            rounded_or_none((optimistic <= 1).mean()),
        ],
        "queries_with_tie_at_first_relevant": int((optimistic != pessimistic).sum()),
        "trivial_queries": int(sum(1 for row in rows if row["n_relevant"] == row["n_units"])),
        "mean_units": rounded_or_none(np.mean(row_values(rows, "n_units")), COUNT_DECIMALS),
        "mean_relevant": rounded_or_none(np.mean(row_values(rows, "n_relevant")), COUNT_DECIMALS),
        "top1_chars": {
            "mean": rounded_or_none(top1_chars.mean(), CHARS_DECIMALS),
            "median": rounded_or_none(np.median(top1_chars), CHARS_DECIMALS),
        },
        "mean_unit_chars": rounded_or_none(
            np.mean(row_values(rows, "mean_unit_chars")), CHARS_DECIMALS
        ),
    }


def paired_rows(baseline: list[Record], other: list[Record]) -> list[Pair]:
    by_id = {row["query_id"]: row for row in baseline}
    return [(by_id[row["query_id"]], row) for row in other if row["query_id"] in by_id]


def column(pairs: list[Pair], side: int, name: str) -> np.ndarray:
    return np.array([pair[side][name] for pair in pairs], dtype=np.float64)


def pair_hearings(pairs: list[Pair]) -> np.ndarray:
    return np.array([pair[0]["hearing_id"] for pair in pairs])


def compare_retrievers(
    baseline: list[Record], other: list[Record], label: str, evaluation: Record
) -> Record:
    """E1: acc@1 (McNemar) and reciprocal rank (hearing sign-flip) of the paired queries."""
    paired = paired_rows(baseline, other)
    hearings = pair_hearings(paired)
    base_rank, other_rank = column(paired, 0, "rank"), column(paired, 1, "rank")
    base_hit, other_hit = base_rank <= 1, other_rank <= 1
    rr_diff = 1.0 / other_rank - 1.0 / base_rank
    return {
        "queries_paired": len(paired),
        "queries_only_in_other": len(other) - len(paired),
        "acc_at_1": {
            "baseline": rounded_or_none(base_hit.mean()),
            "other": rounded_or_none(other_hit.mean()),
            "difference": hearing_interval(
                other_hit.astype(np.float64) - base_hit, hearings, f"{label}|acc_diff", evaluation
            ),
            "mcnemar": mcnemar_exact(base_hit, other_hit),
        },
        "mrr": {
            "baseline": rounded_or_none((1.0 / base_rank).mean()),
            "other": rounded_or_none((1.0 / other_rank).mean()),
            "difference": hearing_interval(rr_diff, hearings, f"{label}|mrr_diff", evaluation),
            "sign_flip": hearing_sign_flip(rr_diff, hearings, f"{label}|flip", evaluation),
        },
        "top1_chars_mean": {
            "baseline": rounded_or_none(column(paired, 0, "top1_chars").mean(), CHARS_DECIMALS),
            "other": rounded_or_none(column(paired, 1, "top1_chars").mean(), CHARS_DECIMALS),
        },
    }


def trivial_counts(paired: list[Pair]) -> list[int]:
    return [
        sum(1 for pair in paired if pair[side]["n_relevant"] == pair[side]["n_units"])
        for side in (0, 1)
    ]


def compare_units(
    reference: list[Record], other: list[Record], label: str, evaluation: Record
) -> Record:
    """E2: the paired lift over chance (hearing sign-flip) and the paired top-1 length."""
    paired = paired_rows(reference, other)
    hearings = pair_hearings(paired)
    hits = [(column(paired, side, "rank") <= 1).astype(np.float64) for side in (0, 1)]
    randoms = [column(paired, side, "random_acc_at_1") for side in (0, 1)]
    lifts = [hit - random for hit, random in zip(hits, randoms, strict=True)]
    lift_diff = lifts[1] - lifts[0]
    chars = [column(paired, side, "top1_chars") for side in (0, 1)]
    trivial = trivial_counts(paired)
    return {
        "queries_paired": len(paired),
        "queries_only_in_other": len(other) - len(paired),
        "acc_at_1_minus_random": {
            "reference": rounded_or_none(lifts[0].mean()),
            "other": rounded_or_none(lifts[1].mean()),
            "difference": hearing_interval(lift_diff, hearings, f"{label}|lift_diff", evaluation),
            "sign_flip": hearing_sign_flip(lift_diff, hearings, f"{label}|lift_flip", evaluation),
        },
        "untested_descriptives": {
            "acc_at_1": {
                "reference": rounded_or_none(hits[0].mean()),
                "other": rounded_or_none(hits[1].mean()),
            },
            "random_acc_at_1": {
                "reference": rounded_or_none(randoms[0].mean()),
                "other": rounded_or_none(randoms[1].mean()),
            },
            "trivial_queries": {"reference": trivial[0], "other": trivial[1]},
        },
        "top1_chars": {
            "reference": rounded_or_none(chars[0].mean(), CHARS_DECIMALS),
            "other": rounded_or_none(chars[1].mean(), CHARS_DECIMALS),
            "difference": hearing_interval(
                chars[1] - chars[0], hearings, f"{label}|chars_diff", evaluation
            ),
        },
    }


def apply_holm(items: list[Record], tests: tuple[tuple[str, str], ...], family: str) -> Record:
    sizes: Record = {}
    for section, test in tests:
        name = f"{family}|{section}.{test}"
        p_values = [item[section][test]["p_value"] for item in items]
        for item, adjusted in zip(items, holm(p_values), strict=True):
            item[section][test]["p_holm"] = adjusted
            item[section][test]["holm_family"] = name
        sizes[name] = len(items)
    return sizes


def ordered_units(selected: RowGroups) -> list[str]:
    present = {unit for unit, _ in selected}
    return [kind for kind in UNIT_KINDS if kind in present]


def retriever_comparisons(
    selected: RowGroups, baseline_retriever: str, prefix: str, evaluation: Record
) -> Record:
    result: Record = {}
    for unit in ordered_units(selected):
        baseline = selected.get((unit, baseline_retriever))
        entry: Record = {"baseline": {"retriever": baseline_retriever, "unit": unit}}
        if baseline is None:
            result[unit] = {**entry, "baseline_missing": True, "holm_families": {}, "items": []}
            continue
        others = sorted(r for u, r in selected if u == unit and r != baseline_retriever)
        items = [
            {
                "retriever": retriever,
                "unit": unit,
                **compare_retrievers(
                    baseline,
                    selected[(unit, retriever)],
                    f"{prefix}|{unit}|{retriever}|vs",
                    evaluation,
                ),
            }
            for retriever in others
        ]
        families = apply_holm(items, E1_TESTS, f"E1|{prefix}|{unit}")
        result[unit] = {**entry, "holm_families": families, "items": items}
    return result


def unit_comparisons(
    selected: RowGroups, baseline_unit: str, prefix: str, evaluation: Record
) -> Record:
    result: Record = {}
    for retriever in sorted({r for _, r in selected}):
        units = [unit for unit in ordered_units(selected) if (unit, retriever) in selected]
        others = [unit for unit in units if unit != baseline_unit]
        if not others:
            continue
        reference = selected.get((baseline_unit, retriever))
        entry: Record = {"reference": {"retriever": retriever, "unit": baseline_unit}}
        if reference is None:
            result[retriever] = {
                **entry,
                "reference_missing": True,
                "holm_families": {},
                "items": [],
            }
            continue
        items = [
            {
                "retriever": retriever,
                "unit": unit,
                **compare_units(
                    reference,
                    selected[(unit, retriever)],
                    f"{prefix}|{retriever}|{unit}|vs_unit_{baseline_unit}",
                    evaluation,
                ),
            }
            for unit in others
        ]
        families = apply_holm(items, E2_TESTS, f"E2|{prefix}|{retriever}")
        result[retriever] = {**entry, "holm_families": families, "items": items}
    return result


def load_query_rows(
    run_dir: Path, splits: tuple[str, ...]
) -> tuple[dict[tuple[str, str, str], list[Record]], list[Record]]:
    grouped: dict[tuple[str, str, str], list[Record]] = {}
    files = []
    for path in sorted(run_dir.glob("queries/*/*/*/*.jsonl")):
        if path.stem not in splits:
            continue
        rows = load_jsonl(path)
        files.append(file_record(path, len(rows)))
        for row in rows:
            grouped.setdefault((row["bench"], row["unit"], row["retriever"]), []).append(row)
    return grouped, files


def split_groups(splits: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {split: (split,) for split in splits}
    if len(splits) > 1:
        groups["+".join(splits)] = splits
    return groups


def bench_rows(
    grouped: dict[tuple[str, str, str], list[Record]], bench: str, members: tuple[str, ...]
) -> RowGroups:
    selected = {
        (unit, retriever): [row for row in rows if row["split"] in members]
        for (row_bench, unit, retriever), rows in grouped.items()
        if row_bench == bench
    }
    return {key: rows for key, rows in selected.items() if rows}


def summarize_groups(
    grouped: dict[tuple[str, str, str], list[Record]],
    splits: tuple[str, ...],
    baseline: tuple[str, str],
    evaluation: Record,
) -> tuple[Record, Record]:
    """Per bench and split group: the summary of each (unit, retriever) and the E1/E2 tests."""
    baseline_unit, baseline_retriever = baseline
    summaries: Record = {}
    comparisons: Record = {}
    for bench in sorted({key[0] for key in grouped}):
        summaries[bench], comparisons[bench] = {}, {}
        for group, members in split_groups(splits).items():
            selected = bench_rows(grouped, bench, members)
            summaries[bench][group] = {}
            for (unit, retriever), rows in sorted(selected.items()):
                label = f"{bench}|{group}|{unit}|{retriever}"
                summaries[bench][group].setdefault(unit, {})[retriever] = summarize_rows(
                    rows, label, evaluation
                )
            prefix = f"{bench}|{group}"
            comparisons[bench][group] = {
                "retrievers_within_unit": retriever_comparisons(
                    selected, baseline_retriever, prefix, evaluation
                ),
                "units_within_retriever": unit_comparisons(
                    selected, baseline_unit, prefix, evaluation
                ),
            }
    return summaries, comparisons


def print_summary(summaries: Record) -> None:
    for bench, groups in summaries.items():
        for group, units in groups.items():
            for unit, retrievers in units.items():
                for retriever, summary in retrievers.items():
                    acc, mrr = summary["acc_at_1"], summary["mrr"]
                    print(
                        f"{bench:13s} {group:18s} {unit:8s} {retriever:22s} "
                        f"n={summary['queries']:4d} acc@1={acc['point']:.3f} "
                        f"[{acc['low']:.3f},{acc['high']:.3f}] mrr={mrr['point']:.3f} "
                        f"random@1={summary['random_acc_at_1']:.3f} "
                        f"top1_chars={summary['top1_chars']['mean']}"
                    )
