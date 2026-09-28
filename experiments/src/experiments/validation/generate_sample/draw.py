"""The strata of a UDV run, their population per split, and the seeded draw of each stratum."""

from typing import Any

import numpy as np

from experiments.common.splits import SPLIT_NAMES
from experiments.validation.generate_sample.config import NO_EVIDENCE_SUPPORT, Stratum

Record = dict[str, Any]


def record_order(record: Record) -> tuple[int, ...]:
    return tuple(int(part) for part in record["id"].split("-")[1:])


def support_of(record: Record) -> str:
    evidence = record["evidence"]
    return NO_EVIDENCE_SUPPORT if evidence is None else evidence["support_type"]


def stratum_of(record: Record, strata: tuple[Stratum, ...]) -> Stratum:
    cell = (record["tier"], support_of(record))
    matches = [s for s in strata if cell[0] in s.tiers and cell[1] in s.support_types]
    if len(matches) != 1:
        raise SystemExit(
            f"{record['id']}: (tier, support) {cell} matches {len(matches)} strata; the strata "
            "must partition every UDV of the run"
        )
    return matches[0]


def population_counts(
    records: list[Record],
    split_of: dict[int, str],
    strata: tuple[Stratum, ...],
    readable: set[str],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        split: {stratum.name: 0 for stratum in strata} for split in SPLIT_NAMES if split in readable
    }
    for record in records:
        counts[split_of[record["hearing_id"]]][stratum_of(record, strata).name] += 1
    return counts


def sum_population(counts: dict[str, dict[str, int]], splits: tuple[str, ...]) -> dict[str, int]:
    names = next(iter(counts.values())).keys()
    return {name: sum(counts[split][name] for split in splits) for name in names}


def draw_strata(
    records: list[Record], strata: tuple[Stratum, ...], rng: np.random.Generator
) -> tuple[dict[str, list[Record]], dict[str, Record]]:
    """Up to ``target`` UDVs of each stratum without replacement, and what each draw took."""
    pools: dict[str, list[Record]] = {stratum.name: [] for stratum in strata}
    for record in records:
        pools[stratum_of(record, strata).name].append(record)
    drawn: dict[str, list[Record]] = {}
    summary: dict[str, Record] = {}
    for stratum in strata:
        pool = sorted(pools[stratum.name], key=record_order)
        size = min(stratum.target, len(pool))
        picks = sorted(int(i) for i in rng.choice(len(pool), size=size, replace=False))
        drawn[stratum.name] = [pool[index] for index in picks]
        summary[stratum.name] = {
            "question": stratum.question,
            "target": stratum.target,
            "population": len(pool),
            "drawn_udvs": size,
            "all_taken": size == len(pool),
            "shortfall": stratum.target - size,
        }
    return drawn, summary
