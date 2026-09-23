import platform
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bookworm.config import SplitConfig
from bookworm.data.dates import article_date, summarize_date_extraction
from bookworm.data.io import JsonObject, json_dict_factory
from bookworm.data.schemas import HearingRecord
from bookworm.errors import SplitError
from bookworm.udv.schemas import UdvRecord

SplitName = Literal["train", "validation", "test"]
SplitGroups = dict[SplitName, tuple[int, ...]]

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")
GROUPING_METHOD = (
    "temporal: hearings ordered by the article publication date extracted from materia, "
    "cut at dates whose gap to the next dated hearing is at least min_boundary_gap_days"
)
DATE_FIELD = "first DD/MM/YYYY - HH:MM timestamp in materia (article publication)"
NEAR_DUPLICATE_METHOD = "tfidf cosine over materia, min_df=2, sublinear_tf"
RECOMPUTED_BOUNDARY_KEYS = (
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "train_gap_days",
    "test_gap_days",
)


@dataclass(frozen=True, slots=True)
class CutCandidate:
    day: date
    next_day: date
    gap_days: int
    cumulative: int
    fraction: float


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    train_gap_days: int
    test_gap_days: int
    train_fraction_reached: float
    validation_end_fraction_reached: float

    @classmethod
    def from_cuts(cls, train_cut: CutCandidate, validation_cut: CutCandidate) -> "SplitBoundaries":
        return cls(
            train_end=train_cut.day,
            validation_start=train_cut.next_day,
            validation_end=validation_cut.day,
            test_start=validation_cut.next_day,
            train_gap_days=train_cut.gap_days,
            test_gap_days=validation_cut.gap_days,
            train_fraction_reached=round(train_cut.fraction, 4),
            validation_end_fraction_reached=round(validation_cut.fraction, 4),
        )

    def to_dict(self) -> JsonObject:
        return asdict(self, dict_factory=json_dict_factory)


@dataclass(frozen=True)
class TemporalSplit:
    dates: Mapping[int, date]
    candidates: tuple[CutCandidate, ...]
    boundaries: SplitBoundaries
    groups: SplitGroups


@dataclass(frozen=True)
class SplitArtifacts:
    split: TemporalSplit
    manifest: JsonObject
    report: JsonObject


def dates_by_hearing(hearings: Sequence[HearingRecord]) -> dict[int, date]:
    dates: dict[int, date] = {}
    for hearing in hearings:
        extracted = article_date(hearing.materia)
        if extracted is None:
            raise SplitError(f"hearing {hearing.id}: no publication timestamp found in materia")
        dates[hearing.id] = extracted
    return dates


def cut_candidates(dates: Mapping[int, date], min_gap_days: int) -> list[CutCandidate]:
    ordered = sorted(set(dates.values()))
    counts = Counter(dates.values())
    total = len(dates)
    candidates: list[CutCandidate] = []
    cumulative = 0
    for position, day in enumerate(ordered[:-1]):
        cumulative += counts[day]
        following = ordered[position + 1]
        gap_days = (following - day).days
        if gap_days >= min_gap_days:
            candidates.append(
                CutCandidate(
                    day=day,
                    next_day=following,
                    gap_days=gap_days,
                    cumulative=cumulative,
                    fraction=cumulative / total,
                )
            )
    return candidates


def choose_boundary(
    candidates: Sequence[CutCandidate], target_fraction: float, after: date | None
) -> CutCandidate:
    eligible = [c for c in candidates if after is None or c.day > after]
    if not eligible:
        raise SplitError(f"no cut date available for target fraction {target_fraction}")
    return min(
        eligible,
        key=lambda c: (abs(c.fraction - target_fraction), -c.gap_days, c.day),
    )


def choose_cuts(
    candidates: Sequence[CutCandidate], train_fraction: float, validation_fraction: float
) -> tuple[CutCandidate, CutCandidate]:
    train_cut = choose_boundary(candidates, train_fraction, None)
    validation_cut = choose_boundary(
        candidates, train_fraction + validation_fraction, train_cut.day
    )
    return train_cut, validation_cut


def assign_splits(dates: Mapping[int, date], train_end: date, validation_end: date) -> SplitGroups:
    groups: dict[SplitName, list[int]] = {name: [] for name in SPLIT_NAMES}
    for hearing_id, day in sorted(dates.items(), key=lambda item: (item[1], item[0])):
        if day <= train_end:
            groups["train"].append(hearing_id)
        elif day <= validation_end:
            groups["validation"].append(hearing_id)
        else:
            groups["test"].append(hearing_id)
    return {name: tuple(ids) for name, ids in groups.items()}


def compute_temporal_split(hearings: Sequence[HearingRecord], config: SplitConfig) -> TemporalSplit:
    dates = dates_by_hearing(hearings)
    candidates = cut_candidates(dates, config.min_boundary_gap_days)
    train_cut, validation_cut = choose_cuts(
        candidates, config.train_fraction, config.validation_fraction
    )
    return TemporalSplit(
        dates=dates,
        candidates=tuple(candidates),
        boundaries=SplitBoundaries.from_cuts(train_cut, validation_cut),
        groups=assign_splits(dates, train_cut.day, validation_cut.day),
    )


def people_count(ids: Sequence[int], hearings_by_id: Mapping[int, HearingRecord]) -> int:
    return sum(len(hearings_by_id[i].metadados.envolvidos) for i in ids)


def opinion_count(ids: Sequence[int], hearings_by_id: Mapping[int, HearingRecord]) -> int:
    return sum(
        len(person.opinioes) for i in ids for person in hearings_by_id[i].metadados.envolvidos
    )


def summarize_split(
    ids: Sequence[int],
    hearings_by_id: Mapping[int, HearingRecord],
    dates: Mapping[int, date],
    udvs_by_hearing: Mapping[int, Sequence[UdvRecord]],
    total_hearings: int,
) -> JsonObject:
    days = [dates[hearing_id] for hearing_id in ids]
    udvs = [udv for hearing_id in ids for udv in udvs_by_hearing.get(hearing_id, [])]
    return {
        "hearings": len(ids),
        "share_of_hearings": round(len(ids) / total_hearings, 4),
        "first_date": min(days).isoformat(),
        "last_date": max(days).isoformat(),
        "distinct_dates": len(set(days)),
        "people": people_count(ids, hearings_by_id),
        "opinions": opinion_count(ids, hearings_by_id),
        "udvs": len(udvs),
        "udvs_by_tier": dict(sorted(Counter(udv.tier for udv in udvs).items())),
    }


def actor_overlap(groups: SplitGroups, hearings_by_id: Mapping[int, HearingRecord]) -> JsonObject:
    splits_by_actor: dict[str, set[str]] = defaultdict(set)
    for name, ids in groups.items():
        for hearing_id in ids:
            for person in hearings_by_id[hearing_id].metadados.envolvidos:
                splits_by_actor[person.nome.strip().upper()].add(name)
    return {
        "distinct_actors": len(splits_by_actor),
        "in_more_than_one_split": sum(1 for s in splits_by_actor.values() if len(s) > 1),
        "in_train_and_test": sum(1 for s in splits_by_actor.values() if {"train", "test"} <= s),
    }


def cross_split_similarity(
    groups: SplitGroups,
    hearings_by_id: Mapping[int, HearingRecord],
    threshold: float,
    top_pairs: int,
) -> JsonObject:
    ids = [hearing_id for name in SPLIT_NAMES for hearing_id in groups[name]]
    split_of = {hearing_id: name for name in SPLIT_NAMES for hearing_id in groups[name]}
    matrix = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(
        [hearings_by_id[hearing_id].materia for hearing_id in ids]
    )
    similarities = cosine_similarity(matrix)
    pairs = [
        (float(similarities[a, b]), ids[a], ids[b])
        for a in range(len(ids))
        for b in range(a + 1, len(ids))
        if split_of[ids[a]] != split_of[ids[b]]
    ]
    pairs.sort(reverse=True)
    scores = np.array([score for score, _, _ in pairs])
    return {
        "method": NEAR_DUPLICATE_METHOD,
        "threshold": threshold,
        "cross_split_pairs": len(pairs),
        "above_threshold": int((scores >= threshold).sum()),
        "max": round(float(scores.max()), 3),
        "p99": round(float(np.percentile(scores, 99)), 3),
        "median": round(float(np.median(scores)), 3),
        "top_pairs": [
            {
                "similarity": round(score, 3),
                "hearings": [first, second],
                "splits": [split_of[first], split_of[second]],
            }
            for score, first, second in pairs[:top_pairs]
        ],
    }


def calendar_summary(dates: Mapping[int, date]) -> JsonObject:
    return {
        "first_date": min(dates.values()).isoformat(),
        "last_date": max(dates.values()).isoformat(),
        "distinct_dates": len(set(dates.values())),
        "hearings_per_date_max": max(Counter(dates.values()).values()),
        "by_year": dict(sorted(Counter(day.year for day in dates.values()).items())),
    }


def split_environment() -> JsonObject:
    return {"python": platform.python_version(), "platform": platform.platform()}


def timestamp(created_at: datetime | None) -> str:
    moment = datetime.now(UTC) if created_at is None else created_at
    return moment.isoformat(timespec="seconds")


def build_manifest(
    split: TemporalSplit, config: SplitConfig, *, created_at: datetime | None = None
) -> JsonObject:
    return {
        "split_version": config.split_version,
        "grouping_method": GROUPING_METHOD,
        "seed": config.seed,
        "created_at": timestamp(created_at),
        "dataset": {"path": str(config.lds_path), "sha256": config.expected_sha256},
        "date_field": DATE_FIELD,
        "boundaries": split.boundaries.to_dict(),
        "train": list(split.groups["train"]),
        "validation": list(split.groups["validation"]),
        "test": list(split.groups["test"]),
        "article_dates": {str(i): split.dates[i].isoformat() for i in sorted(split.dates)},
    }


def group_udvs(udvs: Sequence[UdvRecord] | None) -> dict[int, list[UdvRecord]]:
    udvs_by_hearing: dict[int, list[UdvRecord]] = defaultdict(list)
    for udv in udvs or []:
        udvs_by_hearing[udv.hearing_id].append(udv)
    return udvs_by_hearing


def build_report(
    hearings: Sequence[HearingRecord],
    split: TemporalSplit,
    config: SplitConfig,
    udvs: Sequence[UdvRecord] | None = None,
    *,
    created_at: datetime | None = None,
    environment: Mapping[str, Any] | None = None,
) -> JsonObject:
    hearings_by_id = {hearing.id: hearing for hearing in hearings}
    udvs_by_hearing = group_udvs(udvs)
    return {
        "split_version": config.split_version,
        "created_at": timestamp(created_at),
        "date_extraction": summarize_date_extraction(hearings).to_dict(),
        "calendar": calendar_summary(split.dates),
        "boundaries": split.boundaries.to_dict(),
        "cut_candidates_considered": len(split.candidates),
        "splits": {
            name: summarize_split(
                split.groups[name], hearings_by_id, split.dates, udvs_by_hearing, len(hearings)
            )
            for name in SPLIT_NAMES
        },
        "leakage": {
            "actors": actor_overlap(split.groups, hearings_by_id),
            "near_duplicates": cross_split_similarity(
                split.groups,
                hearings_by_id,
                config.near_duplicate_threshold,
                config.near_duplicate_pairs,
            ),
        },
        "udv_source": str(config.udv_path) if udvs is not None else None,
        "environment": dict(split_environment() if environment is None else environment),
        "config": config.source,
    }


def build_temporal_split(
    hearings: Sequence[HearingRecord],
    config: SplitConfig,
    udvs: Sequence[UdvRecord] | None = None,
    *,
    created_at: datetime | None = None,
    environment: Mapping[str, Any] | None = None,
) -> SplitArtifacts:
    split = compute_temporal_split(hearings, config)
    moment = datetime.now(UTC) if created_at is None else created_at
    return SplitArtifacts(
        split=split,
        manifest=build_manifest(split, config, created_at=moment),
        report=build_report(
            hearings, split, config, udvs, created_at=moment, environment=environment
        ),
    )
