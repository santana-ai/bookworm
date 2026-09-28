"""The split and date functions the splits notebook was written against, kept for that notebook.

``notebooks/splits.ipynb`` explores the temporal split with these dict-based functions, which
predate the library. The split itself is built and checked by ``bookworm build-splits`` and
``bookworm verify-splits`` (``bookworm.data.splits``, ``bookworm.data.dates`` and
``bookworm.data.verify_splits``), which reproduce ``artifacts/splits/temporal_v1.json`` and its
report; ``bookworm/tests/integration/test_parity_splits.py`` compares the library with these
functions on the 206 hearings. Only the functions the notebook and that test call are kept; the
command-line entry points were removed with the scripts they belonged to.
"""

import re
import tomllib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Record = dict[str, Any]

TIMESTAMP_PATTERN = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})")
UPDATED_AT_PATTERN = re.compile(
    r"Atualizado\s+em\s+(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2}):(\d{2})", re.IGNORECASE
)
WEEKDAY_MENTION_PATTERN = re.compile(
    r"(nest[ae]\s+)?(segunda|ter[cç]a|quarta|quinta|sexta)-feira\s*\((\d{1,2})\)", re.IGNORECASE
)
WEEKDAY_INDEX = {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4}
MENTION_LOOKBACK_DAYS = 30


def strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def parse_timestamp(match: re.Match[str]) -> datetime:
    day, month, year, hour, minute = (int(group) for group in match.groups())
    return datetime(year, month, day, hour, minute)


def updated_spans(article: str) -> list[tuple[int, int]]:
    return [(match.start(1), match.end(5)) for match in UPDATED_AT_PATTERN.finditer(article)]


def published_at(article: str) -> datetime | None:
    skip = updated_spans(article)
    for match in TIMESTAMP_PATTERN.finditer(article):
        if any(start <= match.start() < end for start, end in skip):
            continue
        return parse_timestamp(match)
    return None


def updated_at(article: str) -> datetime | None:
    match = UPDATED_AT_PATTERN.search(article)
    return parse_timestamp(match) if match else None


def article_date(article: str) -> date | None:
    published = published_at(article)
    return published.date() if published else None


def weekday_mentions(article: str) -> list[Record]:
    return [
        {
            "text": match.group(0).strip(),
            "names_current_event": match.group(1) is not None,
            "weekday": WEEKDAY_INDEX[strip_accents(match.group(2)).lower()],
            "day_of_month": int(match.group(3)),
        }
        for match in WEEKDAY_MENTION_PATTERN.finditer(article)
    ]


def resolve_mention_date(reference: date, day_of_month: int) -> date | None:
    for offset in range(MENTION_LOOKBACK_DAYS + 1):
        candidate = date.fromordinal(reference.toordinal() - offset)
        if candidate.day == day_of_month:
            return candidate
    return None


def check_weekday_mention(reference: date, mention: Record) -> Record:
    resolved = resolve_mention_date(reference, mention["day_of_month"])
    return {
        "text": mention["text"],
        "names_current_event": mention["names_current_event"],
        "resolved_date": resolved.isoformat() if resolved else None,
        "lag_days": reference.toordinal() - resolved.toordinal() if resolved else None,
        "weekday_agrees": resolved is not None and resolved.weekday() == mention["weekday"],
    }


def check_article_date(article: str) -> Record:
    reference = article_date(article)
    if reference is None:
        return {"article_date": None, "updated_at": None, "mentions": []}
    updated = updated_at(article)
    return {
        "article_date": reference.isoformat(),
        "updated_at": updated.isoformat() if updated else None,
        "mentions": [check_weekday_mention(reference, m) for m in weekday_mentions(article)],
    }


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    lds_path: Path
    expected_sha256: str
    split_version: str
    train_fraction: float
    validation_fraction: float
    min_boundary_gap_days: int
    near_duplicate_threshold: float
    near_duplicate_pairs: int
    seed: int
    output_dir: Path
    udv_path: Path
    source: Record = field(default_factory=dict)


def load_config(config_path: Path) -> SplitConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return SplitConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        expected_sha256=raw["dataset"]["sha256"],
        split_version=raw["temporal"]["split_version"],
        train_fraction=raw["temporal"]["train_fraction"],
        validation_fraction=raw["temporal"]["validation_fraction"],
        min_boundary_gap_days=raw["temporal"]["min_boundary_gap_days"],
        near_duplicate_threshold=raw["leakage"]["near_duplicate_threshold"],
        near_duplicate_pairs=raw["leakage"]["near_duplicate_pairs"],
        seed=raw["run"]["seed"],
        output_dir=Path(raw["run"]["output_dir"]),
        udv_path=Path(raw["run"]["udv_path"]),
        source=raw,
    )


def dates_by_hearing(records: list[Record]) -> dict[int, date]:
    dates: dict[int, date] = {}
    for record in records:
        extracted = article_date(record["materia"])
        if extracted is None:
            raise SystemExit(f"hearing {record['id']}: no publication timestamp found in materia")
        dates[record["id"]] = extracted
    return dates


def cut_candidates(dates: dict[int, date], min_gap_days: int) -> list[Record]:
    ordered = sorted(set(dates.values()))
    counts = Counter(dates.values())
    total = len(dates)
    candidates: list[Record] = []
    cumulative = 0
    for position, day in enumerate(ordered[:-1]):
        cumulative += counts[day]
        following = ordered[position + 1]
        gap_days = (following - day).days
        if gap_days >= min_gap_days:
            candidates.append(
                {
                    "date": day,
                    "next_date": following,
                    "gap_days": gap_days,
                    "cumulative": cumulative,
                    "fraction": cumulative / total,
                }
            )
    return candidates


def choose_boundary(candidates: list[Record], target_fraction: float, after: date | None) -> Record:
    eligible = [c for c in candidates if after is None or c["date"] > after]
    if not eligible:
        raise SystemExit(f"no cut date available for target fraction {target_fraction}")
    return min(
        eligible,
        key=lambda c: (abs(c["fraction"] - target_fraction), -c["gap_days"], c["date"]),
    )


def assign_splits(
    dates: dict[int, date], train_end: date, validation_end: date
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for hearing_id, day in sorted(dates.items(), key=lambda item: (item[1], item[0])):
        if day <= train_end:
            groups["train"].append(hearing_id)
        elif day <= validation_end:
            groups["validation"].append(hearing_id)
        else:
            groups["test"].append(hearing_id)
    return groups


def summarize_date_extraction(records: list[Record]) -> Record:
    checks = {record["id"]: check_article_date(record["materia"]) for record in records}
    current = [
        (hearing_id, mention)
        for hearing_id, check in checks.items()
        for mention in check["mentions"]
        if mention["names_current_event"]
    ]
    agreeing = [pair for pair in current if pair[1]["weekday_agrees"]]
    mentioning_hearings = {hearing_id for hearing_id, _ in current}
    confirmed_hearings = {hearing_id for hearing_id, _ in agreeing}
    return {
        "hearings": len(records),
        "with_publication_timestamp": sum(1 for c in checks.values() if c["article_date"]),
        "with_update_timestamp": sum(1 for c in checks.values() if c["updated_at"]),
        "with_current_event_mention": len(mentioning_hearings),
        "confirmed_by_at_least_one_mention": len(confirmed_hearings),
        "confirmed_by_no_mention": len(mentioning_hearings - confirmed_hearings),
        "current_event_mentions": len(current),
        "weekday_agrees": len(agreeing),
        "weekday_disagrees": len(current) - len(agreeing),
        "lag_days_when_agreeing": dict(
            sorted(Counter(mention["lag_days"] for _, mention in agreeing).items())
        ),
        "disagreements": [
            {
                "hearing_id": hearing_id,
                "article_date": checks[hearing_id]["article_date"],
                "mention": mention["text"],
                "resolved_date": mention["resolved_date"],
            }
            for hearing_id, mention in current
            if not mention["weekday_agrees"]
        ],
        "hearings_without_any_agreeing_mention": sorted(mentioning_hearings - confirmed_hearings),
    }


def actor_overlap(groups: dict[str, list[int]], records_by_id: dict[int, Record]) -> Record:
    splits_by_actor: dict[str, set[str]] = defaultdict(set)
    for name, ids in groups.items():
        for hearing_id in ids:
            for person in records_by_id[hearing_id]["metadados"]["envolvidos"]:
                splits_by_actor[person["nome"].strip().upper()].add(name)
    return {
        "distinct_actors": len(splits_by_actor),
        "in_more_than_one_split": sum(1 for s in splits_by_actor.values() if len(s) > 1),
        "in_train_and_test": sum(1 for s in splits_by_actor.values() if {"train", "test"} <= s),
    }


def cross_split_similarity(
    groups: dict[str, list[int]], records_by_id: dict[int, Record], config: SplitConfig
) -> Record:
    ids = [hearing_id for name in SPLIT_NAMES for hearing_id in groups[name]]
    split_of = {hearing_id: name for name in SPLIT_NAMES for hearing_id in groups[name]}
    matrix = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(
        [records_by_id[hearing_id]["materia"] for hearing_id in ids]
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
        "method": "tfidf cosine over materia, min_df=2, sublinear_tf",
        "threshold": config.near_duplicate_threshold,
        "cross_split_pairs": len(pairs),
        "above_threshold": int((scores >= config.near_duplicate_threshold).sum()),
        "max": round(float(scores.max()), 3),
        "p99": round(float(np.percentile(scores, 99)), 3),
        "median": round(float(np.median(scores)), 3),
        "top_pairs": [
            {
                "similarity": round(score, 3),
                "hearings": [first, second],
                "splits": [split_of[first], split_of[second]],
            }
            for score, first, second in pairs[: config.near_duplicate_pairs]
        ],
    }


MANIFEST_KEYS = ("split_version", "grouping_method", "seed", "train", "validation", "test")


def check_manifest_shape(manifest: Record) -> list[str]:
    problems = [f"missing_key:{key}" for key in MANIFEST_KEYS if key not in manifest]
    for name in SPLIT_NAMES:
        ids = manifest.get(name)
        if ids is not None and not all(isinstance(i, int) for i in ids):
            problems.append(f"non_integer_ids:{name}")
    return problems


def check_partition(manifest: Record, dates: dict[int, date]) -> list[str]:
    assigned = [i for name in SPLIT_NAMES for i in manifest[name]]
    problems: list[str] = []
    for hearing_id, count in Counter(assigned).items():
        if count > 1:
            problems.append(f"duplicate_hearing:{hearing_id}")
    for hearing_id in sorted(set(dates) - set(assigned)):
        problems.append(f"unassigned_hearing:{hearing_id}")
    for hearing_id in sorted(set(assigned) - set(dates)):
        problems.append(f"unknown_hearing:{hearing_id}")
    return problems


def check_dates(manifest: Record, dates: dict[int, date]) -> list[str]:
    recorded = manifest.get("article_dates", {})
    problems: list[str] = []
    if len(recorded) != len(dates):
        problems.append(f"article_dates_size:{len(recorded)} != {len(dates)}")
    for hearing_id, day in sorted(dates.items()):
        if recorded.get(str(hearing_id)) != day.isoformat():
            problems.append(f"article_date_mismatch:{hearing_id}")
    return problems


def check_chronology(manifest: Record, dates: dict[int, date], min_gap_days: int) -> list[str]:
    ranges = {
        name: (
            min(dates[i] for i in manifest[name]),
            max(dates[i] for i in manifest[name]),
        )
        for name in SPLIT_NAMES
        if manifest[name]
    }
    problems: list[str] = []
    for earlier, later in (("train", "validation"), ("validation", "test")):
        if earlier not in ranges or later not in ranges:
            problems.append(f"empty_split:{earlier if earlier not in ranges else later}")
            continue
        gap = (ranges[later][0] - ranges[earlier][1]).days
        if gap <= 0:
            problems.append(f"not_chronological:{earlier}->{later}")
        elif gap < min_gap_days:
            problems.append(f"boundary_gap_below_minimum:{earlier}->{later}:{gap}")
    days_by_split = defaultdict(set)
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            days_by_split[dates[hearing_id]].add(name)
    for day, names in sorted(days_by_split.items()):
        if len(names) > 1:
            problems.append(f"date_straddles_splits:{day.isoformat()}")
    return problems


def check_boundaries(manifest: Record, dates: dict[int, date], config: SplitConfig) -> list[str]:
    candidates = cut_candidates(dates, config.min_boundary_gap_days)
    train_cut = choose_boundary(candidates, config.train_fraction, None)
    validation_cut = choose_boundary(
        candidates, config.train_fraction + config.validation_fraction, train_cut["date"]
    )
    recomputed = {
        "train_end": train_cut["date"].isoformat(),
        "validation_start": train_cut["next_date"].isoformat(),
        "validation_end": validation_cut["date"].isoformat(),
        "test_start": validation_cut["next_date"].isoformat(),
        "train_gap_days": train_cut["gap_days"],
        "test_gap_days": validation_cut["gap_days"],
    }
    recorded = manifest["boundaries"]
    return [
        f"boundary_mismatch:{key}: manifest {recorded.get(key)}, recomputed {value}"
        for key, value in recomputed.items()
        if recorded.get(key) != value
    ]
