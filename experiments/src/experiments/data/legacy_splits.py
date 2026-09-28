import argparse
import platform
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataset_io import load_gated_jsonl, load_jsonl, write_json
from utils.hearing_dates import article_date, check_article_date

Record = dict[str, Any]

SPLIT_NAMES = ("train", "validation", "test")
GROUPING_METHOD = (
    "temporal: hearings ordered by the article publication date extracted from materia, "
    "cut at dates whose gap to the next dated hearing is at least min_boundary_gap_days"
)


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


def summarize_split(
    ids: list[int],
    records_by_id: dict[int, Record],
    dates: dict[int, date],
    udvs_by_hearing: dict[int, list[Record]],
    total_hearings: int,
) -> Record:
    days = [dates[hearing_id] for hearing_id in ids]
    udvs = [udv for hearing_id in ids for udv in udvs_by_hearing.get(hearing_id, [])]
    return {
        "hearings": len(ids),
        "share_of_hearings": round(len(ids) / total_hearings, 4),
        "first_date": min(days).isoformat(),
        "last_date": max(days).isoformat(),
        "distinct_dates": len(set(days)),
        "people": sum(len(records_by_id[i]["metadados"]["envolvidos"]) for i in ids),
        "opinions": sum(
            len(person["opinioes"])
            for i in ids
            for person in records_by_id[i]["metadados"]["envolvidos"]
        ),
        "udvs": len(udvs),
        "udvs_by_tier": dict(sorted(Counter(udv["tier"] for udv in udvs).items())),
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


def build_manifest(
    groups: dict[str, list[int]],
    dates: dict[int, date],
    boundaries: Record,
    config: SplitConfig,
) -> Record:
    return {
        "split_version": config.split_version,
        "grouping_method": GROUPING_METHOD,
        "seed": config.seed,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": str(config.lds_path), "sha256": config.expected_sha256},
        "date_field": "first DD/MM/YYYY - HH:MM timestamp in materia (article publication)",
        "boundaries": boundaries,
        "train": groups["train"],
        "validation": groups["validation"],
        "test": groups["test"],
        "article_dates": {str(i): dates[i].isoformat() for i in sorted(dates)},
    }


def build_report(
    records: list[Record],
    groups: dict[str, list[int]],
    dates: dict[int, date],
    boundaries: Record,
    candidates: list[Record],
    config: SplitConfig,
) -> Record:
    records_by_id = {record["id"]: record for record in records}
    udvs_by_hearing: dict[int, list[Record]] = defaultdict(list)
    if config.udv_path.exists():
        for udv in load_jsonl(config.udv_path):
            udvs_by_hearing[udv["hearing_id"]].append(udv)
    return {
        "split_version": config.split_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "date_extraction": summarize_date_extraction(records),
        "calendar": {
            "first_date": min(dates.values()).isoformat(),
            "last_date": max(dates.values()).isoformat(),
            "distinct_dates": len(set(dates.values())),
            "hearings_per_date_max": max(Counter(dates.values()).values()),
            "by_year": dict(sorted(Counter(day.year for day in dates.values()).items())),
        },
        "boundaries": boundaries,
        "cut_candidates_considered": len(candidates),
        "splits": {
            name: summarize_split(groups[name], records_by_id, dates, udvs_by_hearing, len(records))
            for name in SPLIT_NAMES
        },
        "leakage": {
            "actors": actor_overlap(groups, records_by_id),
            "near_duplicates": cross_split_similarity(groups, records_by_id, config),
        },
        "udv_source": str(config.udv_path) if config.udv_path.exists() else None,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "config": config.source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the temporal split manifest for the 206 hearings of the LDS file."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/splits.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = load_gated_jsonl(config.lds_path, config.expected_sha256)
    dates = dates_by_hearing(records)

    candidates = cut_candidates(dates, config.min_boundary_gap_days)
    train_cut = choose_boundary(candidates, config.train_fraction, None)
    validation_cut = choose_boundary(
        candidates, config.train_fraction + config.validation_fraction, train_cut["date"]
    )
    groups = assign_splits(dates, train_cut["date"], validation_cut["date"])
    boundaries = {
        "train_end": train_cut["date"].isoformat(),
        "validation_start": train_cut["next_date"].isoformat(),
        "validation_end": validation_cut["date"].isoformat(),
        "test_start": validation_cut["next_date"].isoformat(),
        "train_gap_days": train_cut["gap_days"],
        "test_gap_days": validation_cut["gap_days"],
        "train_fraction_reached": round(train_cut["fraction"], 4),
        "validation_end_fraction_reached": round(validation_cut["fraction"], 4),
    }

    manifest = build_manifest(groups, dates, boundaries, config)
    report = build_report(records, groups, dates, boundaries, candidates, config)
    write_json(manifest, config.output_dir / f"{config.split_version}.json")
    write_json(report, config.output_dir / f"{config.split_version}_report.json")
    for name in SPLIT_NAMES:
        summary = report["splits"][name]
        print(
            f"{name:11s} {summary['hearings']:3d} hearings "
            f"({summary['share_of_hearings']:.1%}) "
            f"{summary['first_date']}..{summary['last_date']} "
            f"{summary['udvs']:4d} udvs"
        )
    print(f"boundaries: {boundaries}")


if __name__ == "__main__":
    main()
