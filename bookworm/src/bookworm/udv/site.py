"""Demo data directory: one JSON per hearing and ``index.json`` (``export-site``)."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm.data.io import JsonObject, write_json
from bookworm.data.schemas import HearingRecord
from bookworm.data.splits import SplitName
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder
from bookworm.profiles.site import ProfileSiteBuilder, ProfileSiteExport, clear_profile_site
from bookworm.transcript.text import normalize_whitespace
from bookworm.udv.build import EvidenceSettings
from bookworm.udv.export import DEFAULT_TOP_K, export_hearing, split_of
from bookworm.udv.schemas import SUPPORT_TYPES, TIERS, UdvRecord
from bookworm.udv.signals import SiteSignals
from bookworm.udv.site_validation import SiteValidation

HearingCallback = Callable[[int, JsonObject, int], None]

INDEX_FILE_NAME = "index.json"
HEARINGS_DIR_NAME = "hearings"
TITLE_MAX_CHARS = 120
TITLE_ELLIPSIS = "…"
TITLE_TRAILING_CHARACTERS = " ,;:-"


@dataclass(frozen=True)
class SiteExport:
    index: JsonObject
    hearing_bytes: dict[int, int]
    index_bytes: int
    profiles: ProfileSiteExport | None = None

    @property
    def total_bytes(self) -> int:
        extra = 0 if self.profiles is None else self.profiles.total_bytes
        return self.index_bytes + sum(self.hearing_bytes.values()) + extra


def display_title(materia: str, assunto: str, max_chars: int = TITLE_MAX_CHARS) -> str:
    headline = next(
        (line for line in (normalize_whitespace(raw) for raw in materia.splitlines()) if line), ""
    )
    title = headline or normalize_whitespace(assunto)
    if len(title) <= max_chars:
        return title
    room = title[: max_chars - len(TITLE_ELLIPSIS) + 1]
    boundary = room.rfind(" ")
    kept = room[:boundary] if boundary > 0 else room[: max_chars - len(TITLE_ELLIPSIS)]
    return kept.rstrip(TITLE_TRAILING_CHARACTERS) + TITLE_ELLIPSIS


def site_index_entry(payload: Mapping[str, Any]) -> JsonObject:
    hearing = payload["hearing"]
    people = payload["people"]
    udvs = payload["udvs"]
    tiers = Counter(udv["tier"] for udv in udvs)
    support_types = Counter(
        udv["evidence"]["support_type"] for udv in udvs if udv["evidence"] is not None
    )
    return {
        "id": hearing["id"],
        "split": hearing["split"],
        "article_date": hearing["article_date"],
        "assunto": hearing["assunto"],
        "title": display_title(hearing["materia"], hearing["assunto"]),
        "n_udvs": len(udvs),
        "n_people": len(people),
        "n_people_resolved": sum(1 for person in people if person["resolved"]),
        "tiers": {tier: tiers[tier] for tier in TIERS},
        "support_types": {kind: support_types[kind] for kind in SUPPORT_TYPES},
        "transcript_words": hearing["transcript_words"],
        "actors": [person["name"] for person in people],
    }


def hearing_file(output_dir: Path, hearing_id: int) -> Path:
    return output_dir / HEARINGS_DIR_NAME / f"{hearing_id}.json"


def group_records(
    records: Sequence[UdvRecord], hearing_ids: Sequence[int]
) -> dict[int, list[UdvRecord]]:
    grouped: dict[int, list[UdvRecord]] = {hearing_id: [] for hearing_id in hearing_ids}
    unknown = sorted({record.hearing_id for record in records} - grouped.keys())
    if unknown:
        raise ConfigError(f"the run has records of hearings outside its coverage: {unknown[:3]}")
    for record in records:
        grouped[record.hearing_id].append(record)
    return grouped


def hearing_splits(
    hearings: Sequence[HearingRecord], split_manifest: Mapping[str, Any] | None
) -> dict[int, SplitName]:
    if split_manifest is None:
        return {}
    return {hearing.id: split_of(split_manifest, hearing.id) for hearing in hearings}


def clear_site(output_dir: Path) -> Path:
    index_path = output_dir / INDEX_FILE_NAME
    index_path.unlink(missing_ok=True)
    clear_profile_site(output_dir)
    return index_path


def check_same_run(run: JsonObject, payload: Mapping[str, Any], first_hearing_id: int) -> None:
    if payload["run"] != run:
        hearing_id = payload["hearing"]["id"]
        raise ConfigError(
            f"hearing {hearing_id}: run block {payload['run']} differs from {run} "
            f"of hearing {first_hearing_id}"
        )


def write_hearing(payload: JsonObject, output_dir: Path) -> int:
    path = hearing_file(output_dir, payload["hearing"]["id"])
    write_json(payload, path)
    return path.stat().st_size


def export_site(
    hearings: Sequence[HearingRecord],
    records: Sequence[UdvRecord],
    encoder: CachedEncoder,
    output_dir: Path,
    *,
    run_name: str,
    pipeline: object,
    top_k: int = DEFAULT_TOP_K,
    split_manifest: Mapping[str, Any] | None = None,
    on_hearing: HearingCallback | None = None,
    signals: SiteSignals | None = None,
    profiles: ProfileSiteBuilder | None = None,
    settings: EvidenceSettings | None = None,
    validation: SiteValidation | None = None,
) -> SiteExport:
    """Write the demo JSON of every hearing and then ``index.json`` to ``output_dir``."""
    ordered = sorted(hearings, key=lambda hearing: hearing.id)
    if not ordered:
        raise ConfigError(f"run {run_name} has no hearings to export")
    grouped = group_records(records, [hearing.id for hearing in ordered])
    splits = hearing_splits(ordered, split_manifest)
    index_path = clear_site(output_dir)
    entries: list[JsonObject] = []
    hearing_bytes: dict[int, int] = {}
    run: JsonObject | None = None
    for number, hearing in enumerate(ordered, start=1):
        payload = export_hearing(
            hearing,
            grouped[hearing.id],
            encoder,
            run_name=run_name,
            pipeline=pipeline,
            top_k=top_k,
            split=splits.get(hearing.id),
            settings=settings,
            signals=signals,
        )
        if run is None:
            run = payload["run"]
        else:
            check_same_run(run, payload, ordered[0].id)
        hearing_bytes[hearing.id] = write_hearing(payload, output_dir)
        entry = site_index_entry(payload)
        entries.append(entry)
        if profiles is not None:
            profiles.add_hearing(payload, entry)
        if on_hearing is not None:
            on_hearing(number, entry, hearing_bytes[hearing.id])
    profile_site = None if profiles is None else profiles.write(output_dir, run)
    index: JsonObject = {"run": run, "hearings": entries}
    if validation is not None:
        index["validation"] = validation.index_block(signals)
    write_json(index, index_path)
    return SiteExport(index, hearing_bytes, index_path.stat().st_size, profile_site)
