"""Premise units built from the UDV records."""

from collections import Counter
from typing import Any

from bookworm import load_jsonl

from experiments.common.transcript import normalize_whitespace
from experiments.udv.calibrate_threshold import load_split_lookup
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.config import (
    VerifierConfig,
)
from experiments.verifier.udv_scores.config import SUPPORT_TYPE_NONE, UdvVerifierConfig

Record = dict[str, Any]


def support_type(record: Record) -> str:
    evidence = record.get("evidence")
    return evidence["support_type"] if evidence else SUPPORT_TYPE_NONE


def udv_units(
    records: list[Record], tiers: tuple[str, ...], split_of: dict[int, str]
) -> tuple[list[PremiseUnit], Counter[str]]:
    units: list[PremiseUnit] = []
    unscored: Counter[str] = Counter()
    for record in records:
        if record["tier"] not in tiers:
            if record.get("evidence"):
                raise SystemExit(f"{record['id']}: tier {record['tier']} carries evidence")
            unscored[record["tier"]] += 1
            continue
        text = normalize_whitespace((record.get("evidence") or {}).get("text") or "")
        if not text:
            raise SystemExit(f"{record['id']}: evidence tier {record['tier']} without text")
        split = split_of.get(record["hearing_id"])
        if split is None:
            raise SystemExit(f"{record['id']}: hearing {record['hearing_id']} not in the manifest")
        units.append(
            PremiseUnit(
                unit_id=record["id"],
                hearing_id=record["hearing_id"],
                split=split,
                hypothesis=normalize_whitespace(record["proposition"]),
                items=(text,),
            )
        )
    duplicated = [key for key, count in Counter(u.unit_id for u in units).items() if count > 1]
    if duplicated:
        raise SystemExit(f"UDV ids are not unique: {duplicated[:10]}")
    return units, unscored


def load_udvs(
    config: UdvVerifierConfig, verifier: VerifierConfig
) -> tuple[list[Record], list[PremiseUnit], Counter[str], dict[int, str], Record]:
    records = load_jsonl(config.udv_path)
    split_of, split_source = load_split_lookup(verifier.manifest_path, verifier.lds_sha256)
    units, unscored = udv_units(records, config.evidence_tiers, split_of)
    return records, units, unscored, split_of, split_source
