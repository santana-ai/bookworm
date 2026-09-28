"""The temporal split manifest as the experiment scripts read it: hearing id to split name."""

import json
from pathlib import Path
from typing import Any

from bookworm import SPLIT_NAMES, sha256_of_file

__all__ = ["SPLIT_NAMES", "load_split_lookup", "read_split_lookup", "split_groups", "split_lookup"]

Record = dict[str, Any]


def split_lookup(manifest: Record) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for name in SPLIT_NAMES:
        for hearing_id in manifest[name]:
            if hearing_id in lookup:
                raise SystemExit(f"hearing {hearing_id} is listed in two splits")
            lookup[hearing_id] = name
    return lookup


def read_split_lookup(manifest_path: Path) -> dict[int, str]:
    with open(manifest_path) as f:
        return split_lookup(json.load(f))


def load_split_lookup(manifest_path: Path, lds_sha256: str) -> tuple[dict[int, str], Record]:
    """The split of each hearing and the manifest's source record, refusing another LDS file."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != lds_sha256:
        raise SystemExit(f"{manifest_path} was built from another LDS file")
    source = {
        "path": str(manifest_path),
        "sha256": sha256_of_file(manifest_path),
        "split_version": manifest["split_version"],
    }
    return split_lookup(manifest), source


def split_groups() -> dict[str, list[str]]:
    """Each split on its own and ``all`` of them together, the groups a benchmark report uses."""
    groups: dict[str, list[str]] = {name: [name] for name in SPLIT_NAMES}
    groups["all"] = list(SPLIT_NAMES)
    return groups
