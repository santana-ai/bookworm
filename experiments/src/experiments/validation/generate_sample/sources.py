"""What a sample reads: the splits it may touch, its output folders, the UDV run and calibration."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file

from experiments.common.provenance import REPOSITORY_DIR
from experiments.validation.generate_sample.config import ValidationConfig, check_split_names

Record = dict[str, Any]

ALWAYS_READABLE_SPLITS = ("train", "validation")
DRY_RUN_PREFIX = "validation_dry_run_"


@dataclass(frozen=True)
class Locations:
    sample_dir: Path
    transcripts_dir: Path


def resolve_splits(
    config: ValidationConfig, override: list[str] | None, dry_run: bool, final_test: bool
) -> tuple[str, ...]:
    if override is not None and not dry_run:
        raise SystemExit("--splits overrides the configured splits only together with --dry-run")
    splits = tuple(override) if override is not None else config.sample_splits
    check_split_names(splits)
    if "test" in splits and not final_test:
        raise SystemExit(
            f"the sample is drawn from {splits}, which includes test; pass --final-test to "
            "confirm that test hearings may be read"
        )
    return splits


def readable_splits(splits: tuple[str, ...], final_test: bool) -> set[str]:
    return {*ALWAYS_READABLE_SPLITS, *splits, *(("test",) if final_test else ())}


def resolve_locations(
    config: ValidationConfig,
    sample_name: str,
    dry_run: bool,
    output_dir: Path | None,
    create_temp: bool,
) -> Locations:
    """The sample and transcript folders; a dry run writes outside the repository only."""
    if not dry_run:
        if output_dir is not None:
            raise SystemExit("--output-dir is accepted only together with --dry-run")
        return Locations(
            config.output_dir / sample_name, config.transcripts_dir / sample_name / "transcripts"
        )
    if output_dir is None:
        if not create_temp:
            raise SystemExit("a dry-run repeat stage needs --output-dir of its dry-run sample")
        output_dir = Path(tempfile.mkdtemp(prefix=DRY_RUN_PREFIX))
    base = output_dir.resolve()
    if base == REPOSITORY_DIR or REPOSITORY_DIR in base.parents:
        raise SystemExit(f"dry-run output must be outside the repository {REPOSITORY_DIR}")
    return Locations(
        base / "validation" / sample_name, base / "cache" / sample_name / "transcripts"
    )


def calibration_artifact_problems(
    artifact: Record,
    threshold: float,
    splits: tuple[str, ...],
    split_of: dict[int, str],
    manifest_sha256: str,
) -> tuple[Record, list[str]]:
    leak = artifact.get("leak_check") or {}
    calibration_splits = list(leak.get("calibration_splits") or [])
    hearings = list(leak.get("hearing_ids_used") or [])
    rules = artifact.get("rules") or {}
    producing = sorted(
        name for name, rule in rules.items() if rule.get("threshold_rounded") == threshold
    )
    manifest = (artifact.get("sources") or {}).get("splits", {}).get("sha256")
    overlap = sorted(hearing for hearing in hearings if split_of.get(hearing) in splits)
    details = {
        "calibration_version": artifact.get("calibration_version"),
        "calibration_splits": calibration_splits,
        "calibration_hearings": len(hearings),
        "calibration_hearings_in_sampled_splits": overlap,
        "rules_giving_this_threshold": producing,
        "leak_check_passed": leak.get("passed"),
        "split_manifest_matches": manifest == manifest_sha256,
    }
    problems = []
    if not calibration_splits or not hearings:
        problems.append("the calibration artifact records no calibration splits or hearings")
    if "test" in calibration_splits or set(calibration_splits) & set(splits):
        problems.append(
            f"calibrated on {calibration_splits}, which includes test or a sampled split"
        )
    if overlap:
        problems.append(f"{len(overlap)} calibration hearings belong to the sampled splits")
    if not producing:
        problems.append(f"no rule of the calibration artifact gives the threshold {threshold}")
    if leak.get("passed") is not True:
        problems.append("the leak check of the calibration artifact did not pass")
    if manifest != manifest_sha256:
        problems.append("the calibration artifact was built on another split manifest")
    return details, problems


def threshold_calibration_check(
    coverage: Record,
    splits: tuple[str, ...],
    split_of: dict[int, str],
    manifest_sha256: str,
    rule: str,
) -> Record:
    """Whether the run's tier boundary was calibrated on hearings the sample never draws."""
    evidence = coverage["config"]["evidence"]
    threshold = evidence["embedding_threshold"]
    source = evidence.get("calibration_source")
    record: Record = {
        "rule": rule,
        "embedding_threshold": threshold,
        "calibration_source": source,
        "calibration_method": evidence.get("calibration_method"),
        "sampled_splits": list(splits),
    }
    path = Path(source) if isinstance(source, str) else None
    if path is None or path.suffix != ".json" or not path.is_file():
        problems = [f"calibration_source {source!r} is not a calibration artifact file"]
    else:
        with open(path) as f:
            artifact = json.load(f)
        details, problems = calibration_artifact_problems(
            artifact, threshold, splits, split_of, manifest_sha256
        )
        record |= {"calibration_artifact_sha256": sha256_of_file(path), **details}
    return {**record, "problems": problems, "passed": not problems}


def run_paths(config: ValidationConfig, run_name: str) -> tuple[Path, Path]:
    return config.udv_dir / f"{run_name}.jsonl", config.udv_dir / f"{run_name}_coverage.json"


def load_coverage(config: ValidationConfig, run_name: str) -> Record:
    run_path, coverage_path = run_paths(config, run_name)
    for path in (run_path, coverage_path):
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
    with open(coverage_path) as f:
        coverage: Record = json.load(f)
    if coverage["config"]["dataset"]["sha256"] != config.lds_sha256:
        raise SystemExit(f"{coverage_path} was built from another LDS file")
    return coverage


def load_run(
    config: ValidationConfig, run_name: str, split_of: dict[int, str], readable: set[str]
) -> tuple[list[Record], Record]:
    """The UDVs of the run in the readable splits, and the run's source record."""
    run_path, coverage_path = run_paths(config, run_name)
    coverage = load_coverage(config, run_name)
    all_records = load_jsonl(run_path)
    missing = sorted({r["hearing_id"] for r in all_records if r["hearing_id"] not in split_of})
    if missing:
        raise SystemExit(f"hearings of the run missing from the split manifest: {missing}")
    records = [record for record in all_records if split_of[record["hearing_id"]] in readable]
    source = {
        "run_name": run_name,
        "path": str(run_path),
        "sha256": sha256_of_file(run_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": sha256_of_file(coverage_path),
        "coverage_created_at": coverage.get("created_at"),
        "records_in_run": len(all_records),
        "records_read": len(records),
        "embedding_threshold": coverage["config"]["evidence"]["embedding_threshold"],
        "encoders": sorted(
            {f"{r['method']['encoder']}@{r['method']['revision']}" for r in records}
        ),
    }
    return records, source
