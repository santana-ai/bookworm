"""The translate, smoke, decide and score commands of the literature grounding scorers."""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import transformers
from bookworm import sha256_of_file, write_json, write_jsonl

from experiments.common.hub_offline import enforce_offline
from experiments.common.reporting import rounded, utc_timestamp
from experiments.common.udv_run import seed_everything, select_device
from experiments.verifier.grounding.config import EA_SPLITS, Config
from experiments.verifier.grounding.models import load_scorer as load_grounding_scorer
from experiments.verifier.grounding.models import release
from experiments.verifier.grounding.provenance import run_record
from experiments.verifier.grounding.scorers import (
    HHEM_CARD_PAIRS,
    HHEM_CARD_SCORES,
    HHEM_CARD_TOLERANCE,
    CandidateSpec,
    PairScorer,
    ScoreCache,
    cached_scores,
    with_model,
)
from experiments.verifier.grounding.units import (
    all_pairs,
    ea_units,
    language_units,
    open_store,
    unit_pairs,
)
from experiments.verifier.nli.benchmark import PremiseUnit
from experiments.verifier.nli.translated import translation_texts
from experiments.verifier.translate.seq2seq import load_translator, translate_missing
from experiments.verifier.translate.store import selected_model

Record = dict[str, Any]


def score_rows(
    scorer: PairScorer,
    units: list[PremiseUnit],
    concatenated: bool,
    cache: ScoreCache | None,
    progress: Callable[[int, int], None] | None = None,
) -> list[Record]:
    pairs = all_pairs(units, concatenated)
    scored = cached_scores(scorer, pairs, cache, progress=progress)
    lookup = dict(zip(pairs, scored, strict=True))
    rows = []
    for unit in units:
        items, joined = unit_pairs(unit, concatenated)
        item_records = [lookup[(item, unit.hypothesis)] if item else None for item in unit.items]
        values = [lookup[(item, unit.hypothesis)]["value"] for item in items]
        joined_record = lookup[(joined, unit.hypothesis)] if joined is not None else None
        if joined_record is not None:
            joined_value = joined_record["value"]
        elif concatenated and len(values) == 1:
            joined_value = values[0]
        else:
            joined_value = None
        rows.append(
            {
                "id": unit.unit_id,
                "hearing_id": unit.hearing_id,
                "split": unit.split,
                "item_count": len(unit.items),
                "nonempty_items": len(items),
                "items": item_records,
                "concatenated": joined_record,
                "scores": {"max": max(values) if values else None, "concatenated": joined_value},
            }
        )
    return rows


def truncation_summary(rows: list[Record]) -> Record:
    item_records = [item for row in rows for item in row["items"] if item]
    joined = [row["concatenated"] for row in rows if row["concatenated"]]
    summary: Record = {}
    for name, records in (("items", item_records), ("concatenated", joined)):
        tokens = [record["tokens"] for record in records]
        summary[name] = {
            "pairs": len(records),
            "truncated": sum(1 for record in records if record["truncated"]),
            "max_tokens": max(tokens) if tokens else None,
            "median_tokens": float(np.median(tokens)) if tokens else None,
        }
        masses = [record["answer_mass"] for record in records if "answer_mass" in record]
        if masses:
            summary[name]["answer_mass"] = {
                "mean": rounded(float(np.mean(masses))),
                "min": rounded(float(np.min(masses))),
                "below_0.5": sum(1 for mass in masses if mass < 0.5),
            }
    return summary


def cache_for(config: Config, scorer: PairScorer, spec: CandidateSpec, device: str) -> ScoreCache:
    path = config.cache_dir / f"{spec.key}_{spec.revision[:12]}_{device}.jsonl"
    return ScoreCache(path, scorer.signature)


def progress_printer(key: str) -> Callable[[int, int], None]:
    started = time.perf_counter()

    def report(done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate / 60 if rate else float("nan")
        print(f"[{key}] {done}/{total} pairs, {rate:.2f}/s, eta {eta:.1f} min", flush=True)

    return report


def prepare_device(device: str) -> str:
    enforce_offline()
    seed_everything(0)
    transformers.logging.set_verbosity_error()
    return select_device(device)


def smoke_units(units: list[PremiseUnit], count: int, seed: int) -> list[PremiseUnit]:
    train = [unit for unit in units if unit.split == "train"]
    picked = np.random.default_rng(seed).choice(len(train), size=count, replace=False)
    return [train[index] for index in sorted(picked.tolist())]


def full_pair_count(spec: CandidateSpec, ea: list[PremiseUnit], store: Any) -> int:
    return len(set(all_pairs(language_units(ea, spec, store), spec.concatenated)))


def hhem_probe(scorer: PairScorer) -> Record:
    values = [score.value for score in scorer.score(list(HHEM_CARD_PAIRS))]
    gaps = [abs(v - e) for v, e in zip(values, HHEM_CARD_SCORES, strict=True)]
    return {
        "card_expected": list(HHEM_CARD_SCORES),
        "scores": [rounded(value) for value in values],
        "max_abs_gap": rounded(max(gaps)),
        "passed": max(gaps) <= HHEM_CARD_TOLERANCE,
    }


def command_translate(args: argparse.Namespace, config: Config) -> None:
    ea, _ = ea_units(config)
    translation_config, store = open_store(config, writable=True)
    texts = translation_texts(ea, [], store)
    missing = store.missing(texts)
    run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
    if missing:
        device = prepare_device(args.device)
        spec = selected_model(translation_config, config.translation_model)
        translator = load_translator(spec, translation_config.decoding, device)
        run |= translate_missing(
            translator,
            store,
            missing,
            translation_config.progress_every,
            translation_config.decoding.batch_token_budget,
        )
    report = {
        "created_at": utc_timestamp(),
        "model": config.translation_model,
        "run": run,
        "store": store.summary(),
        "still_missing": len(store.missing(texts)),
        **run_record(config),
    }
    write_json(report, config.output_dir / "translate_report.json")
    print(json.dumps(run, indent=2), flush=True)


def command_smoke(args: argparse.Namespace, config: Config) -> None:
    ea, _ = ea_units(config)
    units = smoke_units(ea, config.smoke_items, config.smoke_seed)
    _, store = open_store(config)
    device = prepare_device(args.device)
    for key in args.candidates or config.order:
        spec = config.candidates[key]
        path = config.smoke_dir / f"{key}.json"
        if path.exists():
            raise SystemExit(f"{path} exists: the smoke is run once per candidate")
        if args.fallback:
            spec = with_model(spec, spec.raw["fallback_name"], spec.raw["fallback_revision"])
        record: Record = {"candidate": key, "name": spec.name, "revision": spec.revision}
        try:
            scorer = load_grounding_scorer(spec, device)
        except Exception as error:
            record |= {"status": "load_failed", "error": repr(error)[:2000]}
            write_json(record | run_record(config), path)
            print(f"[{key}] load failed: {error!r}", flush=True)
            continue
        spec_units = language_units(units, spec, store)
        cache = cache_for(config, scorer, spec, device)
        started = time.perf_counter()
        rows = score_rows(scorer, spec_units, spec.concatenated, cache)
        seconds = time.perf_counter() - started
        if cache.computed == 0:
            raise SystemExit(f"{key}: every smoke pair was cached, no throughput to measure")
        per_pair = seconds / cache.computed
        full = full_pair_count(spec, ea, store)
        record |= {
            "status": "scored",
            "created_at": utc_timestamp(),
            "device": device,
            "model": scorer.info,
            "opinions": len(rows),
            "pairs_computed": cache.computed,
            "cache_hits": cache.hits,
            "seconds": round(seconds, 3),
            "pairs_per_second": round(1 / per_pair, 3),
            "full_pairs": full,
            "projected_hours": round(full * per_pair / 3600, 3),
            "truncation": truncation_summary(rows),
            "opinion_ids_sha256": hashlib.sha256(
                "\n".join(unit.unit_id for unit in units).encode()
            ).hexdigest(),
        }
        if spec.kind == "hhem":
            record["probe"] = hhem_probe(scorer)
        write_json(record | run_record(config), path)
        print(json.dumps({k: record[k] for k in ("pairs_per_second", "projected_hours")}))
        del scorer
        release(device)


def smoke_decision(record: Record, spec: CandidateSpec | None, budget: float) -> tuple[str, str]:
    if record["status"] != "scored":
        return "dropped", f"load failed: {record.get('error', '')[:300]}"
    probe = record.get("probe")
    if probe is not None and not probe["passed"]:
        return "dropped", f"probe failed (max gap {probe['max_abs_gap']})"
    if spec is not None and spec.kind in ("llm_judge", "granite_guardian"):
        masses = [
            part["answer_mass"]["mean"]
            for part in record["truncation"].values()
            if "answer_mass" in part
        ]
        if masses and min(masses) <= 0.5:
            return "dropped", f"Yes/No mass {min(masses)} at most 0.5"
    if record["projected_hours"] > budget:
        return "dropped", f"projected {record['projected_hours']} h over the {budget} h budget"
    return "run", f"projected {record['projected_hours']} h within the {budget} h budget"


def command_decide(args: argparse.Namespace, config: Config) -> None:
    decisions: Record = {}
    laya_keys = [f"laya_new_{scorer}" for scorer in config.raw["laya"]["scorers"]]
    for key in [*config.order, *laya_keys]:
        path = config.smoke_dir / f"{key}.json"
        if not path.exists():
            decisions[key] = {"status": "not_smoked"}
            continue
        with open(path) as f:
            record = json.load(f)
        spec = config.candidates.get(key)
        status, reason = smoke_decision(record, spec, config.budget_hours)
        decisions[key] = {
            "status": status,
            "reason": reason,
            "name": record["name"],
            "revision": record["revision"],
            "pairs_per_second": record.get("pairs_per_second"),
            "projected_hours": record.get("projected_hours"),
            "smoke_sha256": sha256_of_file(path),
        }
    report = {
        "created_at": utc_timestamp(),
        "rule": config.raw["smoke"]["decision_rule"],
        "budget_hours": config.budget_hours,
        "decisions": decisions,
        **run_record(config),
    }
    write_json(report, config.smoke_dir / "decision.json")
    print(json.dumps(decisions, indent=2), flush=True)


def decided_spec(config: Config, key: str) -> CandidateSpec:
    path = config.smoke_dir / "decision.json"
    if not path.exists():
        raise SystemExit("run smoke and decide before score")
    with open(path) as f:
        decision = json.load(f)["decisions"].get(key)
    if not decision or decision["status"] != "run":
        raise SystemExit(f"{key}: not decided to run ({decision})")
    spec = config.candidates[key]
    if decision["name"] != spec.name:
        spec = with_model(spec, decision["name"], decision["revision"])
    return spec


def split_files(config: Config, key: str) -> dict[str, Path]:
    return {split: config.scores_dir / f"{key}_{split}.jsonl" for split in EA_SPLITS}


def command_score(args: argparse.Namespace, config: Config) -> None:
    _, store = open_store(config)
    device = prepare_device(args.device)
    units, context = ea_units(config)
    for key in args.candidates or config.order:
        spec = decided_spec(config, key)
        report_path = config.scores_dir / f"{key}_ea_report.json"
        if report_path.exists():
            raise SystemExit(f"{report_path} exists")
        spec_units = language_units(units, spec, store)
        started = time.perf_counter()
        scorer = load_grounding_scorer(spec, device)
        cache = cache_for(config, scorer, spec, device)
        rows = score_rows(scorer, spec_units, spec.concatenated, cache, progress_printer(key))
        seconds = time.perf_counter() - started
        written: Record = {}
        for split, path in split_files(config, key).items():
            split_rows = [r for r in rows if r["split"] == split]
            write_jsonl(split_rows, path)
            written[split] = {"path": str(path), "rows": len(split_rows)}
            written[split]["sha256"] = sha256_of_file(path)
        report = {
            "created_at": utc_timestamp(),
            "candidate": key,
            "set": "ea",
            "model": scorer.info,
            "language": spec.language,
            "concatenated": spec.concatenated,
            "units": len(rows),
            "cache": {"path": str(cache.path), "hits": cache.hits, "computed": cache.computed},
            "seconds": round(seconds, 3),
            "truncation": truncation_summary(rows),
            "files": written,
            "context": context,
            "translation": store.summary() if spec.language == "en" else None,
            **run_record(config),
        }
        write_json(report, report_path)
        print(f"[{key}] ea: {len(rows)} units in {seconds:.1f} s", flush=True)
        del scorer
        release(device)
