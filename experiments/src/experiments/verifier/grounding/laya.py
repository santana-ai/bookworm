"""The laya-smoke and laya-score commands and the Laya battery systems."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file, write_json, write_jsonl

from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.decision_models import DecisionQuestion, LayaDecisionModel, noul_question
from experiments.verifier.exploration import scores as exploration_scores
from experiments.verifier.grounding.config import LAYA_AGGREGATES, LAYA_RUN, TRUE_OPTION, Config
from experiments.verifier.grounding.models import release
from experiments.verifier.grounding.provenance import run_record
from experiments.verifier.grounding.scoring import prepare_device, smoke_units, split_files
from experiments.verifier.grounding.units import ea_units, open_store
from experiments.verifier.nli.benchmark import PremiseUnit, concatenated_premise
from experiments.verifier.nli.config import ScorerSpec
from experiments.verifier.nli.config import load_config as load_verifier_config
from experiments.verifier.nli.decision import decision_requests, run_decision
from experiments.verifier.nli.decision import laya_spec as verifier_laya_spec
from experiments.verifier.nli.translated import english_units
from experiments.verifier.runtime import now
from experiments.verifier.stats import finite_interval, hearing_draws

Record = dict[str, Any]


def laya_questions(config: Config, scorer: str) -> list[DecisionQuestion]:
    table = config.raw["laya"]["new_questions"]
    language = table["language_of"][scorer]
    return [noul_question(key, table[language][key]) for key in table["keys"]]


def laya_spec(config: Config, scorer: str) -> tuple[Any, ScorerSpec]:
    verifier = load_verifier_config(config.verifier_config)
    spec = verifier.scorers[scorer]
    if spec.kind != "laya":
        raise SystemExit(f"{scorer} is not a Laya scorer")
    if spec.language == "en" and spec.translation_model != config.translation_model:
        raise SystemExit(f"{scorer} reads another translation model")
    return verifier, spec


def laya_units(
    config: Config, units: list[PremiseUnit], spec: ScorerSpec, store: Any
) -> list[PremiseUnit]:
    if spec.language == "pt":
        return units
    return english_units(units, store)


def laya_rows(
    units: list[PremiseUnit],
    answers: dict[tuple[str, str], dict[str, Any]],
    keys: list[str],
    concatenated: bool,
) -> list[Record]:
    def signals(premise: str, hypothesis: str) -> Record:
        found = answers[(premise, hypothesis)]
        return {key: float(found[key].probabilities[TRUE_OPTION]) for key in keys}

    rows = []
    for unit in units:
        items = [
            {"signals": signals(item, unit.hypothesis)} if item else None for item in unit.items
        ]
        joined = concatenated_premise(unit, " ", False) if concatenated else None
        rows.append(
            {
                "id": unit.unit_id,
                "hearing_id": unit.hearing_id,
                "split": unit.split,
                "items": items,
                "concatenated": {"signals": signals(joined, unit.hypothesis)} if joined else None,
            }
        )
    return rows


def run_laya(
    config: Config, scorer: str, units: list[PremiseUnit], concatenated: bool, device: str
) -> tuple[list[Record], Record]:
    verifier, spec = laya_spec(config, scorer)
    _, store = open_store(config)
    spec_units = laya_units(config, units, spec, store)
    questions = laya_questions(config, scorer)
    requests = decision_requests(spec_units, " ", concatenated, False)
    model = LayaDecisionModel(verifier_laya_spec(spec, verifier, device))
    answers, seconds = run_decision(model, questions, requests, scorer, True)
    rows = laya_rows(spec_units, answers, [q.key for q in questions], concatenated)
    record = {
        "scorer": scorer,
        "model": model.describe(),
        "questions": {q.key: q.payload() for q in questions},
        "requests": len(requests),
        "question_count": len(questions),
        "seconds": round(seconds, 3),
        "counters": dict(model.counters),
        "translation": store.summary() if spec.language == "en" else None,
    }
    del model
    release(device)
    return rows, record


def command_laya_smoke(args: argparse.Namespace, config: Config) -> None:
    ea, _ = ea_units(config)
    units = smoke_units(ea, config.smoke_items, config.smoke_seed)
    device = prepare_device(args.device)
    for scorer in args.scorers or config.raw["laya"]["scorers"]:
        path = config.smoke_dir / f"laya_new_{scorer}.json"
        if path.exists():
            raise SystemExit(f"{path} exists: the smoke is run once per scorer")
        rows, record = run_laya(config, scorer, units, True, device)
        computed = record["counters"]["computed"]
        if computed == 0:
            raise SystemExit(f"{scorer}: every smoke request was cached")
        per_request = record["seconds"] / computed
        _, spec = laya_spec(config, scorer)
        _, store = open_store(config)
        requests = decision_requests(laya_units(config, ea, spec, store), " ", True, False)
        full = len(set(requests)) * record["question_count"]
        record |= {
            "status": "scored",
            "name": f"laya_new:{scorer}",
            "revision": spec.revision,
            "opinions": len(rows),
            "pairs_per_second": round(1 / per_request, 3),
            "full_pairs": full,
            "projected_hours": round(full * per_request / 3600, 3),
            "truncation": {},
            "created_at": now(),
        }
        write_json(record | run_record(config), path)
        print(json.dumps({k: record[k] for k in ("pairs_per_second", "projected_hours")}))


def command_laya_score(args: argparse.Namespace, config: Config) -> None:
    device = prepare_device(args.device)
    with open(config.smoke_dir / "decision.json") as f:
        decisions = json.load(f)["decisions"]
    units, context = ea_units(config)
    for scorer in args.scorers or config.raw["laya"]["scorers"]:
        key = f"laya_new_{scorer}"
        if decisions.get(key, {}).get("status") != "run":
            raise SystemExit(f"{key}: not decided to run")
        report_path = config.scores_dir / f"{key}_ea_report.json"
        if report_path.exists():
            raise SystemExit(f"{report_path} exists")
        rows, record = run_laya(config, scorer, units, True, device)
        written: Record = {}
        for split, path in split_files(config, key).items():
            split_rows = [r for r in rows if r["split"] == split]
            write_jsonl(split_rows, path)
            written[split] = {
                "path": str(path),
                "rows": len(split_rows),
                "sha256": sha256_of_file(path),
            }
        report = {
            "created_at": now(),
            "set": "ea",
            **record,
            "files": written,
            "context": context,
            **run_record(config),
        }
        write_json(report, report_path)
        print(f"[{key}] ea: {len(rows)} units in {record['seconds']:.1f} s", flush=True)


def item_signals(rows: dict[str, Record], ids: list[str], pool: str) -> list[list[Record]]:
    lists = []
    for unit in ids:
        row = rows[unit]
        if pool == "concatenated":
            joined = row.get("concatenated")
            if joined:
                lists.append([joined["signals"]])
            else:
                lists.append([item["signals"] for item in row["items"] if item])
        else:
            lists.append([item["signals"] for item in row["items"] if item])
    return lists


def merge_signals(first: list[list[Record]], second: list[list[Record]]) -> list[list[Record]]:
    merged = []
    for items_a, items_b in zip(first, second, strict=True):
        if len(items_a) != len(items_b):
            raise SystemExit("the new-question rows and the battery rows differ in items")
        merged.append([{**a, **b} for a, b in zip(items_a, items_b, strict=True)])
    return merged


def aggregate_item(signals: Record, questions: list[str], how: str) -> float:
    values = np.array([signals[key] for key in questions], dtype=np.float64)
    if how == "mean":
        return float(values.mean())
    if how == "median":
        return float(np.median(values))
    if how == "min":
        return float(values.min())
    raise ValueError(how)


def aggregate_scores(per_unit: list[list[Record]], questions: list[str], how: str) -> np.ndarray:
    return np.array(
        [
            max((aggregate_item(item, questions, how) for item in items), default=0.0)
            for items in per_unit
        ]
    )


def spread_scores(
    per_unit: list[list[Record]], questions: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    stds, ranges = [], []
    for items in per_unit:
        if not items:
            stds.append(0.0)
            ranges.append(0.0)
            continue
        means = [aggregate_item(item, questions, "mean") for item in items]
        chosen = items[int(np.argmax(means))]
        values = np.array([chosen[key] for key in questions], dtype=np.float64)
        stds.append(float(values.std()))
        ranges.append(float(values.max() - values.min()))
    return np.array(stds), np.array(ranges)


def laya_battery_rows(scorer: str, split: str) -> tuple[dict[str, Record], Path]:
    path = exploration_scores.SCORES_ROOT / LAYA_RUN / "scores" / f"{scorer}_{split}.jsonl"
    return {row["id"]: row for row in load_jsonl(path)}, path


def laya_systems(
    config: Config, split: str, ids: list[str], pool: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Record]:
    laya = config.raw["laya"]
    q7 = list(laya["support_questions"])
    new_keys = list(laya["new_questions"]["keys"])
    systems: dict[str, np.ndarray] = {}
    extra: dict[str, np.ndarray] = {}
    sources: Record = {}
    for scorer in laya["scorers"]:
        rows, path = laya_battery_rows(scorer, split)
        missing = [unit for unit in ids if unit not in rows]
        if missing:
            raise SystemExit(f"{path}: {len(missing)} ids missing")
        battery = item_signals(rows, ids, pool)
        sources[f"{scorer}_battery"] = {"path": str(path), "sha256": sha256_of_file(path)}
        for how in LAYA_AGGREGATES:
            systems[f"{scorer}_{how}_q7"] = aggregate_scores(battery, q7, how)
        std, _ = spread_scores(battery, q7)
        extra[f"{scorer}_spread_q7"] = std
        new_path = config.scores_dir / f"laya_new_{scorer}_{split}.jsonl"
        if not new_path.exists():
            continue
        new_rows = {row["id"]: row for row in load_jsonl(new_path)}
        if set(new_rows) != set(ids):
            raise SystemExit(f"{new_path}: ids differ from the evaluated set")
        merged = merge_signals(battery, item_signals(new_rows, ids, pool))
        sources[f"laya_new_{scorer}"] = {"path": str(new_path), "sha256": sha256_of_file(new_path)}
        for how in LAYA_AGGREGATES:
            extra[f"{scorer}_{how}_q11"] = aggregate_scores(merged, q7 + new_keys, how)
        for key in new_keys:
            extra[f"{scorer}_{key}"] = aggregate_scores(merged, [key], "mean")
        extra[f"{scorer}_new_mean"] = aggregate_scores(merged, new_keys, "mean")
        std, _ = spread_scores(merged, q7 + new_keys)
        extra[f"{scorer}_spread_q11"] = std
    return systems, extra, sources


def spread_flag_table(
    spread: np.ndarray,
    threshold: float,
    positive: np.ndarray,
    hearings: np.ndarray,
    samples: int,
    seed: int,
    level: float,
) -> Record:
    flagged = spread > threshold
    draws = hearing_draws(hearings, samples, np.random.default_rng(seed))

    def negative_rate(mask: np.ndarray) -> float:
        return float((~positive[mask]).mean()) if mask.any() else np.nan

    differences = []
    for rows in draws:
        flag, pos = flagged[rows], positive[rows]
        if flag.all() or not flag.any():
            differences.append(np.nan)
            continue
        differences.append(float((~pos[flag]).mean() - (~pos[~flag]).mean()))
    point = negative_rate(flagged) - negative_rate(~flagged)
    return {
        "threshold": rounded(threshold),
        "flagged": int(flagged.sum()),
        "unflagged": int((~flagged).sum()),
        "negative_rate_flagged": rounded(negative_rate(flagged)),
        "negative_rate_unflagged": rounded(negative_rate(~flagged)),
        "difference": None if np.isnan(point) else rounded(point),
        "interval": finite_interval(np.array(differences), level),
    }
