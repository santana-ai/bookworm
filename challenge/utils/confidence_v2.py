import argparse
import json
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    confidence_policies,
    grounding_models,
    grounding_scorers,
    translation,
    udv_verifier,
)
from utils import nli_verifier_experiments as experiments
from utils import nli_verifier_exploration as exploration
from utils.calibrate_threshold import interval, rounded
from utils.confidence_policies import signal_metrics
from utils.dataset_io import load_jsonl, module_path, sha256_of_file, write_json, write_jsonl
from utils.decision_models import DecisionQuestion, LayaDecisionModel, noul_question
from utils.decision_scoring import bootstrap_p_value
from utils.grounding_scorers import (
    CandidateSpec,
    PairScorer,
    ScoreCache,
    cached_scores,
    normalized_ranks,
    parse_candidate,
)
from utils.nli_verifier_experiments import PremiseUnit
from utils.retrieval_stats import holm

Record = dict[str, Any]

EA_SPLITS = ("train", "validation")
COVERAGES = (0.8, 0.9)
REPORTED_METRICS = (
    "roc_auc",
    "average_precision",
    "aurc",
    "e_aurc",
    "precision_at_0.8",
    "precision_at_0.9",
)
COMPARED_METRICS = ("roc_auc", "aurc")
REFERENCE = "cosine_serafim"
PRIMARY = "e3x_primary"
EXISTING = ("cosine_serafim", "e3x_primary", "laya_multi_pt_p4", "xnli_mdeberta")
COMBINATIONS = ("rank_mean", "rank_max")
EXISTING_FEATURES = {
    "cosine_serafim": "cosine_serafim:cosine:max",
    "laya_multi_pt_p4": "laya_multi_pt:p4_supports:max",
    "xnli_mdeberta": "xnli_mdeberta:entailment:max",
}
POOLS = ("max", "concatenated")
LAYA_RUN = "nli_verifier_v2"
LAYA_AGGREGATES = ("mean", "median", "min")
TRUE_OPTION = "true"
CODE_MODULES = (
    grounding_scorers,
    grounding_models,
    confidence_policies,
    udv_verifier,
    experiments,
    exploration,
    translation,
)


@dataclass(frozen=True)
class Config:
    path: Path
    raw: Record
    name: str
    verifier_config: Path
    exploration_config: Path
    udv_verifier_config: Path
    translation_config: Path
    translation_model: str
    candidates: dict[str, CandidateSpec]
    order: tuple[str, ...]
    smoke_items: int
    smoke_seed: int
    budget_hours: float
    bootstrap_samples: int
    level: float
    seed: int
    output_dir: Path
    cache_dir: Path

    @property
    def scores_dir(self) -> Path:
        return self.output_dir / "scores"

    @property
    def smoke_dir(self) -> Path:
        return self.output_dir / "smoke"


def load_config(path: Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    inputs, new, smoke, evaluation = raw["inputs"], raw["new"], raw["smoke"], raw["evaluation"]
    order = tuple(new["order"])
    candidates = {key: parse_candidate(key, new[key]) for key in order}
    return Config(
        path=path,
        raw=raw,
        name=raw["experiment"]["name"],
        verifier_config=Path(inputs["verifier_config"]),
        exploration_config=Path(inputs["exploration_config"]),
        udv_verifier_config=Path(inputs["udv_verifier_config"]),
        translation_config=Path(inputs["translation_config"]),
        translation_model=inputs["translation_model"],
        candidates=candidates,
        order=order,
        smoke_items=int(smoke["items"]),
        smoke_seed=int(smoke["seed"]),
        budget_hours=float(smoke["budget_hours"]),
        bootstrap_samples=int(evaluation["bootstrap_samples"]),
        level=float(evaluation["confidence_level"]),
        seed=int(evaluation["seed"]),
        output_dir=Path(raw["outputs"]["output_dir"]),
        cache_dir=Path(raw["outputs"]["cache_dir"]),
    )


def code_hashes() -> Record:
    hashes = {f"utils/{module_path(m).name}": sha256_of_file(module_path(m)) for m in CODE_MODULES}
    return {"utils/confidence_v2.py": sha256_of_file(Path(__file__)), **hashes}


def run_record(config: Config) -> Record:
    return {
        "config": {"path": str(config.path), "sha256": sha256_of_file(config.path)},
        "code": code_hashes(),
        "environment": experiments.environment(),
    }


def ea_units(config: Config) -> tuple[list[PremiseUnit], Record]:
    verifier = experiments.load_config(config.verifier_config)
    units, context = experiments.prepare_benchmark_units(verifier, EA_SPLITS, None)
    if {unit.split for unit in units} - set(EA_SPLITS):
        raise SystemExit("E-A units outside train and validation")
    return units, context


def open_store(config: Config, writable: bool = False) -> tuple[Any, Any]:
    translation_config = translation.load_config(config.translation_config)
    store = translation.open_store(translation_config, config.translation_model, writable=writable)
    return translation_config, store


def language_units(units: list[PremiseUnit], spec: CandidateSpec, store: Any) -> list[PremiseUnit]:
    if spec.language == "pt":
        return units
    if store is None:
        raise SystemExit(f"{spec.key}: no translation store opened")
    return experiments.english_units(units, store)


def unit_pairs(unit: PremiseUnit, concatenated: bool) -> tuple[list[str], str | None]:
    items = experiments.distinct_items(unit.items)
    joined = experiments.concatenated_premise(unit, " ", False) if concatenated else None
    return items, joined


def all_pairs(units: list[PremiseUnit], concatenated: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for unit in units:
        items, joined = unit_pairs(unit, concatenated)
        pairs.extend((item, unit.hypothesis) for item in items)
        if joined is not None:
            pairs.append((joined, unit.hypothesis))
    return pairs


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
    experiments.enforce_offline()
    experiments.seed_everything(0)
    experiments.transformers.logging.set_verbosity_error()
    return experiments.select_device(device)


def smoke_units(units: list[PremiseUnit], count: int, seed: int) -> list[PremiseUnit]:
    train = [unit for unit in units if unit.split == "train"]
    picked = np.random.default_rng(seed).choice(len(train), size=count, replace=False)
    return [train[index] for index in sorted(picked.tolist())]


def full_pair_count(spec: CandidateSpec, ea: list[PremiseUnit], store: Any) -> int:
    return len(set(all_pairs(language_units(ea, spec, store), spec.concatenated)))


def hhem_probe(scorer: PairScorer) -> Record:
    values = [score.value for score in scorer.score(list(grounding_scorers.HHEM_CARD_PAIRS))]
    gaps = [abs(v - e) for v, e in zip(values, grounding_scorers.HHEM_CARD_SCORES, strict=True)]
    return {
        "card_expected": list(grounding_scorers.HHEM_CARD_SCORES),
        "scores": [rounded(value) for value in values],
        "max_abs_gap": rounded(max(gaps)),
        "passed": max(gaps) <= grounding_scorers.HHEM_CARD_TOLERANCE,
    }


def command_translate(args: argparse.Namespace, config: Config) -> None:
    ea, _ = ea_units(config)
    translation_config, store = open_store(config, writable=True)
    texts = experiments.translation_texts(ea, [], store)
    missing = store.missing(texts)
    run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
    if missing:
        device = prepare_device(args.device)
        spec = translation.selected_model(translation_config, config.translation_model)
        translator = translation.load_translator(spec, translation_config.decoding, device)
        run |= translation.translate_missing(
            translator,
            store,
            missing,
            translation_config.progress_every,
            translation_config.decoding.batch_token_budget,
        )
    report = {
        "created_at": experiments.now(),
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
            spec = grounding_scorers.with_model(
                spec, spec.raw["fallback_name"], spec.raw["fallback_revision"]
            )
        record: Record = {"candidate": key, "name": spec.name, "revision": spec.revision}
        try:
            scorer = grounding_models.load_scorer(spec, device)
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
            "created_at": experiments.now(),
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
            "opinion_ids_sha256": grounding_scorers.hashlib.sha256(
                "\n".join(unit.unit_id for unit in units).encode()
            ).hexdigest(),
        }
        if spec.kind == "hhem":
            record["probe"] = hhem_probe(scorer)
        write_json(record | run_record(config), path)
        print(json.dumps({k: record[k] for k in ("pairs_per_second", "projected_hours")}))
        del scorer
        grounding_models.release(device)


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
        "created_at": experiments.now(),
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
        spec = grounding_scorers.with_model(spec, decision["name"], decision["revision"])
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
        scorer = grounding_models.load_scorer(spec, device)
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
            "created_at": experiments.now(),
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
        grounding_models.release(device)


def filled(values: list[float | None], probability: bool) -> np.ndarray:
    present = [value for value in values if value is not None]
    floor = 0.0 if probability else (min(present) if present else 0.0)
    return np.array([floor if value is None else value for value in values], dtype=np.float64)


def new_scores(
    config: Config, key: str, split: str, ids: list[str], pool: str
) -> tuple[np.ndarray, Record]:
    path = config.scores_dir / f"{key}_{split}.jsonl"
    rows = {row["id"]: row for row in load_jsonl(path)}
    if set(rows) != set(ids):
        raise SystemExit(f"{path}: ids differ from the evaluated set")
    values = [rows[unit]["scores"][pool] for unit in ids]
    spec = config.candidates[key]
    source = {"path": str(path), "sha256": sha256_of_file(path), "pool": pool}
    return filled(values, spec.probability), source


def with_combinations(systems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cosine = normalized_ranks(systems[REFERENCE])
    primary = normalized_ranks(systems[PRIMARY])
    return {
        **systems,
        "rank_mean": (cosine + primary) / 2,
        "rank_max": np.maximum(cosine, primary),
    }


def point_metrics(scores: np.ndarray, positive: np.ndarray) -> Record:
    metrics = signal_metrics(scores, positive, COVERAGES)
    return {name: metrics[name] for name in REPORTED_METRICS}


def replicate_metrics(
    systems: dict[str, np.ndarray], positive: np.ndarray, draws: list[np.ndarray]
) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in REPORTED_METRICS} for name in systems
    }
    for rows in draws:
        labels = positive[rows]
        single = labels.all() or not labels.any()
        for name, scores in systems.items():
            metrics = None if single else point_metrics(scores[rows], labels)
            for metric in REPORTED_METRICS:
                values[name][metric].append(np.nan if metrics is None else metrics[metric])
    return {
        name: {metric: np.array(series) for metric, series in by_metric.items()}
        for name, by_metric in values.items()
    }


def finite(values: np.ndarray) -> list[float]:
    return values[np.isfinite(values)].tolist()


def system_table(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    replicates: dict[str, dict[str, np.ndarray]],
    level: float,
) -> Record:
    table: Record = {}
    for name, scores in systems.items():
        metrics = point_metrics(scores, positive)
        table[name] = {
            metric: {
                "point": rounded(metrics[metric]),
                "interval": interval(finite(replicates[name][metric]), level),
            }
            for metric in REPORTED_METRICS
        }
        table[name]["ties"] = int(len(scores) - len(np.unique(scores)))
    return table


def paired_comparisons(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    replicates: dict[str, dict[str, np.ndarray]],
    pairs: list[tuple[str, str]],
    level: float,
) -> list[Record]:
    entries: list[Record] = []
    for name, reference in pairs:
        metrics = point_metrics(systems[name], positive)
        base = point_metrics(systems[reference], positive)
        for metric in COMPARED_METRICS:
            deltas = replicates[name][metric] - replicates[reference][metric]
            valid = deltas[np.isfinite(deltas)]
            entries.append(
                {
                    "candidate": name,
                    "reference": reference,
                    "metric": metric,
                    "delta": rounded(metrics[metric] - base[metric]),
                    "interval": interval(valid.tolist(), level),
                    "p_value": bootstrap_p_value(valid),
                }
            )
    if not entries:
        return entries
    adjusted = holm([float(entry["p_value"]) for entry in entries])
    for entry, value in zip(entries, adjusted, strict=True):
        entry["p_holm"] = rounded(value)
        entry["p_value"] = rounded(entry["p_value"])
    return entries


def evaluate_set(
    systems: dict[str, np.ndarray],
    positive: np.ndarray,
    hearings: np.ndarray,
    samples: int,
    seed: int,
    level: float,
) -> Record:
    draws = experiments.hearing_draws(hearings, samples, np.random.default_rng(seed))
    replicates = replicate_metrics(systems, positive, draws)
    return {
        "items": int(len(positive)),
        "hearings": int(len(set(hearings.tolist()))),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "systems": system_table(systems, positive, replicates, level),
        "bootstrap": f"{samples} hearing resamples, seed {seed}, percentile interval; replicates "
        "with a single class are skipped",
        "_replicates": replicates,
    }


def subset_table(table: Record, names: list[str]) -> Record:
    return {name: table[name] for name in names if name in table}


def exploration_data(
    expl: exploration.ExplorationConfig, keys: tuple[str, ...], split: str, ids: list[str]
) -> dict[str, exploration.ScorerData]:
    data = {}
    for key in keys:
        kind, run = expl.scorers[key]
        scorer = exploration.load_scorer(key, kind, run, split, ids, expl)
        if scorer is None:
            raise SystemExit(f"{key}: no {split} score file in run {run}")
        data[key] = scorer
    return data


def fitted_primary(
    config: Config,
) -> tuple[udv_verifier.FittedPrimary, exploration.ExplorationConfig, Record]:
    udv_config = udv_verifier.load_config(config.udv_verifier_config)
    expl = exploration.load_exploration_config(udv_config.exploration_config)
    candidate = udv_verifier.primary_candidate(expl, udv_config.primary)
    with open(udv_config.selection_path) as f:
        selection = json.load(f)
    with open(udv_config.final_test_path) as f:
        final_test = json.load(f)
    scorers = udv_verifier.candidate_scorers(candidate)
    labels, hearings, data, files = udv_verifier.load_fit_data(expl, scorers, selection)
    fitted = udv_verifier.fit_primary(candidate, labels, hearings, data, expl)
    check = udv_verifier.refit_check(fitted, final_test, udv_config.primary)
    return fitted, expl, {"refit_check": check, "fit_files": files, "key": udv_config.primary}


def ea_existing(
    split: str,
    ids: list[str],
    fitted: udv_verifier.FittedPrimary,
    expl: exploration.ExplorationConfig,
) -> tuple[dict[str, np.ndarray], Record]:
    scorer_keys = tuple(
        dict.fromkeys(
            [
                *udv_verifier.candidate_scorers(fitted.candidate),
                *(feature.split(":")[0] for feature in EXISTING_FEATURES.values()),
            ]
        )
    )
    data = exploration_data(expl, scorer_keys, split, ids)
    systems = {
        name: exploration.candidate_scores(exploration.feature_candidate(feature), data)
        for name, feature in EXISTING_FEATURES.items()
    }
    systems[PRIMARY] = fitted.probabilities(data)
    sources = {
        key: {"path": str(d.path), "sha256": sha256_of_file(d.path)} for key, d in data.items()
    }
    return {name: systems[name] for name in EXISTING}, sources


def available_candidates(config: Config, splits: tuple[str, ...]) -> tuple[list[str], Record]:
    present, absent = [], {}
    for key in config.order:
        paths = [config.scores_dir / f"{key}_{split}.jsonl" for split in splits]
        if all(path.exists() for path in paths):
            present.append(key)
        else:
            absent[key] = "no score file (dropped at the smoke or not run)"
    return present, absent


def literature_scores(
    config: Config, candidates: list[str], split: str, ids: list[str], pool: str
) -> tuple[dict[str, np.ndarray], Record]:
    systems: dict[str, np.ndarray] = {}
    sources: Record = {}
    for key in candidates:
        if pool == "concatenated" and not config.candidates[key].concatenated:
            continue
        systems[key], sources[key] = new_scores(config, key, split, ids, pool)
    return systems, sources


def split_extra(
    extra: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    spreads = {name: values for name, values in extra.items() if "_spread_" in name}
    others = {name: values for name, values in extra.items() if "_spread_" not in name}
    return others, spreads


def added_value_pairs(extra: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    return [
        (name, name.removesuffix("_q11") + "_q7")
        for name in extra
        if name.endswith("_q11") and "_spread_" not in name
    ]


def spread_thresholds(config: Config) -> tuple[dict[str, float], Record]:
    udv_config = udv_verifier.load_config(config.udv_verifier_config)
    expl = exploration.load_exploration_config(udv_config.exploration_config)
    ids, _, _ = exploration.load_labels(expl, "train")
    _, extra, sources = laya_systems(config, "train", ids, "max")
    _, spreads = split_extra(extra)
    return {name: float(np.median(values)) for name, values in spreads.items()}, sources


def spread_section(
    spreads: dict[str, np.ndarray],
    thresholds: dict[str, float],
    positive: np.ndarray,
    hearings: np.ndarray,
    table: Record,
    config: Config,
) -> Record:
    return {
        name: {
            "as_signal": table[f"neg_{name}"],
            "flag": spread_flag_table(
                values,
                thresholds[name],
                positive,
                hearings,
                config.bootstrap_samples,
                config.seed,
                config.level,
            ),
            "median": rounded(float(np.median(values))),
        }
        for name, values in spreads.items()
        if name in thresholds
    }


def evaluate_all(
    config: Config,
    base: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    positive: np.ndarray,
    hearings: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Record]:
    systems = with_combinations(base)
    others, spreads = split_extra(extra)
    signals = {**systems, **others, **{f"neg_{name}": -values for name, values in spreads.items()}}
    result = evaluate_set(
        signals, positive, hearings, config.bootstrap_samples, config.seed, config.level
    )
    return systems, spreads, result


def command_evaluate_ea(args: argparse.Namespace, config: Config) -> None:
    path = config.output_dir / "ea_report.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    started = time.perf_counter()
    fitted, expl, primary_record = fitted_primary(config)
    candidates, absent = available_candidates(config, EA_SPLITS)
    thresholds, _ = spread_thresholds(config)
    results: Record = {}
    inputs: Record = {}
    for split in EA_SPLITS:
        ids, labels, hearings = exploration.load_labels(expl, split)
        existing, sources = ea_existing(split, ids, fitted, expl)
        laya_main, laya_extra, laya_sources = laya_systems(config, split, ids, "max")
        literature, literature_sources = literature_scores(config, candidates, split, ids, "max")
        base = {**existing, **laya_main, **literature}
        systems, spreads, result = evaluate_all(config, base, laya_extra, labels, hearings)
        table, replicates = result["systems"], result["_replicates"]
        family = [(name, REFERENCE) for name in base if name != REFERENCE]
        combos = [(name, REFERENCE) for name in COMBINATIONS]
        extra_names = [name for name in laya_extra if "_spread_" not in name]
        joined_laya, joined_extra, _ = laya_systems(config, split, ids, "concatenated")
        joined_literature, _ = literature_scores(config, candidates, split, ids, "concatenated")
        joined = {**joined_literature, **joined_laya, **split_extra(joined_extra)[0]}
        signals = {**systems, **{name: laya_extra[name] for name in extra_names}}
        results[split] = {
            "items": result["items"],
            "hearings": result["hearings"],
            "positives": result["positives"],
            "negatives": result["negatives"],
            "systems": subset_table(table, list(systems)),
            "comparisons": paired_comparisons(signals, labels, replicates, family, config.level),
            "combination_comparisons": paired_comparisons(
                signals, labels, replicates, combos, config.level
            ),
            "added_questions": {
                "systems": subset_table(table, extra_names),
                "comparisons": paired_comparisons(
                    signals, labels, replicates, added_value_pairs(laya_extra), config.level
                ),
            },
            "spread": spread_section(spreads, thresholds, labels, hearings, table, config),
            "concatenated": {name: point_metrics(v, labels) for name, v in joined.items()},
            "in_sample": [PRIMARY, *COMBINATIONS] if split == "train" else [],
            "bootstrap": result["bootstrap"],
        }
        if split == "train":
            results[split]["note"] = (
                "descriptive: the Holm families are the validation ones; the E3x primary, and "
                "the rank combinations that use it, are in-sample on train"
            )
        inputs[split] = {**sources, **laya_sources, **literature_sources}
    report = {
        "experiment": config.name,
        "part": "E-A",
        "created_at": experiments.now(),
        "splits_read": list(EA_SPLITS),
        "label": config.raw["evaluation"]["positive_ea"],
        "label_semantics": config.raw["experiment"]["label_semantics"],
        "premise": {
            "max": config.raw["premise"]["ea_max"],
            "concatenated": config.raw["premise"]["ea_concatenated"],
            "domain_shift": config.raw["premise"]["domain_shift"],
        },
        "family": config.raw["evaluation"]["ea_family"],
        "laya": {
            key: config.raw["laya"][key]
            for key in ("support_rule", "aggregate_rule", "spread_rule")
        },
        "spread_thresholds_train": {name: rounded(value) for name, value in thresholds.items()},
        "candidates": {"literature_scored": candidates, "absent": absent},
        "primary": primary_record,
        "results": results,
        "inputs": inputs,
        "seconds": round(time.perf_counter() - started, 3),
        **run_record(config),
    }
    write_json(report, path)
    print_table(results["validation"], "E-A validation")


def laya_questions(config: Config, scorer: str) -> list[DecisionQuestion]:
    table = config.raw["laya"]["new_questions"]
    language = table["language_of"][scorer]
    return [noul_question(key, table[language][key]) for key in table["keys"]]


def laya_spec(config: Config, scorer: str) -> tuple[Any, experiments.ScorerSpec]:
    verifier = experiments.load_config(config.verifier_config)
    spec = verifier.scorers[scorer]
    if spec.kind != "laya":
        raise SystemExit(f"{scorer} is not a Laya scorer")
    if spec.language == "en" and spec.translation_model != config.translation_model:
        raise SystemExit(f"{scorer} reads another translation model")
    return verifier, spec


def laya_units(
    config: Config, units: list[PremiseUnit], spec: experiments.ScorerSpec, store: Any
) -> list[PremiseUnit]:
    if spec.language == "pt":
        return units
    return experiments.english_units(units, store)


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
        joined = experiments.concatenated_premise(unit, " ", False) if concatenated else None
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
    requests = experiments.decision_requests(spec_units, " ", concatenated, False)
    model = LayaDecisionModel(experiments.laya_spec(spec, verifier, device))
    answers, seconds = experiments.run_decision(model, questions, requests, scorer, True)
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
    grounding_models.release(device)
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
        requests = experiments.decision_requests(
            laya_units(config, ea, spec, store), " ", True, False
        )
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
            "created_at": experiments.now(),
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
            "created_at": experiments.now(),
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
    path = exploration.SCORES_ROOT / LAYA_RUN / "scores" / f"{scorer}_{split}.jsonl"
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
    draws = experiments.hearing_draws(hearings, samples, np.random.default_rng(seed))

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
        "interval": interval(finite(np.array(differences)), level),
    }


def print_table(result: Record, title: str) -> None:
    print(f"== {title}: {result['items']} items, {result['negatives']} negatives", flush=True)
    for name, metrics in result["systems"].items():
        cells = []
        for metric in ("roc_auc", "aurc", "precision_at_0.8", "precision_at_0.9"):
            entry = metrics[metric]
            span = entry["interval"]
            cells.append(f"{metric} {entry['point']:.3f} [{span.get('low')}, {span.get('high')}]")
        print(f"{name:20s} " + " | ".join(cells), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence signals for UDV evidence: literature grounding scorers against "
        "the serafim cosine and the E3x primary, on the train and validation opinions of the "
        "NLI benchmark (E-A)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/confidence_v2.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("translate", "smoke", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=experiments.DEVICES, default="mps")
        if name != "translate":
            command.add_argument("--candidates", nargs="+", default=None)
        if name == "smoke":
            command.add_argument("--fallback", action="store_true")
    for name in ("laya-smoke", "laya-score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=experiments.DEVICES, default="mps")
        command.add_argument("--scorers", nargs="+", default=None)
    commands.add_parser("decide")
    commands.add_parser("evaluate-ea")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {
        "translate": command_translate,
        "smoke": command_smoke,
        "decide": command_decide,
        "laya-smoke": command_laya_smoke,
        "laya-score": command_laya_score,
        "score": command_score,
        "evaluate-ea": command_evaluate_ea,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
