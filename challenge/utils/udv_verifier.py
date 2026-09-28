import argparse
import csv
import json
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import nli_verifier_experiments as experiments
from utils import nli_verifier_exploration as exploration
from utils.calibrate_threshold import interval, rounded
from utils.dataset_io import load_jsonl, sha256_of_file, write_json, write_jsonl
from utils.nli_verifier_experiments import (
    PremiseUnit,
    ScorerSpec,
    VerifierConfig,
    english_probes,
    english_units,
    max_f1_not_inferable_optimum,
    portuguese_probes,
    score_units,
    translation_texts,
)
from utils.nli_verifier_exploration import (
    MATCH_TOLERANCE,
    Candidate,
    ExplorationConfig,
    ScorerData,
    candidate_scores,
    check_reading,
    feature_candidate,
    feature_matrix,
    fit_learned,
    learned_candidates,
    load_exploration_config,
    load_labels,
    load_scorer,
)
from utils.udv_pipeline import normalize_whitespace

Record = dict[str, Any]
ScoreFunction = Callable[..., tuple[list[Record], Record]]

UDV_SPLIT = "udv"
POOLS = ("max", "mean", "concatenated")
SUPPORT_TYPE_NONE = "none"


@dataclass(frozen=True)
class UdvVerifierConfig:
    path: Path
    raw: Record
    name: str
    udv_path: Path
    verifier_config: Path
    exploration_config: Path
    selection_path: Path
    final_test_path: Path
    evidence_tiers: tuple[str, ...]
    primary: str
    secondary: str
    scorers: tuple[str, ...]
    device: str
    quantiles: tuple[float, ...]
    semantic_tiers: tuple[str, ...]
    lowest_tiers: tuple[str, ...]
    lowest_count: int
    output_path: Path
    report_path: Path
    udv_threshold_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return exploration.SCORES_ROOT / self.name


@dataclass(frozen=True)
class UdvThreshold:
    value: float
    path: Path
    sha256: str
    rule: str
    record: Record


@dataclass(frozen=True)
class FittedPrimary:
    candidate: Candidate
    scaler: StandardScaler
    model: LogisticRegression
    threshold: float
    record: Record

    def probabilities(self, data: dict[str, ScorerData]) -> np.ndarray:
        matrix = feature_matrix(self.candidate, data)
        return self.model.predict_proba(self.scaler.transform(matrix))[:, 1]


def load_config(path: Path) -> UdvVerifierConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["udv_verifier"]
    inputs, pairs, scoring = raw["inputs"], raw["pairs"], raw["scoring"]
    report, outputs = raw["report"], raw["outputs"]
    if outputs["score_run"] != raw["name"]:
        raise SystemExit("outputs.score_run must equal the run name")
    return UdvVerifierConfig(
        path=path,
        raw=raw,
        name=raw["name"],
        udv_path=Path(inputs["udv_path"]),
        verifier_config=Path(inputs["verifier_config"]),
        exploration_config=Path(inputs["exploration_config"]),
        selection_path=Path(inputs["selection_path"]),
        final_test_path=Path(inputs["final_test_path"]),
        evidence_tiers=tuple(pairs["evidence_tiers"]),
        primary=scoring["primary"],
        secondary=scoring["secondary"],
        scorers=tuple(scoring["scorers"]),
        device=scoring["device"],
        quantiles=tuple(float(q) for q in report["quantiles"]),
        semantic_tiers=tuple(report["semantic_tiers"]),
        lowest_tiers=tuple(report["lowest_tiers"]),
        lowest_count=int(report["lowest_count"]),
        output_path=Path(outputs["output_path"]),
        report_path=Path(outputs["report_path"]),
        udv_threshold_path=(
            Path(scoring["udv_threshold_path"]) if "udv_threshold_path" in scoring else None
        ),
    )


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
    split_of, split_source = experiments.load_split_lookup(
        verifier.manifest_path, verifier.lds_sha256
    )
    units, unscored = udv_units(records, config.evidence_tiers, split_of)
    return records, units, unscored, split_of, split_source


def declared_scorers(config: ExplorationConfig) -> dict[str, ScorerData]:
    return {
        key: ScorerData(key, kind, Path(), {}, {}, {}) for key, (kind, _) in config.scorers.items()
    }


def primary_candidate(config: ExplorationConfig, key: str) -> Candidate:
    for candidate in learned_candidates(config, declared_scorers(config)):
        if candidate.key == key:
            return candidate
    raise SystemExit(f"{key} is not a learned candidate of {config.path}")


def candidate_scorers(candidate: Candidate) -> tuple[str, ...]:
    return tuple(dict.fromkeys(feature.split(":")[0] for feature in candidate.features))


def check_scorer_list(config: UdvVerifierConfig, candidate: Candidate) -> None:
    needed = set(candidate_scorers(candidate)) | {config.secondary.split(":")[0]}
    if needed != set(config.scorers):
        raise SystemExit(
            f"scoring.scorers {sorted(config.scorers)} differs from the scorers the primary and "
            f"secondary read {sorted(needed)}"
        )


def selected_specs(config: UdvVerifierConfig, verifier: VerifierConfig) -> list[ScorerSpec]:
    return experiments.selected_scorers(verifier, list(config.scorers))


def english_specs(specs: list[ScorerSpec]) -> list[ScorerSpec]:
    return [spec for spec in specs if spec.language == "en"]


def open_run(config: UdvVerifierConfig) -> tuple[VerifierConfig, ExplorationConfig]:
    verifier = experiments.load_config(config.verifier_config)
    exploration_config = load_exploration_config(config.exploration_config)
    if exploration_config.raw["verifier_config"] != str(config.verifier_config):
        raise SystemExit("the exploration config reads another verifier config")
    check_scorer_list(config, primary_candidate(exploration_config, config.primary))
    return verifier, exploration_config


def prepare_models(verifier: VerifierConfig) -> None:
    if verifier.hf_hub_offline:
        experiments.enforce_offline()
    experiments.seed_everything(verifier.seed)
    experiments.transformers.logging.set_verbosity_error()


def command_translate(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    verifier, _ = open_run(config)
    _, units, _, _, _ = load_udvs(config, verifier)
    specs = english_specs(selected_specs(config, verifier))
    translation_config = experiments.load_translation_config(verifier)
    report: Record = {"created_at": experiments.now(), "models": {}}
    for model_key in dict.fromkeys(str(spec.translation_model) for spec in specs):
        spec = experiments.translation.selected_model(translation_config, model_key)
        store = experiments.translation.open_store(translation_config, model_key, writable=True)
        texts = translation_texts(units, portuguese_probes(verifier), store)
        missing = store.missing(texts)
        run: Record = {"requested": len(texts), "already_cached": len(texts) - len(missing)}
        info = None
        if missing:
            prepare_models(verifier)
            device = experiments.select_device(args.device or config.device)
            translator = experiments.translation.load_translator(
                spec, translation_config.decoding, device
            )
            print(f"{spec.name} on {device}: {len(missing)} texts to translate", flush=True)
            run |= experiments.translation.translate_missing(
                translator,
                store,
                missing,
                translation_config.progress_every,
                translation_config.decoding.batch_token_budget,
            )
            info = translator.info
            del translator
            experiments.release_device(device)
        report["models"][model_key] = {
            "model": experiments.translation.model_record(translation_config, spec),
            "run": run,
            "model_info": info,
            "store": store.summary(),
            "still_missing": len(store.missing(texts)),
        }
    write_json(report, config.run_dir / "translate_report.json")
    print(json.dumps({k: v["run"] for k, v in report["models"].items()}, indent=2), flush=True)


def score_scorers(
    units: list[PremiseUnit],
    specs: list[ScorerSpec],
    verifier: VerifierConfig,
    device: str,
    translations: experiments.Translations | None,
    prefix: str,
    score_function: ScoreFunction = score_units,
) -> dict[str, tuple[list[Record], Record]]:
    results: dict[str, tuple[list[Record], Record]] = {}
    for spec in specs:
        spec_units, probes, extra = units, portuguese_probes(verifier), {}
        if spec.language == "en":
            if translations is None:
                raise SystemExit(f"{spec.key}: no translation store opened")
            store = translations.store(spec)
            spec_units, probes = english_units(units, store), english_probes(verifier, store)
            extra["translation"] = experiments.translation_summary(verifier, translations, spec)
        rows, details = score_function(
            spec_units, spec, verifier, device, True, prefix, probes, "live"
        )
        results[spec.key] = (rows, details | extra)
        print(f"{spec.key}: {len(rows)} UDVs, timing {json.dumps(details['timing'])}", flush=True)
    return results


def write_scores(
    results: dict[str, tuple[list[Record], Record]],
    specs: list[ScorerSpec],
    verifier: VerifierConfig,
    config: UdvVerifierConfig,
    context: Record,
) -> None:
    by_key = {spec.key: spec for spec in specs}
    for key, (rows, details) in results.items():
        spec = by_key[key]
        experiments.check_score_names(rows, spec)
        path = experiments.score_file(config.run_dir, key, UDV_SPLIT)
        write_jsonl(rows, path)
        files = {UDV_SPLIT: {"path": str(path), "rows": len(rows), "sha256": sha256_of_file(path)}}
        splits = {"hearing_splits": dict(sorted(Counter(row["split"] for row in rows).items()))}
        report = experiments.score_report(
            "udv", config.name, spec, rows, details, context, splits, verifier, files
        )
        write_json(report, experiments.score_report_file(config.run_dir, key))


def command_score(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    verifier, _ = open_run(config)
    _, units, _, _, split_source = load_udvs(config, verifier)
    specs = selected_specs(config, verifier)
    translations = experiments.open_translations(verifier, specs, None)
    for spec in specs:
        if spec.kind in experiments.DECISION_KINDS:
            experiments.check_decision_scorer(spec, verifier, "live")
        if spec.language == "en" and translations is not None:
            experiments.check_translations(spec, units, verifier, translations.store(spec))
    prepare_models(verifier)
    device = experiments.select_device(args.device or config.device)
    context = {
        "sources": {
            "udv": {"path": str(config.udv_path), "sha256": sha256_of_file(config.udv_path)},
            "splits": split_source,
        },
        "premise": {"rule": config.raw["pairs"]["premise"], "concatenate_single": True},
        "subset": None,
    }
    print(f"{len(units)} UDVs on {device} -> {config.run_dir}", flush=True)
    results = score_scorers(units, specs, verifier, device, translations, config.name)
    write_scores(results, specs, verifier, config, context)


def load_fit_data(
    config: ExplorationConfig, scorers: tuple[str, ...], selection: Record
) -> tuple[np.ndarray, np.ndarray, dict[str, ScorerData], Record]:
    split = config.raw["fit_split"]
    ids, labels, hearings = load_labels(config, split)
    data: dict[str, ScorerData] = {}
    files: Record = {}
    for key in scorers:
        kind, run = config.scorers[key]
        scorer = load_scorer(key, kind, run, split, ids, config)
        if scorer is None:
            raise SystemExit(f"{key}: no {split} score file in run {run}")
        digest = sha256_of_file(scorer.path)
        recorded = selection["files_read"][key]["sha256"]
        if digest != recorded:
            raise SystemExit(f"{scorer.path}: sha256 differs from the file explore read")
        data[key] = scorer
        files[key] = {"path": str(scorer.path), "sha256": digest, "matches_selection": True}
    return labels, hearings, data, files


def fit_primary(
    candidate: Candidate,
    labels: np.ndarray,
    hearings: np.ndarray,
    data: dict[str, ScorerData],
    config: ExplorationConfig,
) -> FittedPrimary:
    matrix = feature_matrix(candidate, data)
    scaler, model, c, means = fit_learned(matrix, labels, hearings, config)
    fit_scores = model.predict_proba(scaler.transform(matrix))[:, 1]
    optimum = max_f1_not_inferable_optimum(fit_scores, labels)
    if optimum is None:
        raise SystemExit("the train split has a single label: no threshold")
    record = {
        "c": c,
        "inner_means": means,
        "coefficients": dict(zip(candidate.features, map(rounded, model.coef_[0]), strict=True)),
        "intercept": float(model.intercept_[0]),
        "threshold": float(optimum["threshold"]),
        "train_opinions": len(labels),
        "train_not_inferable": int((~labels).sum()),
    }
    return FittedPrimary(candidate, scaler, model, float(optimum["threshold"]), record)


def refit_check(fitted: FittedPrimary, final_test: Record, key: str) -> Record:
    stored = final_test["learned"][key]
    threshold = float(final_test["results"][key]["threshold"])
    checks = {
        "c": fitted.record["c"] == stored["c"],
        "inner_means": fitted.record["inner_means"] == stored["inner_means"],
        "coefficients": fitted.record["coefficients"] == stored["coefficients"],
        "threshold": abs(fitted.threshold - threshold) <= MATCH_TOLERANCE,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise SystemExit(f"the refit differs from final_test.json in {failed}")
    return {
        "compared_with": "final_test.json learned and results entries of the primary",
        "checks": checks,
        "threshold_refit": fitted.threshold,
        "threshold_final_test": threshold,
        "threshold_abs_gap": abs(fitted.threshold - threshold),
    }


def pool_agreement(candidate: Candidate, data: dict[str, ScorerData]) -> Record:
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for feature in candidate.features:
        scorer, signal, pool = feature.split(":")
        groups.setdefault((scorer, signal), {})[pool] = feature
    gaps: dict[str, float] = {}
    for (scorer, signal), pools in groups.items():
        if len(pools) < 2:
            continue
        columns = [candidate_scores(feature_candidate(f), data) for f in pools.values()]
        gap = float(np.max(np.abs(np.ptp(np.column_stack(columns), axis=1))))
        if gap > MATCH_TOLERANCE:
            raise SystemExit(f"{scorer}:{signal}: pools {sorted(pools)} differ by {gap}")
        gaps[scorer] = max(gaps.get(scorer, 0.0), gap)
    return {
        "rule": "with one premise item the max, mean and concatenated pools of a signal coincide",
        "max_abs_gap_by_scorer": gaps,
    }


def quantile_summary(values: np.ndarray, quantiles: tuple[float, ...]) -> Record:
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": rounded(float(np.mean(values))),
        "quantiles": {str(q): rounded(float(np.quantile(values, q))) for q in quantiles},
    }


def grouped_summaries(
    values: np.ndarray, groups: list[str], quantiles: tuple[float, ...]
) -> Record:
    keys = list(dict.fromkeys(groups))
    return {
        key: quantile_summary(values[[g == key for g in groups]], quantiles) for key in keys
    } | {"all": quantile_summary(values, quantiles)}


def supported_shares(decisions: np.ndarray, groups: list[str]) -> Record:
    result: Record = {}
    for key in [*dict.fromkeys(groups), "all"]:
        mask = np.ones(len(groups), bool) if key == "all" else np.array([g == key for g in groups])
        count = int(mask.sum())
        supported = int(decisions[mask].sum())
        result[key] = {
            "n": count,
            "supported": supported,
            "share": rounded(supported / count) if count else None,
        }
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> Record:
    if len(first) < 3:
        return {"n": int(len(first)), "rho": None, "p_value": None}
    result = spearmanr(first, second)
    return {
        "n": int(len(first)),
        "rho": rounded(float(result.statistic)),
        "p_value": float(result.pvalue),
    }


def cosine_relation(
    scores: dict[str, np.ndarray], cosine: np.ndarray, tiers: list[str], semantic: tuple[str, ...]
) -> Record:
    result: Record = {}
    for label, members in [*((tier, (tier,)) for tier in semantic), ("semantic_all", semantic)]:
        mask = np.array([tier in members for tier in tiers])
        result[label] = {
            name: spearman(values[mask], cosine[mask]) for name, values in scores.items()
        }
    return result


def evidence_score_check(
    recomputed: np.ndarray, recorded: np.ndarray, tiers: list[str], semantic: tuple[str, ...]
) -> Record:
    mask = np.array([tier in semantic for tier in tiers])
    gaps = np.abs(recomputed[mask] - recorded[mask])
    return {
        "rule": "cosine_serafim sentence_max.cosine of each semantic UDV against evidence.score",
        "n": int(mask.sum()),
        "max_abs_gap": rounded(float(gaps.max())) if len(gaps) else None,
        "within_1e-4": int((gaps <= 1e-4).sum()),
    }


def lowest_udvs(
    records: list[Record],
    primary: np.ndarray,
    secondary: np.ndarray,
    tiers: tuple[str, ...],
    count: int,
) -> Record:
    result: Record = {}
    for tier in tiers:
        rows = [
            (float(primary[index]), record["id"], index)
            for index, record in enumerate(records)
            if record["tier"] == tier
        ]
        rows.sort()
        result[tier] = [
            {
                "udv_id": records[index]["id"],
                "hearing_id": records[index]["hearing_id"],
                "support_type": support_type(records[index]),
                "primary_probability": rounded(score),
                "p4_supports": rounded(float(secondary[index])),
                "evidence_score": records[index]["evidence"]["score"],
                "proposition": records[index]["proposition"],
                "evidence_text": records[index]["evidence"]["text"],
            }
            for score, _, index in rows[:count]
        ]
    return result


def benchmark_overlap(verifier: Record, units: list[PremiseUnit]) -> Record:
    splits_of: dict[str, set[str]] = {}
    for row in load_jsonl(Path(verifier["benchmark"]["path"])):
        splits_of.setdefault(normalize_whitespace(row["opinion"]), set()).add(row["split"])
    matched: Counter[str] = Counter()
    for unit in units:
        found = splits_of.get(unit.hypothesis)
        matched["+".join(sorted(found)) if found else "not_in_benchmark"] += 1
    return {
        "rule": "UDVs whose normalized proposition equals a normalized NLI benchmark opinion, by "
        "the benchmark split of that opinion; train opinions were hypotheses of the verifier fit, "
        "paired there with retrieved chunks instead of the evidence sentence",
        "counts": dict(sorted(matched.items())),
    }


def score_reports(config: UdvVerifierConfig) -> Record:
    reports: Record = {}
    for key in config.scorers:
        path = experiments.score_report_file(config.run_dir, key)
        with open(path) as f:
            report = json.load(f)
        reports[key] = {
            "path": str(path),
            "sha256": sha256_of_file(path),
            "created_at": report["created_at"],
            "device": (report.get("model") or {}).get("device"),
            "timing": report["timing"],
            "truncation": report["truncation"],
            "label_probes_agreeing": None
            if report.get("label_probes") is None
            else f"{report['label_probes']['agreeing']}/{report['label_probes']['total']}",
            "files": report["files"],
        }
    return reports


def translate_record(config: UdvVerifierConfig) -> Record | None:
    path = config.run_dir / "translate_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        report = json.load(f)
    return {
        "path": str(path),
        "sha256": sha256_of_file(path),
        "created_at": report["created_at"],
        "runs": {key: value["run"] for key, value in report["models"].items()},
        "still_missing": {key: value["still_missing"] for key, value in report["models"].items()},
    }


def code_hashes() -> Record:
    return {
        "utils/udv_verifier.py": sha256_of_file(Path(__file__)),
        "utils/nli_verifier_exploration.py": sha256_of_file(Path(exploration.__file__)),
        **experiments.code_hashes(),
    }


def load_udv_threshold(path: Path, train_threshold: float) -> UdvThreshold:
    with open(path) as f:
        record = json.load(f)
    if not record["leak_check"]["passed"]:
        raise SystemExit(f"{path}: the leak check of the UDV-premise cut did not pass")
    if abs(float(record["train_threshold"]) - train_threshold) > 1e-3:
        raise SystemExit(f"{path}: train_threshold differs from the refitted primary threshold")
    return UdvThreshold(
        value=float(record["udv_threshold_exact"]),
        path=path,
        sha256=sha256_of_file(path),
        rule=record["adopted_rule"],
        record=record,
    )


def row_provenance(
    config: UdvVerifierConfig, train_threshold: float, udv_threshold: UdvThreshold
) -> Record:
    return {
        "verifier": config.primary,
        "premise": "evidence.text",
        "hypothesis": "proposition",
        "train_threshold": rounded(train_threshold),
        "udv_threshold": rounded(udv_threshold.value),
        "udv_threshold_rule": udv_threshold.rule,
        "udv_threshold_source": str(udv_threshold.path),
        "udv_source": str(config.udv_path),
    }


def output_rows(
    records: list[Record],
    split_of: dict[int, str],
    scored: dict[str, tuple[float, bool, float]],
    udv_threshold: float | None = None,
    provenance: Record | None = None,
) -> list[Record]:
    rows = []
    for record in records:
        values = scored.get(record["id"])
        row = {
            "udv_id": record["id"],
            "hearing_id": record["hearing_id"],
            "split": split_of[record["hearing_id"]],
            "tier": record["tier"],
            "support_type": None if not record.get("evidence") else support_type(record),
            "scored": values is not None,
            "primary_probability": None if values is None else values[0],
            "supported_at_train_threshold": None if values is None else values[1],
            "p4_supports": None if values is None else values[2],
        }
        if udv_threshold is not None:
            row["supported_at_udv_threshold"] = (
                None if values is None else values[0] >= udv_threshold
            )
            row["provenance"] = provenance
        rows.append(row)
    return rows


def analysis(
    config: UdvVerifierConfig,
    scored_records: list[Record],
    primary: np.ndarray,
    secondary: np.ndarray,
    decisions: np.ndarray,
    recomputed_cosine: np.ndarray,
    splits: list[str],
) -> Record:
    tiers = [record["tier"] for record in scored_records]
    types = [support_type(record) for record in scored_records]
    evidence_scores = np.array(
        [
            np.nan if record["evidence"]["score"] is None else record["evidence"]["score"]
            for record in scored_records
        ],
        dtype=float,
    )
    scores = {"primary_probability": primary, "p4_supports": secondary}
    distributions = {
        name: {
            "by_tier": grouped_summaries(values, tiers, config.quantiles),
            "by_support_type": grouped_summaries(values, types, config.quantiles),
            "by_hearing_split": grouped_summaries(values, splits, config.quantiles),
        }
        for name, values in scores.items()
    }
    return {
        "distributions": distributions,
        "supported_at_train_threshold": {
            "by_tier": supported_shares(decisions, tiers),
            "by_support_type": supported_shares(decisions, types),
            "by_hearing_split": supported_shares(decisions, splits),
        },
        "cosine_relation": {
            "rule": config.raw["report"]["cosine_rule"],
            "spearman": cosine_relation(scores, evidence_scores, tiers, config.semantic_tiers),
        },
        "evidence_score_check": evidence_score_check(
            recomputed_cosine, evidence_scores, tiers, config.semantic_tiers
        ),
        "lowest": {
            "rule": config.raw["report"]["lowest_rule"],
            "udvs": lowest_udvs(
                scored_records,
                primary,
                secondary,
                config.lowest_tiers,
                config.lowest_count,
            ),
        },
    }


def command_apply(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    started = time.perf_counter()
    for path in (config.output_path, config.report_path):
        if path.exists():
            raise SystemExit(f"{path} exists: apply writes new files only")
    verifier, exploration_config = open_run(config)
    records, units, unscored, split_of, split_source = load_udvs(config, verifier)
    candidate = primary_candidate(exploration_config, config.primary)
    with open(config.selection_path) as f:
        selection = json.load(f)
    with open(config.final_test_path) as f:
        final_test = json.load(f)
    if final_test["primary"] != config.primary:
        raise SystemExit("scoring.primary differs from the final_test primary")
    labels, hearings, fit_data, fit_files = load_fit_data(
        exploration_config, candidate_scorers(candidate), selection
    )
    fitted = fit_primary(candidate, labels, hearings, fit_data, exploration_config)
    refit = refit_check(fitted, final_test, config.primary)
    fit_reading = check_reading(exploration_config, fit_data)
    ids = [unit.unit_id for unit in units]
    udv_data: dict[str, ScorerData] = {}
    for key in config.scorers:
        kind, _ = exploration_config.scorers[key]
        scorer = load_scorer(key, kind, config.name, UDV_SPLIT, ids, exploration_config)
        if scorer is None:
            raise SystemExit(f"{key}: no UDV score file in {config.run_dir}; run score first")
        udv_data[key] = scorer
    reading = check_reading(exploration_config, udv_data)
    pools = pool_agreement(candidate, udv_data)
    primary = fitted.probabilities(udv_data)
    secondary = candidate_scores(feature_candidate(config.secondary), udv_data)
    decisions = primary >= fitted.threshold
    recomputed_cosine = candidate_scores(
        feature_candidate("cosine_serafim:sentence_max:max"), udv_data
    )
    scored = {
        unit_id: (float(p), bool(d), float(s))
        for unit_id, p, d, s in zip(ids, primary, decisions, secondary, strict=True)
    }
    udv_threshold = (
        None
        if config.udv_threshold_path is None
        else load_udv_threshold(config.udv_threshold_path, fitted.threshold)
    )
    write_jsonl(
        output_rows(
            records,
            split_of,
            scored,
            None if udv_threshold is None else udv_threshold.value,
            None
            if udv_threshold is None
            else row_provenance(config, fitted.threshold, udv_threshold),
        ),
        config.output_path,
    )
    by_id = {record["id"]: record for record in records}
    scored_records = [by_id[unit_id] for unit_id in ids]
    splits = [unit.split for unit in units]
    report = {
        "experiment": "udv_verifier",
        "name": config.name,
        "created_at": experiments.now(),
        "declared": config.raw["declared"],
        "purpose": config.raw["purpose"],
        "caveats": {
            "domain_shift": config.raw["report"]["domain_shift"],
            "label_semantics": config.raw["report"]["label_semantics"],
            "benchmark_label_semantics": verifier.source["benchmark"]["label_semantics"],
        },
        "counts": {
            "udv_records": len(records),
            "scored": len(units),
            "unscored": sum(unscored.values()),
            "unscored_by_tier": dict(sorted(unscored.items())),
            "scored_by_tier": dict(Counter(r["tier"] for r in scored_records)),
            "scored_by_support_type": dict(Counter(support_type(r) for r in scored_records)),
            "scored_by_hearing_split": dict(sorted(Counter(splits).items())),
        },
        "benchmark_overlap": benchmark_overlap(verifier.source, units),
        "primary": {
            "candidate": config.primary,
            "features": len(candidate.features),
            "scorers": list(candidate_scorers(candidate)),
            "fit": fitted.record,
            "final_test_result": final_test["results"][config.primary],
            "decision_rule": config.raw["scoring"]["decision_rule"],
        },
        "secondary": {
            "signal": config.secondary,
            "reason": config.raw["scoring"]["secondary_reason"],
        },
        "refit_check": refit,
        "fit_files": fit_files,
        "fit_reading_check": fit_reading,
        "udv_reading_check": reading,
        "pool_check": pools,
        **analysis(
            config, scored_records, primary, secondary, decisions, recomputed_cosine, splits
        ),
        "outputs": {
            "path": str(config.output_path),
            "rows": len(records),
            "sha256": sha256_of_file(config.output_path),
        },
        "inputs": {
            "udv": {"path": str(config.udv_path), "sha256": sha256_of_file(config.udv_path)},
            "splits": split_source,
            "selection": {
                "path": str(config.selection_path),
                "sha256": sha256_of_file(config.selection_path),
            },
            "final_test": {
                "path": str(config.final_test_path),
                "sha256": sha256_of_file(config.final_test_path),
            },
            "udv_score_files": {
                key: {"path": str(s.path), "sha256": sha256_of_file(s.path)}
                for key, s in udv_data.items()
            },
        },
        "score_runs": score_reports(config),
        "translation": translate_record(config),
        "timing": {"apply_seconds": round(time.perf_counter() - started, 1)},
        "config": {
            "path": str(config.path),
            "sha256": sha256_of_file(config.path),
            "echo": config.raw,
            "verifier_config": {
                "path": str(config.verifier_config),
                "sha256": sha256_of_file(config.verifier_config),
            },
            "exploration_config": {
                "path": str(config.exploration_config),
                "sha256": sha256_of_file(config.exploration_config),
            },
        },
        "code": code_hashes(),
        "environment": experiments.environment(),
    }
    if udv_threshold is not None:
        udv_decisions = primary >= udv_threshold.value
        tiers = [record["tier"] for record in scored_records]
        report["udv_threshold"] = {
            "value": rounded(udv_threshold.value),
            "exact": udv_threshold.value,
            "rule": udv_threshold.rule,
            "interval": udv_threshold.record["rules"][udv_threshold.rule]["bootstrap"]["threshold"],
            "path": str(udv_threshold.path),
            "sha256": udv_threshold.sha256,
        }
        report["supported_at_udv_threshold"] = {
            "by_tier": supported_shares(udv_decisions, tiers),
            "by_support_type": supported_shares(
                udv_decisions, [support_type(r) for r in scored_records]
            ),
            "by_hearing_split": supported_shares(udv_decisions, splits),
        }
    write_json(report, config.report_path)
    summary = {
        "refit_check": refit["checks"],
        "supported_by_tier": report["supported_at_train_threshold"]["by_tier"],
    }
    print(json.dumps(summary, indent=2), flush=True)


def annotation_labels(
    annotation: Path, key: Path, question: str, label_column: str
) -> dict[str, str]:
    with open(key) as f:
        items = json.load(f)["items"]
    with open(annotation, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    labels: dict[str, str] = {}
    for row in rows:
        item = items.get(row["item_id"])
        if item is None:
            raise SystemExit(f"{row['item_id']}: not in {key}")
        if item["question"] != question or row["pergunta"] != question:
            continue
        if len(item["udv_ids"]) != 1:
            raise SystemExit(f"{row['item_id']}: {question} item with several UDVs")
        value = row[label_column].strip()
        if not value:
            raise SystemExit(f"{row['item_id']}: empty {label_column}")
        labels[item["udv_ids"][0]] = value
    return labels


def agreement_auc(
    scores: np.ndarray,
    positive: np.ndarray,
    hearings: np.ndarray,
    samples: int,
    seed: int,
    level: float,
) -> Record:
    count = int(positive.sum())
    if count == 0 or count == len(positive):
        return {"n": int(len(positive)), "positives": count, "roc_auc": None}
    draws = experiments.hearing_draws(hearings, samples, np.random.default_rng(seed))
    values = [
        float(roc_auc_score(positive[rows], scores[rows]))
        for rows in draws
        if positive[rows].any() and not positive[rows].all()
    ]
    return {
        "n": int(len(positive)),
        "positives": count,
        "negatives": int(len(positive) - count),
        "roc_auc": rounded(float(roc_auc_score(positive, scores))),
        "hearing_bootstrap": interval(values, level),
    }


def command_annotation_agreement(args: argparse.Namespace, config: UdvVerifierConfig) -> None:
    with open(config.report_path) as f:
        report = json.load(f)
    section = args.section
    if section in report:
        raise SystemExit(f"{config.report_path} already has a {section} section")
    labels = annotation_labels(args.annotation, args.key, args.question, args.label_column)
    rows = {row["udv_id"]: row for row in load_jsonl(config.output_path)}
    missing = [udv for udv in labels if not rows.get(udv, {}).get("scored")]
    if missing:
        raise SystemExit(f"annotated UDVs without a verifier score: {missing[:10]}")
    order = sorted(labels)
    values = [labels[udv] for udv in order]
    hearings = np.array([rows[udv]["hearing_id"] for udv in order])
    level = float(experiments.load_config(config.verifier_config).confidence_level)
    contrasts = {
        "correct_vs_rest": ({"correta"}, "positive = correta; negative = parcial or incorreta"),
        "correct_or_partial_vs_incorrect": (
            {"correta", "parcial"},
            "positive = correta or parcial; negative = incorreta",
        ),
    }
    results: Record = {}
    for name, (positive_labels, rule) in contrasts.items():
        positive = np.array([value in positive_labels for value in values])
        results[name] = {
            "rule": rule,
            **{
                score: agreement_auc(
                    np.array([rows[udv][score] for udv in order], dtype=float),
                    positive,
                    hearings,
                    args.bootstrap_samples,
                    args.seed,
                    level,
                )
                for score in ("primary_probability", "p4_supports")
            },
        }
    decisions = np.array([rows[udv]["supported_at_train_threshold"] for udv in order])
    report[section] = {
        "created_at": experiments.now(),
        "annotator": args.annotator,
        "scope": "agreement of the verifier scores with this annotation; it is not a comparison "
        "with a human judgment and not an estimate of accuracy",
        "question": args.question,
        "label_counts": dict(sorted(Counter(values).items())),
        "items": len(order),
        "hearings": int(len(set(hearings.tolist()))),
        "split_counts": dict(sorted(Counter(rows[udv]["split"] for udv in order).items())),
        "tier_counts": dict(sorted(Counter(rows[udv]["tier"] for udv in order).items())),
        "supported_at_train_threshold_by_label": {
            label: {
                "n": sum(1 for v in values if v == label),
                "supported": int(
                    sum(d for v, d in zip(values, decisions, strict=True) if v == label)
                ),
            }
            for label in sorted(set(values))
        },
        "roc_auc": results,
        "bootstrap_rule": f"{args.bootstrap_samples} hearing resamples (seed {args.seed}), "
        "percentile interval; replicates with a single class are skipped",
        "inputs": {
            "annotation_sha256": sha256_of_file(args.annotation),
            "key": {"path": str(args.key), "sha256": sha256_of_file(args.key)},
            "verifier_output_sha256": sha256_of_file(config.output_path),
        },
        "code": {"utils/udv_verifier.py": sha256_of_file(Path(__file__))},
    }
    write_json(report, config.report_path)
    print(json.dumps(results, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entailment-style verifier scores on the UDV evidence: translate, score "
        "with the E3 scorers, then apply the E3x primary refitted on the benchmark train split."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv_verifier.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("translate", "score"):
        command = commands.add_parser(name)
        command.add_argument("--device", choices=experiments.DEVICES, default=None)
    commands.add_parser("apply", help="refit the primary on train and write the UDV scores")
    agreement = commands.add_parser(
        "annotation-agreement",
        help="add the ROC AUC of the verifier scores against an external annotation",
    )
    agreement.add_argument("--annotation", type=Path, required=True)
    agreement.add_argument("--key", type=Path, required=True)
    agreement.add_argument("--annotator", required=True)
    agreement.add_argument("--section", default="annotation_agreement")
    agreement.add_argument("--question", default="trecho_sustenta")
    agreement.add_argument("--label-column", default="julgamento")
    agreement.add_argument("--bootstrap-samples", type=int, default=1000)
    agreement.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = {
        "translate": command_translate,
        "score": command_score,
        "apply": command_apply,
        "annotation-agreement": command_annotation_agreement,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
