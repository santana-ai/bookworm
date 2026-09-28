import argparse
import csv
import itertools
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.common.stats import holm
from experiments.data.nli_benchmark import load_split_lookup
from experiments.udv.calibrate_threshold import interval, rounded
from experiments.verifier.decision_scoring import bootstrap_p_value
from experiments.verifier.nli_experiments import (
    binary_metrics,
    code_hashes,
    environment,
    hearing_draws,
    max_f1_not_inferable_optimum,
    now,
)

Record = dict[str, Any]

SCORES_ROOT = Path("artifacts/experiments/nli_verifier")
OUTPUT_ROOT = Path("artifacts/experiments/nli_verifier_exploration")
KIND_ORDER = {"signal": 0, "panel": 1, "learned": 2}
CHUNK_POOLS = ("max", "mean", "top2_mean", "noisy_or")
BATTERY_SIGNALS = (
    "p1_nli",
    "p2_nli_reversed",
    "p3_inferable",
    "p4_supports",
    "p5_position",
    "p6_position_reversed",
    "p7_coverage",
    "p8_similarity",
)
LEARNED_POOLS = ("max", "mean", "concatenated")
MATCH_TOLERANCE = 1e-9
EMPTY_SCORE = {"decision": 0.0, "nli": 0.0, "cosine": -1.0}


@dataclass(frozen=True)
class ExplorationConfig:
    name: str
    raw: Record
    path: Path
    verifier: Record
    scorers: dict[str, tuple[str, str]]
    decision_signals: tuple[str, ...]
    order_pairs: dict[str, tuple[str, str]]
    panel_components: tuple[str, ...]
    panel_min_size: int
    pools: tuple[str, ...]
    c_grid: tuple[float, ...]
    inner_splits: int
    cross_models: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    n_splits: int
    repeats: int
    cv_seed: int
    references: tuple[str, ...]
    bootstrap_samples: int
    bootstrap_seed: int


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: str
    scorer: str
    components: tuple[str, ...]
    pool: str
    features: tuple[str, ...] = ()


@dataclass
class ScorerData:
    key: str
    kind: str
    path: Path
    chunks: dict[str, list[list[float]]]
    concatenated: dict[str, np.ndarray]
    stored: dict[str, np.ndarray]


def load_exploration_config(path: Path) -> ExplorationConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)["exploration"]
    with open(raw["verifier_config"], "rb") as f:
        verifier = tomllib.load(f)
    inputs, candidates = raw["inputs"], raw["candidates"]
    scorers: dict[str, tuple[str, str]] = {}
    for kind, table in (
        ("decision", inputs["decision_scorers"]),
        ("nli", inputs["nli_scorers"]),
        ("cosine", inputs["cosine_scorers"]),
    ):
        for key, run in table.items():
            scorers[key] = (kind, run)
    order_pairs = {name: (pair[0], pair[1]) for name, pair in candidates["order_pairs"].items()}
    entries = [
        {"name": "cross_model", "bases": [raw_cross["base"]], "extra": raw_cross["extra"]}
        for raw_cross in [candidates["cross_model"]]
        if "cross_model" in candidates
    ] + list(candidates.get("cross_models", []))
    cross_models = []
    for entry in entries:
        if any(len(item.split(":")) != 3 for item in entry["extra"]):
            raise SystemExit("cross model extra entries must be scorer:signal:pool")
        cross_models.append((entry["name"], tuple(entry["bases"]), tuple(entry["extra"])))
    selection = raw["selection"]
    if selection["metric"] != "roc_auc":
        raise SystemExit("only roc_auc is supported as the selection metric")
    for pool in candidates["pools"]:
        if pool not in (*CHUNK_POOLS, "concatenated"):
            raise SystemExit(f"unknown pool {pool!r}")
    return ExplorationConfig(
        name=raw["name"],
        raw=raw,
        path=path,
        verifier=verifier,
        scorers=scorers,
        decision_signals=tuple(candidates["decision_signals"]),
        order_pairs=order_pairs,
        panel_components=tuple(candidates["panel_components"]),
        panel_min_size=int(candidates["panel_min_size"]),
        pools=tuple(candidates["pools"]),
        c_grid=tuple(float(c) for c in candidates["c_grid"]),
        inner_splits=int(candidates["inner_splits"]),
        cross_models=tuple(cross_models),
        n_splits=int(raw["cv"]["n_splits"]),
        repeats=int(raw["cv"]["repeats"]),
        cv_seed=int(raw["cv"]["seed"]),
        references=tuple(raw["confirmation"]["references"]),
        bootstrap_samples=int(raw["confirmation"]["bootstrap_samples"]),
        bootstrap_seed=int(raw["confirmation"]["seed"]),
    )


def load_labels(config: ExplorationConfig, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    verifier = config.verifier
    benchmark = Path(verifier["benchmark"]["path"])
    with open(verifier["benchmark"]["report_path"]) as f:
        recorded = json.load(f)["artifact"]["sha256"]
    if sha256_of_file(benchmark) != recorded:
        raise SystemExit(f"{benchmark}: sha256 differs from its report")
    split_of, _ = load_split_lookup(
        Path(verifier["splits"]["manifest_path"]), verifier["dataset"]["lds_sha256"]
    )
    ids, labels, hearings = [], [], []
    for row in load_jsonl(benchmark):
        if split_of.get(row["hearing_id"]) != row["split"]:
            raise SystemExit(f"{row['id']}: split differs from the manifest")
        if row["split"] == split:
            ids.append(row["id"])
            labels.append(bool(row["label_inferable"]))
            hearings.append(int(row["hearing_id"]))
    return ids, np.array(labels), np.array(hearings)


def item_signals(kind: str, item: Record | None, config: ExplorationConfig) -> dict[str, float]:
    if not item:
        return {}
    if kind == "decision":
        signals = item.get("signals") or {}
        values = {name: float(value) for name, value in signals.items()}
        for name, (first, second) in config.order_pairs.items():
            if first in values and second in values:
                values[name] = (values[first] + values[second]) / 2
        return values
    if kind == "nli":
        return {"entailment": float(item["probabilities"]["entailment"])}
    return {"cosine": float(item["cosine"]), "sentence_max": float(item["sentence_max"])}


def load_scorer(
    key: str, kind: str, run: str, split: str, ids: list[str], config: ExplorationConfig
) -> ScorerData | None:
    path = SCORES_ROOT / run / "scores" / f"{key}_{split}.jsonl"
    if not path.exists():
        return None
    rows = {row["id"]: row for row in load_jsonl(path)}
    if set(rows) != set(ids):
        raise SystemExit(f"{path}: opinions differ from the benchmark {split} rows")
    per_chunk: dict[str, list[list[float]]] = {}
    concatenated: dict[str, list[float]] = {}
    stored: dict[str, list[float]] = {}
    for index, opinion in enumerate(ids):
        row = rows[opinion]
        for name, value in row["scores"].items():
            stored.setdefault(name, []).append(float(value))
        signals = [item_signals(kind, item, config) for item in row.get("items") or []]
        signals = [values for values in signals if values]
        names = set().union(*signals) if signals else set()
        for name in names:
            per_chunk.setdefault(name, [[] for _ in ids])[index] = [v[name] for v in signals]
        joined = item_signals(kind, row.get("concatenated"), config)
        for name, value in joined.items():
            concatenated.setdefault(name, [float("nan")] * len(ids))[index] = value
    return ScorerData(
        key=key,
        kind=kind,
        path=path,
        chunks=per_chunk,
        concatenated={name: np.array(values) for name, values in concatenated.items()},
        stored={name: np.array(values) for name, values in stored.items()},
    )


def pool_values(values: list[float], pool: str, empty: float = 0.0) -> float:
    if not values:
        return empty
    array = np.asarray(values, dtype=float)
    if pool == "max":
        return float(array.max())
    if pool == "mean":
        return float(array.mean())
    if pool == "top2_mean":
        return float(np.sort(array)[-2:].mean())
    if pool == "noisy_or":
        return float(1.0 - np.prod(1.0 - np.clip(array, 0.0, 1.0)))
    raise ValueError(pool)


def panel_chunks(data: ScorerData, components: tuple[str, ...]) -> list[list[float]]:
    count = len(next(iter(data.chunks.values())))
    merged = []
    for index in range(count):
        columns = [data.chunks[name][index] for name in components]
        merged.append([float(np.mean(values)) for values in zip(*columns, strict=True)])
    return merged


def candidate_scores(candidate: Candidate, data: dict[str, ScorerData]) -> np.ndarray:
    scorer = data[candidate.scorer]
    if candidate.pool == "concatenated":
        columns = [scorer.concatenated[name] for name in candidate.components]
        return np.nan_to_num(np.mean(columns, axis=0), nan=EMPTY_SCORE[scorer.kind])
    if candidate.kind == "signal":
        chunks = list(scorer.chunks[candidate.components[0]])
    else:
        chunks = panel_chunks(scorer, candidate.components)
    empty = EMPTY_SCORE[scorer.kind]
    return np.array([pool_values(values, candidate.pool, empty) for values in chunks])


def fixed_candidates(config: ExplorationConfig, data: dict[str, ScorerData]) -> list[Candidate]:
    candidates = []
    for key, scorer in data.items():
        pools = [p for p in config.pools if p != "concatenated" or scorer.concatenated]
        if scorer.kind == "decision":
            signals = config.decision_signals
        elif scorer.kind == "nli":
            signals = tuple(config.raw["candidates"]["nli_signals"])
        else:
            signals = tuple(config.raw["candidates"]["cosine_signals"])
        for signal in signals:
            for pool in pools:
                candidates.append(
                    Candidate(f"{key}:{signal}:{pool}", "signal", key, (signal,), pool)
                )
        if scorer.kind != "decision":
            continue
        for size in range(config.panel_min_size, len(config.panel_components) + 1):
            for subset in itertools.combinations(config.panel_components, size):
                label = "+".join(subset)
                for pool in pools:
                    candidates.append(
                        Candidate(f"{key}:panel[{label}]:{pool}", "panel", key, subset, pool)
                    )
    return candidates


def learned_candidates(config: ExplorationConfig, data: dict[str, ScorerData]) -> list[Candidate]:
    candidates = []
    for key, scorer in data.items():
        if scorer.kind != "decision":
            continue
        features = tuple(f"{key}:{s}:{p}" for s in BATTERY_SIGNALS for p in LEARNED_POOLS)
        candidates.append(Candidate(f"{key}:learned", "learned", key, (), "learned", features))
    for name, bases, extra in config.cross_models:
        if not all(scorer in data for scorer in (*bases, *(e.split(":")[0] for e in extra))):
            continue
        features = tuple(
            f"{base}:{s}:{p}" for base in bases for s in BATTERY_SIGNALS for p in LEARNED_POOLS
        )
        candidates.append(
            Candidate(
                f"{bases[0]}:learned_{name}", "learned", bases[0], (), "learned", features + extra
            )
        )
    return candidates


def feature_candidate(feature: str) -> Candidate:
    scorer, signal, pool = feature.split(":")
    return Candidate(feature, "signal", scorer, (signal,), pool)


def reference_candidate(reference: str) -> Candidate:
    scorer, signal, pool = reference.split(":")
    if signal == "panel":
        return Candidate(reference, "panel", scorer, (), pool)
    return feature_candidate(reference)


def resolve_reference(
    reference: str, config: ExplorationConfig, data: dict[str, ScorerData]
) -> np.ndarray:
    candidate = reference_candidate(reference)
    if candidate.kind == "panel":
        declared = tuple(config.verifier["decision_battery"]["aggregation"]["panel_components"])
        candidate = Candidate(reference, "panel", candidate.scorer, declared, candidate.pool)
    return candidate_scores(candidate, data)


def check_reading(config: ExplorationConfig, data: dict[str, ScorerData]) -> Record:
    checked: Record = {}
    declared = tuple(config.verifier["decision_battery"]["aggregation"]["panel_components"])
    for key, scorer in data.items():
        pairs: list[tuple[str, Candidate]] = []
        if scorer.kind == "decision":
            pairs += [
                (f"max.{s}", Candidate("", "signal", key, (s,), "max")) for s in BATTERY_SIGNALS
            ]
            pairs.append(("max.panel", Candidate("", "panel", key, declared, "max")))
            if scorer.concatenated:
                pairs += [
                    (f"concatenated.{s}", Candidate("", "signal", key, (s,), "concatenated"))
                    for s in BATTERY_SIGNALS
                ]
        elif scorer.kind == "nli":
            pairs.append(("max.entailment", Candidate("", "signal", key, ("entailment",), "max")))
            if scorer.concatenated:
                pairs.append(
                    (
                        "concatenated.entailment",
                        Candidate("", "signal", key, ("entailment",), "concatenated"),
                    )
                )
        else:
            pairs.append(("max.cosine", Candidate("", "signal", key, ("cosine",), "max")))
            pairs.append(
                ("sentence_max.cosine", Candidate("", "signal", key, ("sentence_max",), "max"))
            )
        worst = 0.0
        for stored_name, candidate in pairs:
            gap = float(
                np.max(
                    np.abs(candidate_scores(candidate, {key: scorer}) - scorer.stored[stored_name])
                )
            )
            if gap > MATCH_TOLERANCE:
                raise SystemExit(
                    f"{key}: recomputed {stored_name} differs from the score file by {gap}"
                )
            worst = max(worst, gap)
        checked[key] = {"stored_scores_checked": [name for name, _ in pairs], "max_abs_gap": worst}
    return checked


def cv_folds(
    config: ExplorationConfig, labels: np.ndarray, hearings: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for repeat in range(config.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.cv_seed + repeat
        )
        folds += list(splitter.split(np.zeros(len(labels)), labels.astype(int), hearings))
    return folds


def fold_kappa(scores: np.ndarray, labels: np.ndarray, fit: np.ndarray, held: np.ndarray) -> float:
    optimum = max_f1_not_inferable_optimum(scores[fit], labels[fit])
    if optimum is None:
        return float("nan")
    return binary_metrics(labels[held], scores[held] >= optimum["threshold"])["cohen_kappa"]


def fit_learned(
    matrix: np.ndarray, labels: np.ndarray, hearings: np.ndarray, config: ExplorationConfig
) -> tuple[StandardScaler, LogisticRegression, float, Record]:
    splitter = StratifiedGroupKFold(
        n_splits=config.inner_splits, shuffle=True, random_state=config.cv_seed
    )
    inner = list(splitter.split(matrix, labels.astype(int), hearings))
    means = {}
    for c in config.c_grid:
        values = []
        for fit, held in inner:
            scaler = StandardScaler().fit(matrix[fit])
            model = LogisticRegression(C=c, max_iter=2000, random_state=42)
            model.fit(scaler.transform(matrix[fit]), labels[fit].astype(int))
            values.append(
                roc_auc_score(
                    labels[held], model.predict_proba(scaler.transform(matrix[held]))[:, 1]
                )
            )
        means[c] = float(np.mean(values))
    chosen = max(config.c_grid, key=lambda c: (means[c], -c))
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=chosen, max_iter=2000, random_state=42)
    model.fit(scaler.transform(matrix), labels.astype(int))
    return scaler, model, chosen, {str(c): rounded(v) for c, v in means.items()}


def feature_matrix(candidate: Candidate, data: dict[str, ScorerData]) -> np.ndarray:
    return np.column_stack(
        [candidate_scores(feature_candidate(f), data) for f in candidate.features]
    )


def evaluate_fixed(
    scores: np.ndarray, labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[list[float], list[float]]:
    aucs = [float(roc_auc_score(labels[held], scores[held])) for _, held in folds]
    kappas = [fold_kappa(scores, labels, fit, held) for fit, held in folds]
    return aucs, kappas


def evaluate_learned(
    candidate: Candidate,
    data: dict[str, ScorerData],
    labels: np.ndarray,
    hearings: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ExplorationConfig,
) -> tuple[list[float], list[float], list[float]]:
    matrix = feature_matrix(candidate, data)
    aucs, kappas, chosen = [], [], []
    for fit, held in folds:
        scaler, model, c, _ = fit_learned(matrix[fit], labels[fit], hearings[fit], config)
        fit_scores = model.predict_proba(scaler.transform(matrix[fit]))[:, 1]
        held_scores = model.predict_proba(scaler.transform(matrix[held]))[:, 1]
        aucs.append(float(roc_auc_score(labels[held], held_scores)))
        optimum = max_f1_not_inferable_optimum(fit_scores, labels[fit])
        kappas.append(
            float("nan")
            if optimum is None
            else binary_metrics(labels[held], held_scores >= optimum["threshold"])["cohen_kappa"]
        )
        chosen.append(c)
    return aucs, kappas, chosen


def summary_row(
    candidate: Candidate, aucs: list[float], kappas: list[float], n_splits: int
) -> Record:
    return {
        "candidate": candidate.key,
        "kind": candidate.kind,
        "scorer": candidate.scorer,
        "components": "+".join(candidate.components),
        "component_count": len(candidate.components)
        if candidate.kind != "learned"
        else len(candidate.features),
        "pool": candidate.pool,
        "cv_roc_auc_mean": float(np.mean(aucs)),
        "cv_roc_auc_se": float(np.std(aucs, ddof=1) / np.sqrt(n_splits)),
        "cv_roc_auc_min": float(np.min(aucs)),
        "cv_roc_auc_max": float(np.max(aucs)),
        "cv_kappa_mean": float(np.nanmean(kappas)),
        "cv_kappa_se": float(np.nanstd(kappas, ddof=1) / np.sqrt(n_splits)),
    }


def select(rows: list[Record]) -> tuple[Record, Record, list[Record]]:
    best = max(rows, key=lambda row: row["cv_roc_auc_mean"])
    floor = best["cv_roc_auc_mean"] - best["cv_roc_auc_se"]
    eligible = [row for row in rows if row["cv_roc_auc_mean"] >= floor]
    selected = min(
        eligible,
        key=lambda row: (
            KIND_ORDER[row["kind"]],
            row["component_count"] if row["kind"] == "panel" else 0,
            -row["cv_roc_auc_mean"],
        ),
    )
    return best, selected, eligible


def output_dir(config: ExplorationConfig, override: Path | None) -> Path:
    return (override or OUTPUT_ROOT) / config.name


def load_all(
    config: ExplorationConfig, split: str
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, ScorerData], list[str]]:
    ids, labels, hearings = load_labels(config, split)
    data: dict[str, ScorerData] = {}
    missing = []
    for key, (kind, run) in config.scorers.items():
        scorer = load_scorer(key, kind, run, split, ids, config)
        if scorer is None:
            missing.append(key)
        else:
            data[key] = scorer
    return ids, labels, hearings, data, missing


def write_csv(path: Path, rows: list[Record]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: rounded(v) if isinstance(v, float) else v for k, v in row.items()})


def command_explore(args: argparse.Namespace, config: ExplorationConfig) -> None:
    split = config.raw["fit_split"]
    ids, labels, hearings, data, missing = load_all(config, split)
    reading = check_reading(config, data)
    folds = cv_folds(config, labels, hearings)
    rows = []
    fixed = fixed_candidates(config, data)
    for candidate in fixed:
        aucs, kappas = evaluate_fixed(candidate_scores(candidate, data), labels, folds)
        rows.append(summary_row(candidate, aucs, kappas, config.n_splits))
    print(f"{len(fixed)} fixed candidates evaluated on {len(folds)} folds", flush=True)
    learned_c: Record = {}
    for candidate in learned_candidates(config, data):
        aucs, kappas, chosen = evaluate_learned(candidate, data, labels, hearings, folds, config)
        rows.append(summary_row(candidate, aucs, kappas, config.n_splits))
        learned_c[candidate.key] = {str(c): chosen.count(c) for c in config.c_grid}
        print(f"evaluated {candidate.key}", flush=True)
    references = {}
    for reference in config.raw["selection"]["references"]:
        aucs, kappas = evaluate_fixed(resolve_reference(reference, config, data), labels, folds)
        references[reference] = {
            k: rounded(v) if isinstance(v, float) else v
            for k, v in summary_row(
                reference_candidate(reference), aucs, kappas, config.n_splits
            ).items()
        }
    best, selected, eligible = select(rows)
    rows.sort(key=lambda row: -row["cv_roc_auc_mean"])
    out = output_dir(config, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "cv_results.csv", rows)
    selection = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "created_at": now(),
        "split_read": split,
        "files_read": {
            key: {"path": str(s.path), "sha256": sha256_of_file(s.path)} for key, s in data.items()
        },
        "scorers_missing": missing,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "folds": len(folds),
        "candidates": {
            "total": len(rows),
            "by_kind": {k: sum(r["kind"] == k for r in rows) for k in KIND_ORDER},
        },
        "reading_check": reading,
        "learned_c_chosen_per_fold": learned_c,
        "references_cv": references,
        "best": {k: rounded(v) if isinstance(v, float) else v for k, v in best.items()},
        "eligible_count": len(eligible),
        "selected": {k: rounded(v) if isinstance(v, float) else v for k, v in selected.items()},
        "selection_rule": config.raw["selection"]["rule"],
        "config": {"path": str(config.path), "sha256": sha256_of_file(config.path)},
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(out / "selection.json", "w") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "best": selection["best"],
                "selected": selection["selected"],
                "references_cv": references,
            },
            indent=2,
        )
    )


def selected_candidate(
    selection: Record, config: ExplorationConfig, data: dict[str, ScorerData]
) -> Candidate:
    key = selection["selected"]["candidate"]
    for candidate in [*fixed_candidates(config, data), *learned_candidates(config, data)]:
        if candidate.key == key:
            return candidate
    raise SystemExit(f"selected candidate {key} cannot be rebuilt")


def confirm_candidates(
    selection: Record, config: ExplorationConfig, data: dict[str, ScorerData], out: Path
) -> list[Candidate]:
    floor = config.raw["confirmation"].get("cv_floor")
    keys = [selection["selected"]["candidate"]]
    if floor is not None:
        with open(out / "cv_results.csv") as f:
            keys += [
                row["candidate"]
                for row in csv.DictReader(f)
                if float(row["cv_roc_auc_mean"]) >= float(floor) and row["candidate"] not in keys
            ]
    built = {c.key: c for c in [*fixed_candidates(config, data), *learned_candidates(config, data)]}
    missing = [key for key in keys if key not in built]
    if missing:
        raise SystemExit(f"candidates cannot be rebuilt: {missing}")
    return [built[key] for key in keys]


def system_scores(
    candidate: Candidate,
    fit_data: dict[str, ScorerData],
    data: dict[str, ScorerData],
    fit_labels: np.ndarray,
    fit_hearings: np.ndarray,
    config: ExplorationConfig,
) -> tuple[np.ndarray, np.ndarray, Record]:
    if candidate.kind != "learned":
        return candidate_scores(candidate, fit_data), candidate_scores(candidate, data), {}
    fit_matrix = feature_matrix(candidate, fit_data)
    scaler, model, c, means = fit_learned(fit_matrix, fit_labels, fit_hearings, config)
    learned = {
        "c": c,
        "inner_means": means,
        "coefficients": dict(zip(candidate.features, map(rounded, model.coef_[0]), strict=True)),
    }
    return (
        model.predict_proba(scaler.transform(fit_matrix))[:, 1],
        model.predict_proba(scaler.transform(feature_matrix(candidate, data)))[:, 1],
        learned,
    )


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, draws: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [
            float(roc_auc_score(labels[rows], scores[rows]))
            if labels[rows].any() and not labels[rows].all()
            else float("nan")
            for rows in draws
        ]
    )


def evaluate_split(
    systems: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_labels: np.ndarray,
    labels: np.ndarray,
    hearings: np.ndarray,
    config: ExplorationConfig,
) -> tuple[Record, dict[str, dict[str, np.ndarray]]]:
    draws = hearing_draws(
        hearings, config.bootstrap_samples, np.random.default_rng(config.bootstrap_seed)
    )
    level = float(config.verifier["evaluation"]["confidence_level"])
    results: Record = {}
    replicates: dict[str, dict[str, np.ndarray]] = {}
    for name, (fit_scores, scores) in systems.items():
        optimum = max_f1_not_inferable_optimum(fit_scores, fit_labels)
        threshold = None if optimum is None else optimum["threshold"]
        predictions = scores >= threshold if threshold is not None else np.ones(len(scores), bool)
        auc = bootstrap_auc(labels, scores, draws)
        kappa = np.array(
            [binary_metrics(labels[rows], predictions[rows])["cohen_kappa"] for rows in draws]
        )
        replicates[name] = {"roc_auc": auc, "cohen_kappa": kappa}
        metrics = binary_metrics(labels, predictions)
        results[name] = {
            "roc_auc": rounded(float(roc_auc_score(labels, scores))),
            "roc_auc_interval": interval(auc[~np.isnan(auc)].tolist(), level),
            "threshold": threshold,
            "cohen_kappa": rounded(metrics["cohen_kappa"]),
            "cohen_kappa_interval": interval(kappa[~np.isnan(kappa)].tolist(), level),
            "f1_not_inferable": rounded(metrics["f1_not_inferable"]),
            "precision_not_inferable": rounded(metrics["precision_not_inferable"]),
            "recall_not_inferable": rounded(metrics["recall_not_inferable"]),
        }
    return results, replicates


def compare_systems(
    keys: list[str],
    references: tuple[str, ...],
    results: Record,
    replicates: dict[str, dict[str, np.ndarray]],
    config: ExplorationConfig,
) -> list[Record]:
    level = float(config.verifier["evaluation"]["confidence_level"])
    comparisons: list[Record] = []
    for key in keys:
        for reference in references:
            for metric in ("roc_auc", "cohen_kappa"):
                deltas = replicates[key][metric] - replicates[reference][metric]
                comparisons.append(
                    {
                        "candidate": key,
                        "reference": reference,
                        "metric": metric,
                        "delta": rounded(
                            float(results[key][metric]) - float(results[reference][metric])
                        ),
                        "bootstrap": interval(deltas[~np.isnan(deltas)].tolist(), level),
                        "p_value": bootstrap_p_value(deltas),
                    }
                )
    auc_entries = [entry for entry in comparisons if entry["metric"] == "roc_auc"]
    adjusted = holm([float(entry["p_value"]) for entry in auc_entries])
    for entry, value in zip(auc_entries, adjusted, strict=True):
        entry["p_holm"] = value
    return comparisons


def command_confirm(args: argparse.Namespace, config: ExplorationConfig) -> None:
    out = output_dir(config, args.output_dir)
    target = out / "confirmation.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the confirmation runs once")
    with open(out / "selection.json") as f:
        selection = json.load(f)
    fit_split, confirm_split = config.raw["fit_split"], config.raw["confirm_split"]
    _, fit_labels, fit_hearings, fit_data, _ = load_all(config, fit_split)
    ids, labels, hearings, data, missing = load_all(config, confirm_split)
    candidates = confirm_candidates(selection, config, fit_data, out)
    cv = {}
    with open(out / "cv_results.csv") as f:
        for row in csv.DictReader(f):
            cv[row["candidate"]] = row
    systems: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    learned: Record = {}
    for candidate in candidates:
        fit_scores, scores, model = system_scores(
            candidate, fit_data, data, fit_labels, fit_hearings, config
        )
        systems[candidate.key] = (fit_scores, scores)
        if model:
            learned[candidate.key] = model
    for reference in config.references:
        systems[reference] = (
            resolve_reference(reference, config, fit_data),
            resolve_reference(reference, config, data),
        )
    results, replicates = evaluate_split(systems, fit_labels, labels, hearings, config)
    for name, result in results.items():
        result["role"] = (
            "selected"
            if name == candidates[0].key
            else "reference"
            if name in config.references
            else "above_cv_floor"
        )
        result["cv_roc_auc_mean"] = float(cv[name]["cv_roc_auc_mean"]) if name in cv else None
        result["threshold_fitted_on"] = fit_split
    comparisons = compare_systems(
        [c.key for c in candidates], config.references, results, replicates, config
    )
    report = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "created_at": now(),
        "splits_read": [fit_split, confirm_split],
        "selection": {
            "path": str(out / "selection.json"),
            "sha256": sha256_of_file(out / "selection.json"),
        },
        "scorers_missing": missing,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "candidates_evaluated": len(candidates),
        "scope_rule": config.raw["confirmation"].get("scope_amendment"),
        "learned": learned,
        "results": results,
        "comparisons": comparisons,
        "comparison_rule": config.raw["confirmation"]["comparison_rule"],
        "label_semantics": config.verifier["benchmark"]["label_semantics"],
        "config": {"path": str(config.path), "sha256": sha256_of_file(config.path)},
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(target, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    by_candidate: dict[str, Record] = {}
    for entry in comparisons:
        by_candidate.setdefault(entry["candidate"], {})[
            f"{entry['metric']}|{entry['reference']}"
        ] = entry
    table: list[Record] = []
    for name, result in results.items():
        table_row: Record = {
            "system": name,
            "role": result["role"],
            "cv_roc_auc_mean": result["cv_roc_auc_mean"],
            "roc_auc": result["roc_auc"],
            "roc_auc_low": result["roc_auc_interval"]["low"],
            "roc_auc_high": result["roc_auc_interval"]["high"],
            "cohen_kappa": result["cohen_kappa"],
            "cohen_kappa_low": result["cohen_kappa_interval"]["low"],
            "cohen_kappa_high": result["cohen_kappa_interval"]["high"],
        }
        for reference in config.references:
            found = by_candidate.get(name, {}).get(f"roc_auc|{reference}")
            table_row[f"delta_auc_vs_{reference}"] = None if found is None else found["delta"]
            table_row[f"p_holm_vs_{reference}"] = None if found is None else found["p_holm"]
        table.append(table_row)
    table.sort(key=lambda row: -row["roc_auc"])
    write_csv(out / "confirmation.csv", table)
    print(f"{len(candidates)} candidates evaluated on {confirm_split}", flush=True)


def top_confirmed(out: Path, count: int) -> list[str]:
    with open(out / "confirmation.csv") as f:
        rows = [row for row in csv.DictReader(f) if row["role"] != "reference"]
    rows.sort(key=lambda row: -float(row["roc_auc"]))
    return [row["system"] for row in rows[:count]]


def check_train_copy(fit_data: dict[str, ScorerData], copy: dict[str, ScorerData]) -> Record:
    checked: Record = {}
    for key, scorer in copy.items():
        gaps = [
            float(np.max(np.abs(values - fit_data[key].stored[name])))
            for name, values in scorer.stored.items()
        ]
        worst = max(gaps)
        if worst > MATCH_TOLERANCE:
            raise SystemExit(f"{key}: train scores of the test run differ from explore by {worst}")
        checked[key] = {"scores_compared": len(gaps), "max_abs_gap": worst}
    return checked


def command_final_test(args: argparse.Namespace, config: ExplorationConfig) -> None:
    final = config.raw["final_test"]
    out = output_dir(config, args.output_dir)
    target = out / "final_test.json"
    if target.exists():
        raise SystemExit(f"{target} exists: the final test runs once")
    keys = list(final["candidates"])
    if keys != top_confirmed(out, len(keys)):
        raise SystemExit("final_test.candidates differs from the top rows of confirmation.csv")
    references = tuple(final["references"])
    fit_split = config.raw["fit_split"]
    _, fit_labels, fit_hearings, fit_data, _ = load_all(config, fit_split)
    fit_ids, _, _ = load_labels(config, fit_split)
    ids, labels, hearings = load_labels(config, "test")
    run = final["score_run"]
    test_data: dict[str, ScorerData] = {}
    train_copy: dict[str, ScorerData] = {}
    for key, (kind, _) in config.scorers.items():
        scorer = load_scorer(key, kind, run, "test", ids, config)
        if scorer is None:
            continue
        test_data[key] = scorer
        copy = load_scorer(key, kind, run, fit_split, fit_ids, config)
        if copy is None or key not in fit_data:
            raise SystemExit(f"{key}: the test run lacks its {fit_split} scores")
        train_copy[key] = copy
    reading = check_reading(config, test_data)
    same_train = check_train_copy(fit_data, train_copy)
    built = {
        c.key: c
        for c in [*fixed_candidates(config, fit_data), *learned_candidates(config, fit_data)]
    }
    systems: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    learned: Record = {}
    for key in keys:
        fit_scores, scores, model = system_scores(
            built[key], fit_data, test_data, fit_labels, fit_hearings, config
        )
        systems[key] = (fit_scores, scores)
        if model:
            learned[key] = model
    for reference in references:
        systems[reference] = (
            resolve_reference(reference, config, fit_data),
            resolve_reference(reference, config, test_data),
        )
    results, replicates = evaluate_split(systems, fit_labels, labels, hearings, config)
    for name, result in results.items():
        result["role"] = (
            "primary" if name == final["primary"] else "reference" if name in references else "top5"
        )
        result["threshold_fitted_on"] = fit_split
    comparisons = compare_systems(keys, references, results, replicates, config)
    report = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "stage": "final_test",
        "created_at": now(),
        "splits_read": [fit_split, "test"],
        "declared": final["declared"],
        "primary": final["primary"],
        "score_run": run,
        "files_read": {
            key: {"path": str(s.path), "sha256": sha256_of_file(s.path)}
            for key, s in test_data.items()
        },
        "reading_check": reading,
        "train_copy_check": same_train,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "learned": learned,
        "results": results,
        "comparisons": comparisons,
        "comparison_rule": final["comparison_rule"],
        "label_semantics": config.verifier["benchmark"]["label_semantics"],
        "config": {"path": str(config.path), "sha256": sha256_of_file(config.path)},
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(target, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"results": results, "comparisons": comparisons}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploration of the stored NLI verifier scores: grouped cross-validation "
        "on train to choose one candidate, then one confirmation on validation."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nli_verifier_exploration.toml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "explore", help="cross-validate every candidate on train (reads train only)"
    )
    commands.add_parser("confirm", help="evaluate the selected candidate once on validation")
    commands.add_parser("final-test", help="evaluate the declared final candidates once on test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_exploration_config(args.config)
    if args.command == "explore":
        command_explore(args, config)
    elif args.command == "confirm":
        command_confirm(args, config)
    else:
        command_final_test(args, config)


if __name__ == "__main__":
    main()
