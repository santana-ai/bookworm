import argparse
import importlib.metadata
import platform
import shlex
import subprocess
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import rank_bm25
import scipy
import sentence_transformers
import sklearn
import torch
import transformers

from utils import (
    build_udvs,
    calibrate_threshold,
    dataset_io,
    decision_models,
    decision_scoring,
    hub_offline,
    retrieval_data,
    retrieval_models,
    retrieval_stats,
    retrieval_store,
    udv_pipeline,
)
from utils.build_udvs import load_config as load_udv_config
from utils.build_udvs import seed_everything, select_device
from utils.calibrate_threshold import load_split_lookup
from utils.dataset_io import (
    load_gated_jsonl,
    load_jsonl,
    module_path,
    sha256_of_file,
    write_json,
    write_jsonl,
)
from utils.decision_models import LayaSpec
from utils.decision_scoring import BatteryError, parse_battery
from utils.hub_offline import enforce_offline, offline_environment, offline_state
from utils.retrieval_data import (
    BENCHES,
    SPLIT_NAMES,
    UNIT_KINDS,
    HearingData,
    Query,
    Unit,
    build_queries,
    get_context,
    load_hearing,
)
from utils.retrieval_models import (
    Bm25Retriever,
    DecisionRerankRetriever,
    DecisionRerankSpec,
    DenseRetriever,
    DenseSpec,
    Ranking,
    RerankRetriever,
    RerankSpec,
    Retriever,
    RrfRetriever,
    Runtime,
    TfidfRetriever,
)
from utils.retrieval_stats import bootstrap_mean, holm, mcnemar_exact, sign_flip_test, stream_rng

Record = dict[str, Any]

SPARSE_KINDS = ("tfidf", "bm25")
RETRIEVER_KINDS = (*SPARSE_KINDS, "dense", "rrf", "rerank", "decision_rerank")
DECISION_TRUNCATED_KEY = "premise"
FIT_SCOPES = ("speaker", "hearing")
CODE_MODULES = (
    udv_pipeline,
    build_udvs,
    calibrate_threshold,
    dataset_io,
    decision_models,
    decision_scoring,
    hub_offline,
    retrieval_data,
    retrieval_models,
    retrieval_stats,
    retrieval_store,
)
DEFINITIONS: Record = {
    "rank": "1-based position of the first relevant unit in the retriever's order",
    "rank_optimistic": "1 + number of non-relevant units scored strictly above the best relevant",
    "rank_pessimistic": "1 + number of non-relevant units scored at or above the best relevant",
    "acc_at_k": "share of queries whose rank is at most k",
    "mrr": "mean of 1 / rank",
    "random_acc_at_1": "mean over queries of n_relevant / n_units, the acc@1 of a uniform pick",
    "acc_at_1_minus_random": "acc@1 minus random_acc_at_1, comparable across unit sizes",
    "top1_chars": "characters of the top-1 unit, the evidence the retriever would hand over",
    "ci": "percentile interval over bootstrap replicates that resample whole hearings",
    "mcnemar": (
        "exact two-sided binomial test on the queries where exactly one of the baseline and "
        "the other retriever has the relevant unit at rank 1; it treats queries as independent, "
        "so the hearing-bootstrap interval of the acc@1 difference is reported next to it"
    ),
    "mrr_test": "two-sided sign-flip permutation test that flips whole hearings",
    "lift": "per query: 1 if the rank-1 unit is relevant, else 0, minus n_relevant / n_units",
    "retrievers_within_unit": (
        "E1: each retriever against evaluation.baseline_retriever on the same unit kind and the "
        "same queries (exact McNemar on acc@1, hearing sign-flip on the reciprocal rank); "
        "retrievers are never tested across unit kinds"
    ),
    "units_within_retriever": (
        "E2: each unit kind against evaluation.baseline_unit for the same retriever and the same "
        "queries; the tested quantity is the paired difference of lift (hearing-bootstrap "
        "interval, hearing sign-flip test), reported next to the paired difference of top1_chars; "
        "raw acc@1 and MRR are shown but not tested across unit kinds, because a larger unit is "
        "relevant by chance more often (random_acc_at_1) and a query whose units are all "
        "relevant cannot be missed"
    ),
    "holm_family": (
        "the set of p-values adjusted together: E1|<bench>|<split group>|<unit>|<test> and "
        "E2|<bench>|<split group>|<retriever>|<test>; each adjusted p-value carries the name of "
        "its family and each family its size"
    ),
}
E1_TESTS = (("acc_at_1", "mcnemar"), ("mrr", "sign_flip"))
E2_TESTS = (("acc_at_1_minus_random", "sign_flip"),)


@dataclass(frozen=True)
class ExperimentConfig:
    lds_path: Path
    lds_sha256: str
    manifest_path: Path
    default_splits: tuple[str, ...]
    benches: dict[str, Record]
    unit_kinds: tuple[str, ...]
    window_sizes: dict[str, int]
    retrievers: dict[str, Record]
    evaluation: Record
    cache: Record
    run: Record
    queue_steps: list[Record]
    source: Record = field(default_factory=dict)


def retriever_ids(retrievers: dict[str, Record]) -> dict[str, tuple[str, str | None]]:
    ids: dict[str, tuple[str, str | None]] = {}
    for name, spec in retrievers.items():
        if spec["kind"] in SPARSE_KINDS:
            for scope in spec["fit_scopes"]:
                ids[f"{name}_{scope}"] = (name, scope)
        else:
            ids[name] = (name, None)
    return ids


def validate_retrievers(retrievers: dict[str, Record], evaluation: Record) -> None:
    for name, spec in retrievers.items():
        if spec["kind"] not in RETRIEVER_KINDS:
            raise SystemExit(f"retrievers.{name}.kind must be one of {RETRIEVER_KINDS}")
        if spec["kind"] in SPARSE_KINDS and not set(spec["fit_scopes"]) <= set(FIT_SCOPES):
            raise SystemExit(f"retrievers.{name}.fit_scopes must be among {FIT_SCOPES}")
    ids = retriever_ids(retrievers)
    for name, spec in retrievers.items():
        references = spec.get("components", []) + ([spec["base"]] if "base" in spec else [])
        unknown = [reference for reference in references if reference not in ids]
        if unknown:
            raise SystemExit(f"retrievers.{name} references unknown retrievers {unknown}")
    if evaluation["baseline_retriever"] not in ids:
        raise SystemExit("evaluation.baseline_retriever is not a configured retriever")


def validate_decision_retrievers(retrievers: dict[str, Record], cache: Record) -> None:
    for name, spec in retrievers.items():
        if spec["kind"] == "decision_rerank":
            decision_rerank_spec(name, spec, "cpu", Path(cache["decision_dir"]))


def validate_queue(
    steps: list[Record],
    retrievers: dict[str, Record],
    kinds: tuple[str, ...],
    benches: tuple[str, ...],
) -> None:
    ids = retriever_ids(retrievers)
    names = set(ids) | {base for base, _ in ids.values()}
    for number, step in enumerate(steps):
        if set(step) - {"retrievers", "units", "benches", "device"} or not step.get("retrievers"):
            raise SystemExit(f"queue.steps[{number}] takes retrievers, units, benches, device")
        unknown = [name for name in step["retrievers"] if name not in names]
        unknown += [kind for kind in step.get("units", []) if kind not in kinds]
        unknown += [bench for bench in step.get("benches", []) if bench not in benches]
        if unknown:
            raise SystemExit(f"queue.steps[{number}] names unknown items {unknown}")


def load_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    default_splits = tuple(raw["splits"]["default"])
    if "test" in default_splits:
        raise SystemExit("splits.default must never include test")
    if not set(default_splits) <= set(SPLIT_NAMES):
        raise SystemExit(f"splits.default must be among {SPLIT_NAMES}")
    kinds = tuple(raw["units"]["kinds"])
    if not set(kinds) <= set(UNIT_KINDS) or "sentence" not in kinds:
        raise SystemExit(f"units.kinds must be among {UNIT_KINDS} and include sentence")
    if raw["units"]["stride"] != 1:
        raise SystemExit("units.stride must be 1")
    window_sizes = {kind: int(size) for kind, size in raw["units"]["window_sizes"].items()}
    if set(window_sizes) != {"window2", "window3"} or any(
        size < 2 for size in window_sizes.values()
    ):
        raise SystemExit("units.window_sizes must define window2 and window3 with sizes >= 2")
    if not set(raw["benchmarks"]) <= set(BENCHES):
        raise SystemExit(f"benchmarks must be among {BENCHES}")
    validate_retrievers(raw["retrievers"], raw["evaluation"])
    validate_decision_retrievers(raw["retrievers"], raw["cache"])
    if raw["evaluation"]["baseline_unit"] not in kinds:
        raise SystemExit("evaluation.baseline_unit must be one of units.kinds")
    validate_queue(raw["queue"]["steps"], raw["retrievers"], kinds, tuple(raw["benchmarks"]))
    return ExperimentConfig(
        lds_path=Path(raw["dataset"]["lds_path"]),
        lds_sha256=raw["dataset"]["lds_sha256"],
        manifest_path=Path(raw["splits"]["manifest_path"]),
        default_splits=default_splits,
        benches=raw["benchmarks"],
        unit_kinds=kinds,
        window_sizes=window_sizes,
        retrievers=raw["retrievers"],
        evaluation=raw["evaluation"],
        cache=raw["cache"],
        run=raw["run"],
        queue_steps=raw["queue"]["steps"],
        source=raw,
    )


def parse_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_splits(
    requested: list[str] | None, final_test: bool, config: ExperimentConfig
) -> tuple[str, ...]:
    splits = tuple(requested) if requested else config.default_splits
    if not set(splits) <= set(SPLIT_NAMES) or len(set(splits)) != len(splits):
        raise SystemExit(f"--splits must list distinct names among {SPLIT_NAMES}")
    if "test" in splits and not final_test:
        raise SystemExit("the test split is refused without --final-test")
    return splits


def resolve_retrievers(requested: list[str] | None, config: ExperimentConfig) -> list[str]:
    ids = retriever_ids(config.retrievers)
    if not requested:
        return list(ids)
    selected: list[str] = []
    for name in requested:
        matches = [rid for rid, (base, _) in ids.items() if rid == name or base == name]
        if not matches:
            raise SystemExit(f"unknown retriever {name!r}; known: {sorted(ids)}")
        selected.extend(match for match in matches if match not in selected)
    return selected


def resolve_choice(
    requested: list[str] | None, allowed: tuple[str, ...], flag: str
) -> tuple[str, ...]:
    if not requested:
        return allowed
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise SystemExit(f"{flag} {unknown} not among {allowed}")
    return tuple(item for item in allowed if item in requested)


def dense_spec(name: str, spec: Record) -> DenseSpec:
    return DenseSpec(
        name=name,
        model=spec["model"],
        revision=spec["revision"],
        query_prefix=spec["query_prefix"],
        passage_prefix=spec["passage_prefix"],
        max_seq_length=spec["max_seq_length"],
        batch_size=spec["batch_size"],
        token_budget=spec["token_budget"],
        production_cache=spec["production_cache"],
        source=spec,
    )


def rerank_spec(name: str, spec: Record) -> RerankSpec:
    return RerankSpec(
        name=name,
        model=spec["model"],
        revision=spec["revision"],
        base=spec["base"],
        top_k=spec["top_k"],
        max_length=spec["max_length"],
        batch_size=spec["batch_size"],
        source=spec,
    )


def decision_rerank_spec(
    name: str, spec: Record, device: str, cache_dir: Path
) -> DecisionRerankSpec:
    with open(spec["decision_config"], "rb") as f:
        raw = tomllib.load(f)
    scorer = raw.get("scorers", {}).get(spec["scorer"])
    where = f"retrievers.{name}: {spec['decision_config']} scorers.{spec['scorer']}"
    if scorer is None or scorer["kind"] != "laya":
        raise SystemExit(f"{where} must be a laya scorer")
    if scorer["language"] != "pt":
        raise SystemExit(f"{where} reads a translation; only language pt is supported")
    if scorer["revision"] != spec["revision"]:
        raise SystemExit(f"{where} revision {scorer['revision']} != {spec['revision']}")
    if spec["question"] not in scorer["questions"]:
        raise SystemExit(f"{where} does not ask {spec['question']!r}")
    try:
        battery = parse_battery(raw["decision_battery"])
    except BatteryError as error:
        raise SystemExit(f"{where}: decision_battery: {error}") from error
    return DecisionRerankSpec(
        name=name,
        scorer=spec["scorer"],
        base=spec["base"],
        top_k=spec["top_k"],
        question=battery.questions[spec["question"]],
        laya=LayaSpec(
            repo_id=scorer["name"],
            revision=scorer["revision"],
            subfolder=scorer["subfolder"],
            device=device,
            batch_size=int(scorer["batch_size"]),
            max_length=int(scorer["max_length"]),
            head_max_length=int(scorer["head_max_length"]),
            truncate_key=DECISION_TRUNCATED_KEY,
            cache_dir=cache_dir,
        ),
        source=spec,
    )


def build_retriever(rid: str, config: ExperimentConfig, runtime: Runtime) -> Retriever:
    name, scope = retriever_ids(config.retrievers)[rid]
    spec = config.retrievers[name]
    kind = spec["kind"]
    if kind == "tfidf" and scope is not None:
        return TfidfRetriever(rid, {k: v for k, v in spec.items() if k != "kind"}, scope)
    if kind == "bm25" and scope is not None:
        return Bm25Retriever(rid, {k: spec[k] for k in ("k1", "b", "epsilon")}, scope)
    if kind == "dense":
        return DenseRetriever(dense_spec(name, spec), runtime)
    if kind == "rrf":
        components = [
            build_retriever(component, config, runtime) for component in spec["components"]
        ]
        return RrfRetriever(rid, components, spec["k"])
    base = build_retriever(spec["base"], config, runtime)
    if kind == "decision_rerank":
        decision = decision_rerank_spec(
            name, spec, runtime.device, Path(config.cache["decision_dir"])
        )
        return DecisionRerankRetriever(decision, base, runtime)
    return RerankRetriever(rerank_spec(name, spec), base, runtime)


def load_bench_rows(
    path: Path, split_of: dict[int, str], splits: tuple[str, ...]
) -> tuple[dict[int, list[Record]], Record]:
    by_hearing: dict[int, list[Record]] = {}
    counts: Counter[str] = Counter()
    for row in load_jsonl(path):
        counts["rows_read"] += 1
        if split_of.get(row["hearing_id"]) != row["split"]:
            raise SystemExit(f"{row['id']}: split {row['split']!r} differs from the manifest")
        if row["split"] not in splits:
            counts["rows_outside_requested_splits"] += 1
            continue
        counts[f"rows_{row['split']}"] += 1
        by_hearing.setdefault(row["hearing_id"], []).append(row)
    return by_hearing, {"path": str(path), "sha256": sha256_of_file(path), **counts}


@dataclass
class Workload:
    hearings: list[HearingData]
    queries: dict[int, list[Query]]
    checks: dict[str, Counter[str]]
    span_checks: Counter[str]
    sources: Record
    masked_texts_by_hearing: dict[int, list[str]]


def build_workload(
    config: ExperimentConfig,
    splits: tuple[str, ...],
    benches: tuple[str, ...],
    limit_hearings: int | None,
    limit_queries: int | None,
) -> Workload:
    lds = load_gated_jsonl(config.lds_path, config.lds_sha256)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    rows: dict[str, dict[int, list[Record]]] = {}
    bench_sources: Record = {}
    for bench in benches:
        rows[bench], bench_sources[bench] = load_bench_rows(
            Path(config.benches[bench]["path"]), split_of, splits
        )
    records = [
        record
        for record in lds
        if split_of[record["id"]] in splits and any(record["id"] in rows[b] for b in benches)
    ]
    if limit_hearings is not None:
        records = records[:limit_hearings]
    checks: dict[str, Counter[str]] = {bench: Counter() for bench in benches}
    span_checks: Counter[str] = Counter()
    hearings: list[HearingData] = []
    queries: dict[int, list[Query]] = {}
    remaining = {bench: limit_queries for bench in benches}
    for record in records:
        if all(remaining[bench] == 0 for bench in benches):
            break
        hearing = load_hearing(record, split_of[record["id"]])
        span_checks.update(hearing.span_checks)
        get_context(hearing, None, config.window_sizes)
        hearing_queries: list[Query] = []
        for bench in benches:
            built = build_queries(
                bench,
                rows[bench].get(record["id"], []),
                hearing,
                config.window_sizes,
                checks[bench],
            )
            limit = remaining[bench]
            if limit is not None:
                built = built[:limit]
                remaining[bench] = limit - len(built)
            hearing_queries.extend(built)
        if hearing_queries:
            hearings.append(hearing)
            queries[hearing.hearing_id] = hearing_queries
    problems = {
        bench: {key: count for key, count in counter.items() if "mismatch" in key}
        for bench, counter in checks.items()
    }
    if any(problems.values()):
        raise SystemExit(f"benchmark rows disagree with the current pipeline: {problems}")
    masked = {
        hearing_id: [row["masked_opinion"] for row in hearing_rows]
        for hearing_id, hearing_rows in rows.get("masked_quotes", {}).items()
    }
    return Workload(
        hearings=hearings,
        queries=queries,
        checks=checks,
        span_checks=span_checks,
        sources={
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
            "splits": split_source,
            "benchmarks": bench_sources,
        },
        masked_texts_by_hearing=masked,
    )


def unit_mean_chars(units: list[Unit]) -> float:
    return float(np.mean([len(unit.text) for unit in units])) if units else 0.0


def evaluate_query(
    query: Query,
    kind: str,
    units: list[Unit],
    ranking: Ranking,
    retriever_id: str,
    mean_chars: float,
    evaluation: Record,
) -> Record:
    relevant = np.array(query.relevant[kind], dtype=np.int64)
    is_relevant = np.zeros(len(units), dtype=bool)
    is_relevant[relevant] = True
    first_position = int(np.flatnonzero(is_relevant[ranking.order])[0])
    first_index = int(ranking.order[first_position])
    best = ranking.sort_key[relevant].max()
    others = ranking.sort_key[~is_relevant]
    decimals = evaluation["score_decimals"]
    top = ranking.order[: evaluation["top_n_stored"]]
    top1 = units[int(ranking.order[0])]
    return {
        "bench": query.bench,
        "query_id": query.query_id,
        "split": query.split,
        "hearing_id": query.hearing_id,
        "retriever": retriever_id,
        "unit": kind,
        "n_units": len(units),
        "n_relevant": len(relevant),
        "rank": first_position + 1,
        "rank_optimistic": 1 + int((others > best).sum()),
        "rank_pessimistic": 1 + int((others >= best).sum()),
        "random_acc_at_1": round(len(relevant) / len(units), 6),
        "query_chars": len(query.text),
        "top1_id": top1.unit_id,
        "top1_chars": len(top1.text),
        "mean_unit_chars": round(mean_chars, 1),
        "first_relevant_id": units[first_index].unit_id,
        "top": [
            [
                units[int(index)].unit_id,
                None
                if np.isnan(ranking.display[index])
                else round(float(ranking.display[index]), decimals),
            ]
            for index in top
        ],
    }


def query_file(run_dir: Path, bench: str, kind: str, retriever_id: str, split: str) -> Path:
    return run_dir / "queries" / bench / kind / retriever_id / f"{split}.jsonl"


def environment() -> Record:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rank_bm25": getattr(rank_bm25, "__version__", "0.2.2 (no __version__ attribute)"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "laya": importlib.metadata.version("laya"),
        "platform": platform.platform(),
    }


def code_hashes() -> Record:
    files = {f"utils/{module_path(module).name}": module_path(module) for module in CODE_MODULES}
    files["utils/retrieval_experiments.py"] = Path(__file__)
    return {name: sha256_of_file(path) for name, path in sorted(files.items())}


def run_name_for(args: argparse.Namespace, config: ExperimentConfig) -> str:
    if args.run_name:
        return args.run_name
    limited = args.limit_queries is not None or args.limit_hearings is not None
    return f"{config.run['run_name']}_smoke" if limited else config.run["run_name"]


def run_retriever(
    retriever_id: str,
    workload: Workload,
    kinds: tuple[str, ...],
    config: ExperimentConfig,
    runtime: Runtime,
) -> tuple[list[Record], Record]:
    retriever = build_retriever(retriever_id, config, runtime)
    rows: list[Record] = []
    seconds: list[float] = []
    for number, hearing in enumerate(workload.hearings, start=1):
        started = time.perf_counter()
        queries = workload.queries[hearing.hearing_id]
        for kind in kinds:
            rankings = retriever.rank_hearing(hearing, kind, queries)
            means: dict[str, float] = {}
            for query in queries:
                units = hearing.contexts[query.context_key].units[kind]
                if query.context_key not in means:
                    means[query.context_key] = unit_mean_chars(units)
                rows.append(
                    evaluate_query(
                        query,
                        kind,
                        units,
                        rankings[query.query_id],
                        retriever_id,
                        means[query.context_key],
                        config.evaluation,
                    )
                )
        seconds.append(time.perf_counter() - started)
        print(
            f"[{retriever_id}] [{number}/{len(workload.hearings)}] hearing {hearing.hearing_id}: "
            f"{len(queries)} queries in {seconds[-1]:.1f}s",
            flush=True,
        )
    retriever.close()
    timing = {
        "elapsed_seconds": round(sum(seconds), 1),
        "mean_seconds_per_hearing": round(float(np.mean(seconds)), 2) if seconds else None,
        "max_seconds_per_hearing": round(max(seconds), 2) if seconds else None,
    }
    return rows, {"details": retriever.describe(), "timing": timing}


def write_query_rows(
    rows: list[Record], run_dir: Path, retriever_id: str, splits: tuple[str, ...]
) -> list[Record]:
    grouped: dict[tuple[str, str, str], list[Record]] = {}
    for row in rows:
        grouped.setdefault((row["bench"], row["unit"], row["split"]), []).append(row)
    written = []
    for (bench, kind, split), members in sorted(grouped.items()):
        if split not in splits:
            raise SystemExit(f"row of split {split} outside the requested splits")
        path = query_file(run_dir, bench, kind, retriever_id, split)
        write_jsonl(members, path)
        written.append({"path": str(path), "rows": len(members), "sha256": sha256_of_file(path)})
    return written


def workload_counts(workload: Workload) -> Record:
    counts: Record = {}
    for hearing in workload.hearings:
        for query in workload.queries[hearing.hearing_id]:
            key = f"{query.bench}.{query.split}"
            counts.setdefault(key, {"queries": 0, "hearings": set()})
            counts[key]["queries"] += 1
            counts[key]["hearings"].add(query.hearing_id)
    return {
        key: {"queries": value["queries"], "hearings": len(value["hearings"])}
        for key, value in sorted(counts.items())
    }


def command_run(args: argparse.Namespace, config: ExperimentConfig) -> None:
    splits = resolve_splits(parse_list(args.splits), args.final_test, config)
    selected = resolve_retrievers(parse_list(args.retrievers), config)
    kinds = resolve_choice(parse_list(args.units), config.unit_kinds, "--units")
    benches = resolve_choice(parse_list(args.benches), tuple(config.benches), "--benches")
    run_name = run_name_for(args, config)
    run_dir = Path(config.run["output_dir"]) / run_name
    code = code_hashes()
    offline = enforce_offline() if config.run["hf_hub_offline"] else offline_state()
    seed_everything(config.run["seed"])
    device = select_device(args.device or config.run["device"])
    started = time.perf_counter()
    workload = build_workload(config, splits, benches, args.limit_hearings, args.limit_queries)
    load_seconds = time.perf_counter() - started
    total = sum(len(queries) for queries in workload.queries.values())
    print(
        f"{len(workload.hearings)} hearings, {total} queries ({', '.join(splits)}) | "
        f"retrievers {selected} | units {list(kinds)} | device {device} | "
        f"loaded in {load_seconds:.1f}s",
        flush=True,
    )
    runtime = Runtime(
        device=device,
        embeddings_dir=Path(config.cache["embeddings_dir"]),
        rerank_dir=Path(config.cache["rerank_dir"]),
        shard_size=config.cache["shard_size"],
        udv_config=load_udv_config(Path(config.cache["production_config"])),
        masked_texts_by_hearing=workload.masked_texts_by_hearing,
    )
    for retriever_id in selected:
        rows, details = run_retriever(retriever_id, workload, kinds, config, runtime)
        written = write_query_rows(rows, run_dir, retriever_id, splits)
        report = {
            "run_name": run_name,
            "retriever": retriever_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "splits_used": list(splits),
            "final_test": args.final_test,
            "benches": list(benches),
            "units": list(kinds),
            "limits": {"hearings": args.limit_hearings, "queries_per_bench": args.limit_queries},
            "device": device,
            "hub_offline": {"configured": config.run["hf_hub_offline"], **offline},
            "workload": {
                "hearings": len(workload.hearings),
                "queries": workload_counts(workload),
                "query_checks": {
                    bench: dict(counter) for bench, counter in workload.checks.items()
                },
                "sentence_span_checks": dict(workload.span_checks),
                "load_seconds": round(load_seconds, 1),
            },
            "retriever_details": details["details"],
            "timing": details["timing"],
            "outputs": written,
            "sources": workload.sources,
            "code": code,
            "environment": environment(),
            "config": config.source,
        }
        name = f"{retriever_id}__{'+'.join(splits)}__{'+'.join(kinds)}__{'+'.join(benches)}"
        write_json(report, run_dir / "runs" / f"{name}_report.json")
        print(f"[{retriever_id}] {len(rows)} rows, {details['timing']}", flush=True)


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def interval(result: Record) -> Record:
    return {key: rounded(result[key]) for key in ("point", "low", "high")}


def summarize_rows(rows: list[Record], label: str, evaluation: Record) -> Record:
    seed, samples = evaluation["seed"], evaluation["bootstrap_samples"]
    level = evaluation["confidence_level"]
    hearings = np.array([row["hearing_id"] for row in rows])
    ranks = np.array([row["rank"] for row in rows], dtype=np.float64)
    optimistic = np.array([row["rank_optimistic"] for row in rows])
    pessimistic = np.array([row["rank_pessimistic"] for row in rows])
    random_acc = np.array([row["random_acc_at_1"] for row in rows])
    top1_chars = np.array([row["top1_chars"] for row in rows], dtype=np.float64)
    metrics: Record = {}
    for k in evaluation["ks"]:
        metrics[f"acc_at_{k}"] = interval(
            bootstrap_mean(
                (ranks <= k).astype(np.float64),
                hearings,
                samples,
                level,
                stream_rng(seed, f"{label}|acc{k}"),
            )
        )
    metrics["mrr"] = interval(
        bootstrap_mean(1.0 / ranks, hearings, samples, level, stream_rng(seed, f"{label}|mrr"))
    )
    metrics["acc_at_1_minus_random"] = interval(
        bootstrap_mean(
            (ranks <= 1) - random_acc, hearings, samples, level, stream_rng(seed, f"{label}|lift")
        )
    )
    return {
        "queries": len(rows),
        "hearings": int(len(np.unique(hearings))),
        **metrics,
        "random_acc_at_1": rounded(random_acc.mean()),
        "acc_at_1_tie_bounds": [
            rounded((pessimistic <= 1).mean()),
            rounded((optimistic <= 1).mean()),
        ],
        "queries_with_tie_at_first_relevant": int((optimistic != pessimistic).sum()),
        "trivial_queries": int(sum(1 for row in rows if row["n_relevant"] == row["n_units"])),
        "mean_units": rounded(np.mean([row["n_units"] for row in rows]), 2),
        "mean_relevant": rounded(np.mean([row["n_relevant"] for row in rows]), 2),
        "top1_chars": {
            "mean": rounded(top1_chars.mean(), 1),
            "median": rounded(np.median(top1_chars), 1),
        },
        "mean_unit_chars": rounded(np.mean([row["mean_unit_chars"] for row in rows]), 1),
    }


def paired_rows(baseline: list[Record], other: list[Record]) -> list[tuple[Record, Record]]:
    by_id = {row["query_id"]: row for row in baseline}
    return [(by_id[row["query_id"]], row) for row in other if row["query_id"] in by_id]


def column(pairs: list[tuple[Record, Record]], side: int, name: str) -> np.ndarray:
    return np.array([pair[side][name] for pair in pairs], dtype=np.float64)


def compare_retrievers(
    baseline: list[Record], other: list[Record], label: str, evaluation: Record
) -> Record:
    paired = paired_rows(baseline, other)
    seed, level = evaluation["seed"], evaluation["confidence_level"]
    hearings = np.array([pair[0]["hearing_id"] for pair in paired])
    base_rank, other_rank = column(paired, 0, "rank"), column(paired, 1, "rank")
    base_hit, other_hit = base_rank <= 1, other_rank <= 1
    rr_diff = 1.0 / other_rank - 1.0 / base_rank
    return {
        "queries_paired": len(paired),
        "queries_only_in_other": len(other) - len(paired),
        "acc_at_1": {
            "baseline": rounded(base_hit.mean()),
            "other": rounded(other_hit.mean()),
            "difference": interval(
                bootstrap_mean(
                    other_hit.astype(np.float64) - base_hit,
                    hearings,
                    evaluation["bootstrap_samples"],
                    level,
                    stream_rng(seed, f"{label}|acc_diff"),
                )
            ),
            "mcnemar": mcnemar_exact(base_hit, other_hit),
        },
        "mrr": {
            "baseline": rounded((1.0 / base_rank).mean()),
            "other": rounded((1.0 / other_rank).mean()),
            "difference": interval(
                bootstrap_mean(
                    rr_diff,
                    hearings,
                    evaluation["bootstrap_samples"],
                    level,
                    stream_rng(seed, f"{label}|mrr_diff"),
                )
            ),
            "sign_flip": sign_flip_test(
                rr_diff,
                hearings,
                evaluation["permutation_samples"],
                stream_rng(seed, f"{label}|flip"),
            ),
        },
        "top1_chars_mean": {
            "baseline": rounded(column(paired, 0, "top1_chars").mean(), 1),
            "other": rounded(column(paired, 1, "top1_chars").mean(), 1),
        },
    }


def compare_units(
    reference: list[Record], other: list[Record], label: str, evaluation: Record
) -> Record:
    paired = paired_rows(reference, other)
    seed, level = evaluation["seed"], evaluation["confidence_level"]
    samples = evaluation["bootstrap_samples"]
    hearings = np.array([pair[0]["hearing_id"] for pair in paired])
    hits = [(column(paired, side, "rank") <= 1).astype(np.float64) for side in (0, 1)]
    randoms = [column(paired, side, "random_acc_at_1") for side in (0, 1)]
    lifts = [hit - random for hit, random in zip(hits, randoms, strict=True)]
    lift_diff = lifts[1] - lifts[0]
    chars = [column(paired, side, "top1_chars") for side in (0, 1)]
    trivial = [
        sum(1 for pair in paired if pair[side]["n_relevant"] == pair[side]["n_units"])
        for side in (0, 1)
    ]
    return {
        "queries_paired": len(paired),
        "queries_only_in_other": len(other) - len(paired),
        "acc_at_1_minus_random": {
            "reference": rounded(lifts[0].mean()),
            "other": rounded(lifts[1].mean()),
            "difference": interval(
                bootstrap_mean(
                    lift_diff, hearings, samples, level, stream_rng(seed, f"{label}|lift_diff")
                )
            ),
            "sign_flip": sign_flip_test(
                lift_diff,
                hearings,
                evaluation["permutation_samples"],
                stream_rng(seed, f"{label}|lift_flip"),
            ),
        },
        "untested_descriptives": {
            "acc_at_1": {"reference": rounded(hits[0].mean()), "other": rounded(hits[1].mean())},
            "random_acc_at_1": {
                "reference": rounded(randoms[0].mean()),
                "other": rounded(randoms[1].mean()),
            },
            "trivial_queries": {"reference": trivial[0], "other": trivial[1]},
        },
        "top1_chars": {
            "reference": rounded(chars[0].mean(), 1),
            "other": rounded(chars[1].mean(), 1),
            "difference": interval(
                bootstrap_mean(
                    chars[1] - chars[0],
                    hearings,
                    samples,
                    level,
                    stream_rng(seed, f"{label}|chars_diff"),
                )
            ),
        },
    }


def apply_holm(items: list[Record], tests: tuple[tuple[str, str], ...], family: str) -> Record:
    sizes: Record = {}
    for section, test in tests:
        name = f"{family}|{section}.{test}"
        p_values = [item[section][test]["p_value"] for item in items]
        for item, adjusted in zip(items, holm(p_values), strict=True):
            item[section][test]["p_holm"] = adjusted
            item[section][test]["holm_family"] = name
        sizes[name] = len(items)
    return sizes


def ordered_units(selected: dict[tuple[str, str], list[Record]]) -> list[str]:
    present = {unit for unit, _ in selected}
    return [kind for kind in UNIT_KINDS if kind in present]


def retriever_comparisons(
    selected: dict[tuple[str, str], list[Record]],
    baseline_retriever: str,
    prefix: str,
    evaluation: Record,
) -> Record:
    result: Record = {}
    for unit in ordered_units(selected):
        baseline = selected.get((unit, baseline_retriever))
        entry: Record = {"baseline": {"retriever": baseline_retriever, "unit": unit}}
        if baseline is None:
            result[unit] = {**entry, "baseline_missing": True, "holm_families": {}, "items": []}
            continue
        others = sorted(r for u, r in selected if u == unit and r != baseline_retriever)
        items = [
            {
                "retriever": retriever,
                "unit": unit,
                **compare_retrievers(
                    baseline,
                    selected[(unit, retriever)],
                    f"{prefix}|{unit}|{retriever}|vs",
                    evaluation,
                ),
            }
            for retriever in others
        ]
        families = apply_holm(items, E1_TESTS, f"E1|{prefix}|{unit}")
        result[unit] = {**entry, "holm_families": families, "items": items}
    return result


def unit_comparisons(
    selected: dict[tuple[str, str], list[Record]],
    baseline_unit: str,
    prefix: str,
    evaluation: Record,
) -> Record:
    result: Record = {}
    for retriever in sorted({r for _, r in selected}):
        units = [unit for unit in ordered_units(selected) if (unit, retriever) in selected]
        others = [unit for unit in units if unit != baseline_unit]
        if not others:
            continue
        reference = selected.get((baseline_unit, retriever))
        entry: Record = {"reference": {"retriever": retriever, "unit": baseline_unit}}
        if reference is None:
            result[retriever] = {
                **entry,
                "reference_missing": True,
                "holm_families": {},
                "items": [],
            }
            continue
        items = [
            {
                "retriever": retriever,
                "unit": unit,
                **compare_units(
                    reference,
                    selected[(unit, retriever)],
                    f"{prefix}|{retriever}|{unit}|vs_unit_{baseline_unit}",
                    evaluation,
                ),
            }
            for unit in others
        ]
        families = apply_holm(items, E2_TESTS, f"E2|{prefix}|{retriever}")
        result[retriever] = {**entry, "holm_families": families, "items": items}
    return result


def load_query_rows(
    run_dir: Path, splits: tuple[str, ...]
) -> tuple[dict[tuple[str, str, str], list[Record]], list[Record]]:
    grouped: dict[tuple[str, str, str], list[Record]] = {}
    files = []
    for path in sorted(run_dir.glob("queries/*/*/*/*.jsonl")):
        if path.stem not in splits:
            continue
        rows = load_jsonl(path)
        files.append({"path": str(path), "rows": len(rows), "sha256": sha256_of_file(path)})
        for row in rows:
            grouped.setdefault((row["bench"], row["unit"], row["retriever"]), []).append(row)
    return grouped, files


def split_groups(splits: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, tuple[str, ...]] = {split: (split,) for split in splits}
    if len(splits) > 1:
        groups["+".join(splits)] = splits
    return groups


def command_summarize(args: argparse.Namespace, config: ExperimentConfig) -> None:
    splits = resolve_splits(parse_list(args.splits), args.final_test, config)
    run_name = args.run_name or config.run["run_name"]
    run_dir = Path(config.run["output_dir"]) / run_name
    code = code_hashes()
    grouped, files = load_query_rows(run_dir, splits)
    if not grouped:
        raise SystemExit(f"no query rows for {splits} under {run_dir}")
    evaluation = config.evaluation
    baseline_key = (
        args.baseline_unit or evaluation["baseline_unit"],
        args.baseline_retriever or evaluation["baseline_retriever"],
    )
    summaries: Record = {}
    comparisons: Record = {}
    for bench in sorted({key[0] for key in grouped}):
        summaries[bench], comparisons[bench] = {}, {}
        for group, members in split_groups(splits).items():
            selected = {
                (unit, retriever): [row for row in rows if row["split"] in members]
                for (b, unit, retriever), rows in grouped.items()
                if b == bench
            }
            selected = {key: rows for key, rows in selected.items() if rows}
            summaries[bench][group] = {}
            for (unit, retriever), rows in sorted(selected.items()):
                label = f"{bench}|{group}|{unit}|{retriever}"
                summaries[bench][group].setdefault(unit, {})[retriever] = summarize_rows(
                    rows, label, evaluation
                )
            prefix = f"{bench}|{group}"
            comparisons[bench][group] = {
                "retrievers_within_unit": retriever_comparisons(
                    selected, baseline_key[1], prefix, evaluation
                ),
                "units_within_retriever": unit_comparisons(
                    selected, baseline_key[0], prefix, evaluation
                ),
            }
    report = {
        "run_name": run_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits_used": list(splits),
        "final_test": args.final_test,
        "baseline": {
            "retriever": baseline_key[1],
            "unit": baseline_key[0],
            "retriever_role": "reference retriever of E1, compared within each unit kind",
            "unit_role": "reference unit kind of E2, compared within each retriever",
            "configured": baseline_key
            == (evaluation["baseline_unit"], evaluation["baseline_retriever"]),
        },
        "question": (
            "given an opinion and the speaker's candidate units, which representation ranks a "
            "supporting passage first"
        ),
        "declaration": config.source.get("declarations", {}).get(run_name),
        "benchmarks": config.benches,
        "definitions": DEFINITIONS,
        "summaries": summaries,
        "comparisons": comparisons,
        "inputs": files,
        "run_reports": sorted(str(path) for path in (run_dir / "runs").glob("*_report.json")),
        "code": code,
        "environment": environment(),
        "config": config.source,
    }
    suffix = "final_test_report" if "test" in splits else "report"
    if report["baseline"]["configured"] is False:
        suffix = f"{suffix}_vs_{baseline_key[1]}_{baseline_key[0]}"
    write_json(report, run_dir / f"{run_name}_{suffix}.json")
    print_summary(summaries)


def print_summary(summaries: Record) -> None:
    for bench, groups in summaries.items():
        for group, units in groups.items():
            for unit, retrievers in units.items():
                for retriever, summary in retrievers.items():
                    acc, mrr = summary["acc_at_1"], summary["mrr"]
                    print(
                        f"{bench:13s} {group:18s} {unit:8s} {retriever:22s} "
                        f"n={summary['queries']:4d} acc@1={acc['point']:.3f} "
                        f"[{acc['low']:.3f},{acc['high']:.3f}] mrr={mrr['point']:.3f} "
                        f"random@1={summary['random_acc_at_1']:.3f} "
                        f"top1_chars={summary['top1_chars']['mean']}"
                    )


def queue_commands(args: argparse.Namespace, config: ExperimentConfig) -> list[list[str]]:
    base = [sys.executable, "-m", "utils.retrieval_experiments"]
    shared = ["--config", str(args.config)]
    if args.run_name:
        shared += ["--run-name", args.run_name]
    if args.splits:
        shared += ["--splits", args.splits]
    if args.final_test:
        shared += ["--final-test"]
    commands = []
    for step in config.queue_steps:
        command = [*base, "run", *shared, "--retrievers", ",".join(step["retrievers"])]
        for key in ("units", "benches"):
            if step.get(key):
                command += [f"--{key}", ",".join(step[key])]
        if step.get("device"):
            command += ["--device", step["device"]]
        commands.append(command)
    commands.append([*base, "summarize", *shared])
    return commands


def command_queue(args: argparse.Namespace, config: ExperimentConfig) -> None:
    commands = queue_commands(args, config)
    for number, command in enumerate(commands):
        print(f"[{number}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    for number, command in enumerate(commands):
        if number < args.from_step:
            continue
        print(f"[queue] step {number}: {shlex.join(command)}", flush=True)
        started = time.perf_counter()
        env = offline_environment() if config.run["hf_hub_offline"] else None
        subprocess.run(command, check=True, env=env)
        print(f"[queue] step {number} done in {time.perf_counter() - started:.0f}s", flush=True)


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval_experiments.toml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--splits", default=None, help="comma list; default from the config")
    parser.add_argument("--final-test", action="store_true", help="allow the test split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval harness: rank the speaker's candidate units for each opinion "
        "(masked-quote and NLI benchmarks) with sparse, dense, hybrid and reranking retrievers."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score queries and write per-query rows")
    add_shared(run)
    run.add_argument("--retrievers", default=None, help="comma list of ids or base names")
    run.add_argument("--units", default=None, help=f"comma list among {UNIT_KINDS}")
    run.add_argument("--benches", default=None, help=f"comma list among {BENCHES}")
    run.add_argument("--limit-queries", type=int, default=None, help="first N queries per bench")
    run.add_argument("--limit-hearings", type=int, default=None, help="first N hearings")
    run.add_argument("--device", default=None, help="override run.device (auto, mps, cpu)")
    summarize = commands.add_parser("summarize", help="metrics, intervals and paired tests")
    add_shared(summarize)
    summarize.add_argument("--baseline-retriever", default=None, help="default from the config")
    summarize.add_argument("--baseline-unit", default=None, help="default from the config")
    queue = commands.add_parser("queue", help="run every configured step, one model per process")
    add_shared(queue)
    queue.add_argument("--dry-run", action="store_true", help="only print the commands")
    queue.add_argument("--from-step", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "run":
        command_run(args, config)
    elif args.command == "summarize":
        command_summarize(args, config)
    else:
        command_queue(args, config)


if __name__ == "__main__":
    main()
