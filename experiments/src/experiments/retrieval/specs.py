"""The retriever entries of the harness config: their ids and the specs the retrievers take."""

import tomllib
from pathlib import Path
from typing import Any

from experiments.retrieval.models import DecisionRerankSpec, DenseSpec, RerankSpec
from experiments.verifier.decision_models import LayaSpec
from experiments.verifier.decision_scoring import BatteryError, parse_battery

Record = dict[str, Any]

SPARSE_KINDS = ("tfidf", "bm25")
RETRIEVER_KINDS = (*SPARSE_KINDS, "dense", "rrf", "rerank", "decision_rerank")
FIT_SCOPES = ("speaker", "hearing")
BM25_PARAMS = ("k1", "b", "epsilon")
DECISION_TRUNCATED_KEY = "premise"
DECISION_SCORER_KIND = "laya"
DECISION_LANGUAGE = "pt"


def retriever_ids(retrievers: dict[str, Record]) -> dict[str, tuple[str, str | None]]:
    """Each runnable retriever id with its config name and fit scope (sparse kinds only)."""
    ids: dict[str, tuple[str, str | None]] = {}
    for name, spec in retrievers.items():
        if spec["kind"] in SPARSE_KINDS:
            for scope in spec["fit_scopes"]:
                ids[f"{name}_{scope}"] = (name, scope)
        else:
            ids[name] = (name, None)
    return ids


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


def laya_spec(scorer: Record, device: str, cache_dir: Path) -> LayaSpec:
    return LayaSpec(
        repo_id=scorer["name"],
        revision=scorer["revision"],
        subfolder=scorer["subfolder"],
        device=device,
        batch_size=int(scorer["batch_size"]),
        max_length=int(scorer["max_length"]),
        head_max_length=int(scorer["head_max_length"]),
        truncate_key=DECISION_TRUNCATED_KEY,
        cache_dir=cache_dir,
    )


def check_decision_scorer(scorer: Record | None, spec: Record, where: str) -> Record:
    if scorer is None or scorer["kind"] != DECISION_SCORER_KIND:
        raise SystemExit(f"{where} must be a laya scorer")
    if scorer["language"] != DECISION_LANGUAGE:
        raise SystemExit(f"{where} reads a translation; only language pt is supported")
    if scorer["revision"] != spec["revision"]:
        raise SystemExit(f"{where} revision {scorer['revision']} != {spec['revision']}")
    if spec["question"] not in scorer["questions"]:
        raise SystemExit(f"{where} does not ask {spec['question']!r}")
    return scorer


def decision_rerank_spec(
    name: str, spec: Record, device: str, cache_dir: Path
) -> DecisionRerankSpec:
    """The Laya reranker of a config entry, read from the verifier config it points to."""
    with open(spec["decision_config"], "rb") as f:
        raw = tomllib.load(f)
    where = f"retrievers.{name}: {spec['decision_config']} scorers.{spec['scorer']}"
    scorer = check_decision_scorer(raw.get("scorers", {}).get(spec["scorer"]), spec, where)
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
        laya=laya_spec(scorer, device, cache_dir),
        source=spec,
    )
