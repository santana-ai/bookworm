"""NLI cross-encoder scoring with a signature-keyed logit cache."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bookworm import load_jsonl
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedConfig,
)

from experiments.common import cache_lock
from experiments.common.hub_offline import pinned_weights_file
from experiments.verifier.nli.benchmark import (
    PremiseUnit,
    concatenated_premise,
    distinct_items,
    unit_header,
)
from experiments.verifier.nli.config import (
    CHUNK_AGGREGATES,
    CONTRADICTION,
    ENTAILMENT,
    ScorerSpec,
    VerifierConfig,
)
from experiments.verifier.runtime import throughput

Record = dict[str, Any]

SCORE_MINIMUM = {"entailment": 0.0, "not_contradiction": 0.0, "cosine": -1.0}


@dataclass(frozen=True)
class NliResult:
    logits: np.ndarray
    probabilities: np.ndarray
    tokens: int
    premise_tokens: int
    premise_tokens_kept: int
    truncated: bool


@dataclass
class LogitCache:
    path: Path
    signature: str
    entries: dict[str, list[float]]
    hits: int = 0
    computed: int = 0


@dataclass
class NliModel:
    spec: ScorerSpec
    tokenizer: Any
    model: Any
    device: str
    info: Record
    cache: LogitCache | None = None


def label_values(labels: tuple[str, ...], values: np.ndarray) -> Record:
    return {label: float(value) for label, value in zip(labels, values, strict=True)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def load_model_config(spec: ScorerSpec) -> tuple[Any, Record]:
    config_dict, _ = PreTrainedConfig.get_config_dict(
        spec.name, revision=spec.revision, local_files_only=True
    )
    raw = dict(config_dict.get("id2label") or {})
    cast = {str(index): str(name) for index, name in raw.items()}
    if cast != raw:
        config_dict["id2label"] = cast
        config_dict["label2id"] = {name: int(index) for index, name in cast.items()}
    model_type = config_dict.pop("model_type")
    return AutoConfig.for_model(model_type, **config_dict), {
        "raw_id2label": raw,
        "id2label_values_cast_to_str": cast != raw,
        "resolved_commit": config_dict.get("_commit_hash"),
    }


def check_label_names(raw_id2label: Record, spec: ScorerSpec) -> Record:
    id2label = {int(index): str(name) for index, name in raw_id2label.items()}
    if sorted(id2label) != list(range(len(spec.labels))):
        raise SystemExit(f"{spec.key}: model labels {id2label} do not match {spec.labels}")
    names = [id2label[index] for index in range(len(spec.labels))]
    named = not any(name.isdigit() or name.upper().startswith("LABEL_") for name in names)
    if named and [name.lower() for name in names] != [label.lower() for label in spec.labels]:
        raise SystemExit(f"{spec.key}: model id2label {names} contradicts configured {spec.labels}")
    return {
        "model_config_id2label": {str(index): name for index, name in sorted(id2label.items())},
        "configured_labels": list(spec.labels),
        "verified_against_model_config": named,
    }


def load_nli_model(spec: ScorerSpec, device: str) -> NliModel:
    started = time.perf_counter()
    model_config, loaded = load_model_config(spec)
    if loaded["resolved_commit"] != spec.revision:
        raise SystemExit(f"{spec.key}: resolved {loaded['resolved_commit']} != {spec.revision}")
    weights = pinned_weights_file(spec.name, spec.revision)
    if not weights["snapshot_is_pinned_revision"]:
        raise SystemExit(
            f"{spec.key}: weights file {weights['path']} is not in the pinned snapshot"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, config=model_config, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.name,
        revision=spec.revision,
        config=model_config,
        dtype=getattr(torch, spec.dtype),
        local_files_only=True,
        use_safetensors=weights["format"] == "safetensors",
    )
    model.to(device).eval()
    info = {
        "name": spec.name,
        "revision": spec.revision,
        "config_loading": loaded,
        "weights_file": {
            **weights,
            "loading": (
                "the first of model.safetensors, model.safetensors.index.json, pytorch_model.bin, "
                "pytorch_model.bin.index.json present in the pinned snapshot, loaded with "
                "local_files_only and use_safetensors set to its format"
            ),
        },
        "labels": check_label_names(loaded["raw_id2label"], spec),
        "max_length": spec.max_length,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "tokenizer_class": type(tokenizer).__name__,
        "special_tokens_per_pair": tokenizer.num_special_tokens_to_add(pair=True),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    return NliModel(spec, tokenizer, model, device, info)


def token_counts(tokenizer: Any, texts: list[str], special: bool) -> dict[str, int]:
    distinct = list(dict.fromkeys(texts))
    if not distinct:
        return {}
    encoded = tokenizer(distinct, add_special_tokens=special, truncation=False)["input_ids"]
    return {text: len(ids) for text, ids in zip(distinct, encoded, strict=True)}


def pair_token_arrays(
    nli: NliModel, requests: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    premise_counts = token_counts(nli.tokenizer, [premise for premise, _ in requests], False)
    hypothesis_counts = token_counts(
        nli.tokenizer, [hypothesis for _, hypothesis in requests], False
    )
    premises = np.array([premise_counts[p] for p, _ in requests], dtype=np.int64)
    hypotheses = np.array([hypothesis_counts[h] for _, h in requests], dtype=np.int64)
    totals = premises + hypotheses + nli.info["special_tokens_per_pair"]
    return premises, hypotheses, totals


def cache_signature(spec: ScorerSpec, device: str) -> str:
    return f"{spec.name}@{spec.revision}@{spec.dtype}@{spec.max_length}@{device}"


def request_key(signature: str, premise: str, hypothesis: str) -> str:
    return hashlib.sha256(f"{signature}\x1e{premise}\x1e{hypothesis}".encode()).hexdigest()


def open_logit_cache(config: VerifierConfig, spec: ScorerSpec, device: str) -> LogitCache:
    path = config.cache_dir / f"nli_{spec.key}_{spec.revision[:12]}_{device}.jsonl"
    try:
        cache_lock.acquire_writer_lock(path)
    except cache_lock.CacheLockedError as error:
        raise SystemExit(str(error)) from error
    signature = cache_signature(spec, device)
    entries: dict[str, list[float]] = {}
    if path.exists():
        for entry in load_jsonl(path):
            if entry["signature"] == signature and len(entry["logits"]) == len(spec.labels):
                entries[entry["key"]] = entry["logits"]
    return LogitCache(path=path, signature=signature, entries=entries)


def append_to_cache(cache: LogitCache, keys: list[str], logits: np.ndarray) -> None:
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.path, "a") as f:
        for key, values in zip(keys, logits, strict=True):
            record = {"signature": cache.signature, "key": key, "logits": values.tolist()}
            f.write(json.dumps(record) + "\n")
            cache.entries[key] = values.tolist()


def forward_batch(nli: NliModel, batch: list[tuple[str, str]]) -> np.ndarray:
    encoded = nli.tokenizer(
        [premise for premise, _ in batch],
        [hypothesis for _, hypothesis in batch],
        truncation="only_first",
        max_length=nli.spec.max_length,
        padding=True,
        return_tensors="pt",
    ).to(nli.device)
    with torch.inference_mode():
        output = nli.model(**encoded).logits
    return output.float().cpu().numpy().astype(np.float64)


def print_progress(nli: NliModel, done: int, total: int, started: float) -> None:
    rate, remaining = throughput(done, total, started)
    print(
        f"  {nli.spec.key}: {done}/{total} pairs, {rate:.2f}/s, about {remaining:.1f} min left",
        flush=True,
    )


def forward_logits(
    nli: NliModel, requests: list[tuple[str, str]], order: np.ndarray, progress_every: int
) -> np.ndarray:
    cache = nli.cache
    if cache is None:
        raise SystemExit(f"{nli.spec.key}: no logit cache opened")
    keys = [request_key(cache.signature, premise, hypothesis) for premise, hypothesis in requests]
    missing = [int(row) for row in order if keys[row] not in cache.entries]
    started = time.perf_counter()
    for number, start in enumerate(range(0, len(missing), nli.spec.batch_size), start=1):
        rows = missing[start : start + nli.spec.batch_size]
        logits = forward_batch(nli, [requests[row] for row in rows])
        append_to_cache(cache, [keys[row] for row in rows], logits)
        if progress_every and number % progress_every == 0:
            print_progress(nli, start + len(rows), len(missing), started)
    cache.computed += len(missing)
    cache.hits += len(requests) - len(missing)
    return np.array([cache.entries[key] for key in keys], dtype=np.float64).reshape(
        len(requests), len(nli.spec.labels)
    )


def run_nli(
    nli: NliModel, requests: list[tuple[str, str]], progress_every: int = 0
) -> tuple[list[NliResult], Record]:
    premises, hypotheses, totals = pair_token_arrays(nli, requests)
    order = np.argsort(-totals, kind="stable")
    computed_before = nli.cache.computed if nli.cache is not None else 0
    started = time.perf_counter()
    logits = forward_logits(nli, requests, order, progress_every)
    seconds = time.perf_counter() - started
    computed = (nli.cache.computed if nli.cache is not None else 0) - computed_before
    probabilities = softmax(logits)
    room = np.maximum(nli.spec.max_length - hypotheses - nli.info["special_tokens_per_pair"], 0)
    kept = np.minimum(premises, room)
    results = [
        NliResult(
            logits=logits[row],
            probabilities=probabilities[row],
            tokens=int(totals[row]),
            premise_tokens=int(premises[row]),
            premise_tokens_kept=int(kept[row]),
            truncated=bool(totals[row] > nli.spec.max_length),
        )
        for row in range(len(requests))
    ]
    model_tokens = int(np.minimum(totals, nli.spec.max_length).sum())
    timing = {
        "requests": len(requests),
        "computed": computed,
        "cache_hits": len(requests) - computed,
        "seconds": round(seconds, 2),
        "computed_per_second": round(computed / seconds, 2) if seconds and computed else None,
        "model_input_tokens": model_tokens,
    }
    return results, timing


def portuguese_probes(config: VerifierConfig) -> list[tuple[str, str]]:
    return list(zip(config.probe_premises, config.probe_hypotheses, strict=True))


def run_label_probes(nli: NliModel, requests: list[tuple[str, str]]) -> Record:
    results, _ = run_nli(nli, requests)
    probes = []
    for (premise, hypothesis), expected, result in zip(
        requests, nli.spec.probe_expected, results, strict=True
    ):
        predicted = nli.spec.labels[int(result.probabilities.argmax())]
        probes.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "expected": expected,
                "predicted": predicted,
                "probabilities": label_values(nli.spec.labels, result.probabilities),
                "agrees": predicted == expected,
            }
        )
    return {
        "probes": probes,
        "agreeing": sum(1 for probe in probes if probe["agrees"]),
        "total": len(probes),
    }


def nli_requests(
    units: list[PremiseUnit], separator: str, concatenate_single: bool
) -> list[tuple[str, str]]:
    requests: dict[tuple[str, str], None] = {}
    for unit in units:
        for item in distinct_items(unit.items):
            requests[(item, unit.hypothesis)] = None
        concatenated = concatenated_premise(unit, separator, concatenate_single)
        if concatenated is not None:
            requests[(concatenated, unit.hypothesis)] = None
    return list(requests)


def per_item_scores(result: NliResult, labels: tuple[str, ...]) -> Record:
    scores = {ENTAILMENT: float(result.probabilities[labels.index(ENTAILMENT)])}
    if CONTRADICTION in labels:
        scores["not_contradiction"] = 1.0 - float(result.probabilities[labels.index(CONTRADICTION)])
    return scores


def chunk_scores(results: list[NliResult], labels: tuple[str, ...]) -> Record:
    per_item = [per_item_scores(result, labels) for result in results]
    scores: Record = {}
    for name, aggregate in CHUNK_AGGREGATES.items():
        if name not in per_item[0]:
            continue
        values = [item[name] for item in per_item]
        scores[f"{aggregate}.{name}"] = max(values) if aggregate == "max" else min(values)
    return scores


def concatenated_scores(result: NliResult, labels: tuple[str, ...]) -> Record:
    return {
        f"concatenated.{name}": value for name, value in per_item_scores(result, labels).items()
    }


def minimum_scores(names: list[str]) -> Record:
    return {name: SCORE_MINIMUM[name.split(".", 1)[1]] for name in names}


def nli_item_record(result: NliResult, labels: tuple[str, ...]) -> Record:
    return {
        "probabilities": label_values(labels, result.probabilities),
        "logits": label_values(labels, result.logits),
        "tokens": result.tokens,
        "truncated": result.truncated,
    }


def nli_unit_row(
    unit: PremiseUnit,
    results: dict[tuple[str, str], NliResult],
    spec: ScorerSpec,
    separator: str,
    concatenate_single: bool,
) -> Record:
    items = [
        nli_item_record(results[(item, unit.hypothesis)], spec.labels) if item else None
        for item in unit.items
    ]
    item_results = [results[(item, unit.hypothesis)] for item in distinct_items(unit.items)]
    text = concatenated_premise(unit, separator, concatenate_single)
    joined = results[(text, unit.hypothesis)] if text is not None else None
    concatenated = None
    if joined is not None:
        concatenated = {
            **nli_item_record(joined, spec.labels),
            "premise_tokens": joined.premise_tokens,
            "premise_tokens_kept": joined.premise_tokens_kept,
            "items_joined": sum(1 for item in unit.items if item),
        }
    if item_results:
        scores = chunk_scores(item_results, spec.labels)
        if joined is not None:
            scores |= concatenated_scores(joined, spec.labels)
    else:
        names = [n for n in spec.scores if concatenate_single or not n.startswith("concatenated.")]
        scores = minimum_scores(names)
    return {**unit_header(unit), "items": items, "concatenated": concatenated, "scores": scores}


def release_device(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def score_units_nli(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    probe_pairs: list[tuple[str, str]],
) -> tuple[list[Record], Record]:
    nli = load_nli_model(spec, device)
    nli.cache = open_logit_cache(config, spec, device)
    nli.info["logit_cache"] = {
        "path": str(nli.cache.path),
        "entries_at_start": len(nli.cache.entries),
    }
    probes = run_label_probes(nli, probe_pairs)
    if probes["agreeing"] != probes["total"]:
        print(f"WARNING {spec.key}: {probes['agreeing']}/{probes['total']} label probes agree")
    requests = nli_requests(units, config.concat_separator, concatenate_single)
    results_list, timing = run_nli(nli, requests, config.progress_every)
    results = dict(zip(requests, results_list, strict=True))
    rows = [
        nli_unit_row(unit, results, spec, config.concat_separator, concatenate_single)
        for unit in units
    ]
    details = {"model": nli.info, "label_probes": probes, "timing": timing}
    del nli
    release_device(device)
    return rows, details
