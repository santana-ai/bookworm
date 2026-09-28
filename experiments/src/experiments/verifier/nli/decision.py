"""Decision-model scorers (Laya and JEV) over a question battery."""

import gc
import os
import time
from collections import Counter
from typing import Any

import numpy as np

from experiments.udv.calibrate_threshold import describe
from experiments.verifier.decision.battery import (
    Battery,
    order_pairs_used,
)
from experiments.verifier.decision.jev import (
    DEFAULT_API_KEY_ENV,
    JEV_ENDPOINT,
    JevDecisionModel,
    JevSpec,
    ReplayDecisionModel,
    missing_key_message,
)
from experiments.verifier.decision.laya import (
    LayaDecisionModel,
    LayaSpec,
    laya_checkpoint_dir,
)
from experiments.verifier.decision.panel import (
    item_signals,
    opinion_scores,
)
from experiments.verifier.decision.questions import (
    DecisionAnswer,
    DecisionModel,
    DecisionModelError,
    DecisionQuestion,
)
from experiments.verifier.nli.benchmark import (
    PremiseUnit,
    concatenated_premise,
    distinct_items,
    unit_header,
)
from experiments.verifier.nli.config import ScorerSpec, VerifierConfig
from experiments.verifier.nli.cross_encoder import release_device

Record = dict[str, Any]

TRUNCATED_KEY = "premise"
DECISION_CHUNK = 256


def require_battery(config: VerifierConfig) -> Battery:
    if config.battery is None:
        raise SystemExit("decision-model scorers need a [decision_battery] table")
    return config.battery


def battery_questions(config: VerifierConfig, spec: ScorerSpec) -> list[DecisionQuestion]:
    battery = require_battery(config)
    return [battery.questions[key].question for key in spec.questions]


def laya_spec(spec: ScorerSpec, config: VerifierConfig, device: str) -> LayaSpec:
    return LayaSpec(
        repo_id=spec.name,
        revision=spec.revision,
        subfolder=spec.source["subfolder"],
        device=device,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        head_max_length=int(spec.source["head_max_length"]),
        truncate_key=TRUNCATED_KEY,
        cache_dir=config.decision_cache_dir,
    )


def api_key_env(spec: ScorerSpec) -> str:
    return str(spec.source.get("api_key_env", DEFAULT_API_KEY_ENV))


def jev_spec(spec: ScorerSpec, config: VerifierConfig) -> JevSpec:
    raw = spec.source
    return JevSpec(
        model_version=spec.revision,
        endpoint=raw.get("endpoint", JEV_ENDPOINT),
        api_key_env=api_key_env(spec),
        timeout_seconds=float(raw["timeout_seconds"]),
        max_retries=int(raw["max_retries"]),
        backoff_initial_seconds=float(raw["backoff_initial_seconds"]),
        backoff_max_seconds=float(raw["backoff_max_seconds"]),
        requests_per_minute=float(raw["requests_per_minute"]),
        max_concurrency=int(raw["max_concurrency"]),
        price_usd_per_million_input_tokens=float(raw["price_usd_per_million_input_tokens"]),
        cache_dir=config.decision_cache_dir,
    )


def load_decision_model(
    spec: ScorerSpec, config: VerifierConfig, device: str, mode: str
) -> DecisionModel:
    try:
        if spec.kind == "laya":
            return LayaDecisionModel(laya_spec(spec, config, device))
        if mode == "replay":
            return ReplayDecisionModel(jev_spec(spec, config))
        return JevDecisionModel(jev_spec(spec, config))
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key}: {error}") from error


def check_decision_scorer(spec: ScorerSpec, config: VerifierConfig, mode: str) -> Record:
    if spec.kind == "laya":
        try:
            directory = laya_checkpoint_dir(spec.name, spec.revision, spec.source["subfolder"])
        except DecisionModelError as error:
            raise SystemExit(f"{spec.key}: {error}") from error
        return {"checkpoint_dir": str(directory)}
    env = api_key_env(spec)
    if mode == "live" and not os.environ.get(env):
        raise SystemExit(f"{spec.key}: {missing_key_message(env)}")
    return {"mode": mode, "api_key_env": env, "api_key_set": bool(os.environ.get(env))}


def decision_state(premise: str, hypothesis: str) -> Record:
    return {"premise": premise, "hypothesis": hypothesis}


def decision_requests(
    units: list[PremiseUnit], separator: str, with_concatenated: bool, concatenate_single: bool
) -> list[tuple[str, str]]:
    requests: dict[tuple[str, str], None] = {}
    for unit in units:
        for item in distinct_items(unit.items):
            requests[(item, unit.hypothesis)] = None
        if with_concatenated:
            text = concatenated_premise(unit, separator, concatenate_single)
            if text is not None:
                requests[(text, unit.hypothesis)] = None
    return list(requests)


def run_decision(
    model: DecisionModel,
    questions: list[DecisionQuestion],
    requests: list[tuple[str, str]],
    label: str,
    progress: bool,
) -> tuple[dict[tuple[str, str], dict[str, DecisionAnswer]], float]:
    order = sorted(range(len(requests)), key=lambda row: -len(requests[row][0]))
    answers: dict[tuple[str, str], dict[str, DecisionAnswer]] = {}
    started = time.perf_counter()
    for start in range(0, len(order), DECISION_CHUNK):
        rows = [requests[row] for row in order[start : start + DECISION_CHUNK]]
        try:
            results = model.predict_batch([decision_state(p, h) for p, h in rows], questions)
        except DecisionModelError as error:
            raise SystemExit(f"{label}: {error}") from error
        answers.update(zip(rows, results, strict=True))
        if progress:
            done = start + len(rows)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            left = (len(order) - done) / rate / 60 if rate else float("nan")
            print(
                f"  {label}: {done}/{len(order)} premise texts x {len(questions)} questions, "
                f"{rate:.2f} texts/s, about {left:.1f} min left",
                flush=True,
            )
    return answers, time.perf_counter() - started


def run_decision_probes(
    model: DecisionModel,
    questions: list[DecisionQuestion],
    spec: ScorerSpec,
    pairs: list[tuple[str, str]],
) -> Record:
    try:
        results = model.predict_batch([decision_state(p, h) for p, h in pairs], questions)
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key} label probes: {error}") from error
    question = spec.questions[0]
    probes = []
    for (premise, hypothesis), expected, answers in zip(
        pairs, spec.probe_expected, results, strict=True
    ):
        predicted = answers[question].choice
        probes.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "answers": {key: answer.record() for key, answer in answers.items()},
                "agrees": predicted == expected,
            }
        )
    return {
        "probes": probes,
        "agreeing": sum(1 for probe in probes if probe["agrees"]),
        "total": len(probes),
    }


def decision_item_record(answers: dict[str, DecisionAnswer], signals: Record) -> Record:
    values = list(answers.values())
    truncated = [key for key, answer in answers.items() if answer.truncated]
    record: Record = {
        "answers": {key: answer.record() for key, answer in answers.items()},
        "signals": signals,
        "tokens": max(answer.input_tokens or 0 for answer in values),
        "state_tokens": values[0].state_tokens,
        "truncated": bool(truncated),
        "truncated_questions": truncated,
        "overflow": any(answer.overflow for answer in values),
    }
    if truncated:
        record["premise_chars"] = answers[truncated[0]].premise_chars
        record["premise_chars_kept"] = min(
            int(answers[key].premise_chars_kept or 0) for key in truncated
        )
    return record


def decision_unit_row(
    unit: PremiseUnit,
    answers: dict[tuple[str, str], dict[str, DecisionAnswer]],
    spec: ScorerSpec,
    battery: Battery,
    separator: str,
    concatenate_single: bool,
) -> Record:
    signals: dict[str, Record] = {}
    items: list[Record | None] = []
    for item in unit.items:
        if not item:
            items.append(None)
            continue
        answered = answers[(item, unit.hypothesis)]
        values = signals.setdefault(item, item_signals(battery, spec.questions, answered))
        items.append(decision_item_record(answered, values))
    item_values = [signals[item] for item in distinct_items(unit.items)]
    text = concatenated_premise(unit, separator, concatenate_single) if spec.concatenated else None
    concatenated = None
    joined_values = None
    if text is not None:
        joined = answers[(text, unit.hypothesis)]
        joined_values = item_signals(battery, spec.questions, joined)
        concatenated = {
            **decision_item_record(joined, joined_values),
            "items_joined": sum(1 for item in unit.items if item),
        }
    scores = opinion_scores(
        battery,
        spec.questions,
        item_values,
        joined_values,
        spec.concatenated and concatenate_single,
    )
    return {**unit_header(unit), "items": items, "concatenated": concatenated, "scores": scores}


def counter_delta(after: Record, before: Record) -> Record:
    return {
        key: round(value - before.get(key, 0), 4)
        if isinstance(value, float)
        else value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and key != "max_rounding_gap"
    }


def decision_timing(
    spec: ScorerSpec, before: Record, after: Record, texts: int, seconds: float
) -> Record:
    counters = after.get("counters", {})
    delta = counter_delta(counters, before.get("counters", {}))
    timing: Record = {
        "premise_texts": texts,
        "questions": len(spec.questions),
        "wall_seconds": round(seconds, 2),
        **delta,
    }
    if spec.kind == "laya" and "computed_input_tokens" in delta:
        tokens, forward = delta["computed_input_tokens"], delta["forward_seconds"]
        timing["seconds_per_input_token"] = forward / tokens if tokens else None
        timing["computed_per_second"] = round(delta["computed"] / forward, 2) if forward else None
        timing["max_rounding_gap"] = counters.get("max_rounding_gap")
    return timing


def question_record(battery: Battery, key: str) -> Record:
    spec = battery.questions[key]
    return {
        "payload": spec.question.payload(),
        "support_option": spec.support_option,
        "against_option": spec.against_option,
        "against_signal": spec.against_signal,
        "reverses": spec.reverses,
    }


def score_units_decision(
    units: list[PremiseUnit],
    spec: ScorerSpec,
    config: VerifierConfig,
    device: str,
    concatenate_single: bool,
    probe_pairs: list[tuple[str, str]],
    mode: str,
) -> tuple[list[Record], Record]:
    battery = require_battery(config)
    questions = battery_questions(config, spec)
    model = load_decision_model(spec, config, device, mode)
    probes = run_decision_probes(model, questions, spec, probe_pairs)
    if probes["agreeing"] != probes["total"]:
        print(f"WARNING {spec.key}: {probes['agreeing']}/{probes['total']} label probes agree")
    requests = decision_requests(
        units, config.concat_separator, spec.concatenated, concatenate_single
    )
    before = model.describe()
    answers, seconds = run_decision(model, questions, requests, spec.key, True)
    after = model.describe()
    rows = [
        decision_unit_row(unit, answers, spec, battery, config.concat_separator, concatenate_single)
        for unit in units
    ]
    details = {
        "model": after,
        "label_probes": probes,
        "questions": {key: question_record(battery, key) for key in spec.questions},
        "order_pairs_used": order_pairs_used(battery, spec.questions),
        "timing": decision_timing(spec, before, after, len(requests), seconds),
    }
    del model
    gc.collect()
    release_device(device)
    return rows, details


def decision_truncation_summary(rows: list[Record], questions: tuple[str, ...]) -> Record:
    items = [item for row in rows for item in row["items"] if item is not None]
    joined = [row["concatenated"] for row in rows if row["concatenated"] is not None]

    def by_question(entries: list[Record]) -> Record:
        counts = Counter(key for entry in entries for key in entry["truncated_questions"])
        return {key: counts.get(key, 0) for key in questions}

    summary: Record = {
        "items_scored": len(items),
        "items_truncated": sum(1 for item in items if item["truncated"]),
        "items_truncated_by_question": by_question(items),
        "concatenated_premises": len(joined),
        "concatenated_truncated": sum(1 for entry in joined if entry["truncated"]),
        "concatenated_truncated_by_question": by_question(joined),
        "overflow": sum(1 for entry in [*items, *joined] if entry["overflow"]),
    }
    for name, entries in (("item_state_tokens", items), ("concatenated_state_tokens", joined)):
        tokens = [entry["state_tokens"] for entry in entries if entry["state_tokens"] is not None]
        if tokens:
            summary[name] = describe(np.array(tokens, dtype=np.float64))
    kept = [
        entry["premise_chars_kept"] / entry["premise_chars"]
        for entry in [*items, *joined]
        if entry["truncated"] and entry.get("premise_chars")
    ]
    if kept:
        summary["kept_premise_char_fraction_when_truncated"] = describe(
            np.array(kept, dtype=np.float64)
        )
    return summary
