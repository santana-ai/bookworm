"""Token and time estimates of a score run before any model runs."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
from bookworm import write_json
from transformers import AutoTokenizer

from experiments.common.reporting import utc_timestamp
from experiments.verifier.benchmark_inputs import split_records
from experiments.verifier.decision.laya import LayaTokenizer
from experiments.verifier.decision.questions import DecisionModelError
from experiments.verifier.nli.benchmark import PremiseUnit, concatenated_premise, distinct_items
from experiments.verifier.nli.config import (
    RunDeclaration,
    ScorerSpec,
    VerifierConfig,
    scored_splits,
    selected_scorers,
)
from experiments.verifier.nli.cosine import item_sentences
from experiments.verifier.nli.cross_encoder import (
    NliModel,
    load_model_config,
    nli_requests,
    pair_token_arrays,
    portuguese_probes,
)
from experiments.verifier.nli.decision import battery_questions, decision_requests, decision_state
from experiments.verifier.nli.provenance import code_hashes, environment
from experiments.verifier.nli.scoring import (
    prepare_benchmark_units,
    run_directory,
    score_report_file,
)
from experiments.verifier.nli.translated import (
    Translations,
    english_units,
    open_translations,
    translation_summary,
    translation_texts,
)

Record = dict[str, Any]


def cosine_plan(units: list[PremiseUnit]) -> Record:
    items = {item for unit in units for item in distinct_items(unit.items)}
    sentences = {s for item in items for s in item_sentences(item)[0]}
    return {"distinct_items": len(items), "distinct_sentences": len(sentences)}


def nli_plan(spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig) -> Record:
    model_config, _ = load_model_config(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, config=model_config, local_files_only=True
    )
    nli = NliModel(
        spec,
        tokenizer,
        None,
        "cpu",
        {"special_tokens_per_pair": tokenizer.num_special_tokens_to_add(pair=True)},
    )
    requests = nli_requests(units, config.concat_separator, True)
    _, _, totals = pair_token_arrays(nli, requests)
    concatenated = {
        (concatenated_premise(unit, config.concat_separator, True), unit.hypothesis)
        for unit in units
    }
    truncated = [
        total > spec.max_length
        for request, total in zip(requests, totals, strict=True)
        if request in concatenated
    ]
    return {
        "requests": len(requests),
        "model_input_tokens": int(np.minimum(totals, spec.max_length).sum()),
        "concatenated_requests": len(truncated),
        "concatenated_truncated": int(sum(truncated)),
        "item_requests_truncated": int(
            sum(
                total > spec.max_length
                for request, total in zip(requests, totals, strict=True)
                if request not in concatenated
            )
        ),
    }


def request_modes(
    units: list[PremiseUnit], separator: str, with_concatenated: bool
) -> dict[str, list[tuple[str, str]]]:
    chunks = decision_requests(units, separator, False, True)
    joined = [
        request
        for request in decision_requests(units, separator, with_concatenated, True)
        if request not in set(chunks)
    ]
    return {"chunk": chunks, "concatenated": joined}


def laya_plan(spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig) -> Record:
    try:
        tokenizer = LayaTokenizer(spec.name, spec.revision, spec.source["subfolder"])
    except DecisionModelError as error:
        raise SystemExit(f"{spec.key}: {error}") from error
    questions = battery_questions(config, spec)
    modes = request_modes(units, config.concat_separator, spec.concatenated)
    by_question: Record = {}
    for question in questions:
        room = tokenizer.room(question)
        prefix = tokenizer.max_length - room
        by_question[question.key] = {"state_room": room}
        for mode, requests in modes.items():
            tokens = np.array(
                [
                    tokenizer.count(json.dumps(decision_state(p, h), ensure_ascii=False))
                    for p, h in requests
                ],
                dtype=np.int64,
            )
            by_question[question.key][mode] = {
                "requests": len(requests),
                "model_input_tokens": int((prefix + np.minimum(tokens, room)).sum()),
                "truncated": int((tokens > room).sum()),
            }
    return {
        "premise_texts": {mode: len(requests) for mode, requests in modes.items()},
        "questions": len(questions),
        "requests": sum(len(r) for r in modes.values()) * len(questions),
        "model_input_tokens": sum(
            entry[mode]["model_input_tokens"] for entry in by_question.values() for mode in modes
        ),
        "max_len": tokenizer.max_length,
        "head_max_len": tokenizer.head_max_length,
        "by_question": by_question,
    }


def jev_plan(
    spec: ScorerSpec, units: list[PremiseUnit], config: VerifierConfig, proxy: LayaTokenizer
) -> Record:
    questions = battery_questions(config, spec)
    modes = request_modes(units, config.concat_separator, spec.concatenated)
    question_tokens = sum(proxy.count(question.payload_json()) for question in questions)
    state_tokens = sum(
        proxy.count(json.dumps(decision_state(p, h), ensure_ascii=False))
        for requests in modes.values()
        for p, h in requests
    )
    requests = sum(len(r) for r in modes.values())
    tokens = state_tokens + requests * question_tokens
    rate = float(spec.source["requests_per_minute"])
    return {
        "requests": requests,
        "premise_texts": {mode: len(r) for mode, r in modes.items()},
        "questions_per_request": len(questions),
        "proxy_input_tokens": tokens,
        "proxy_rule": (
            "state and question JSON counted with the laya multilingual tokenizer (Gemma "
            "vocabulary); Jev's own tokenizer is not public, so the cost is an estimate"
        ),
        "proxy_cost_usd": round(
            tokens * float(spec.source["price_usd_per_million_input_tokens"]) / 1e6, 3
        ),
        "minutes_at_rate_limit": round(requests / rate, 1) if rate else None,
    }


def proxy_tokenizer(config: VerifierConfig) -> LayaTokenizer:
    for spec in config.scorers.values():
        if spec.kind == "laya" and spec.source["subfolder"] == "multilingual":
            try:
                return LayaTokenizer(spec.name, spec.revision, "multilingual")
            except DecisionModelError as error:
                raise SystemExit(f"{spec.key}: {error}") from error
    raise SystemExit("the Jev token proxy needs a laya scorer with subfolder multilingual")


def plan_units(
    spec: ScorerSpec,
    units: list[PremiseUnit],
    config: VerifierConfig,
    translations: Translations | None,
) -> tuple[list[PremiseUnit], Record]:
    if spec.language != "en":
        return units, {"texts": "portuguese"}
    if translations is None:
        return units, {
            "texts": "portuguese_proxy",
            "translation_model": spec.translation_model,
            "reason": "no translation store",
        }
    store = translations.store(spec)
    texts = translation_texts(units, portuguese_probes(config), store)
    missing = store.missing(texts)
    source = {"translation_model": spec.translation_model, "store": str(store.path)}
    if missing:
        return units, {
            "texts": "portuguese_proxy",
            **source,
            "missing_translations": len(missing),
            "distinct_texts": len(texts),
            "rule": "the English texts of this translation model are not all translated yet, so "
            "the Portuguese texts are counted with this scorer's tokenizer; the counts are a proxy",
        }
    return english_units(units, store), {
        "texts": "english",
        **source,
        "distinct_texts": len(texts),
    }


def smoke_rate(report: Record) -> Record:
    timing = report["timing"]
    device = report["model"].get("device")
    if report["kind"] == "laya":
        rate = timing.get("seconds_per_input_token")
        basis = "forward_seconds / computed_input_tokens"
    elif timing.get("computed") and timing.get("model_input_tokens"):
        share = timing["computed"] / timing["requests"]
        rate = timing["seconds"] / (timing["model_input_tokens"] * share)
        basis = "seconds / (model_input_tokens x computed share)"
    else:
        rate, basis = None, "no computed request in the smoke"
    return {"device": device, "seconds_per_input_token": rate, "basis": basis}


def scorer_tokens(plan: Record, dropped: set[tuple[str, str]]) -> int:
    if "by_question" not in plan:
        return int(plan["model_input_tokens"])
    return sum(
        entry[mode]["model_input_tokens"]
        for key, entry in plan["by_question"].items()
        for mode in ("chunk", "concatenated")
        if (key, mode) not in dropped
    )


def check_estimate_scorers(planned: list[str], declaration: RunDeclaration) -> None:
    local = list(declaration.source["compute"]["local_scorers"])
    missing = [key for key in local if key not in planned]
    if missing:
        raise SystemExit(
            f"--timing-from estimates every local scorer of declarations.{declaration.name}; "
            f"the plan lacks {missing}: run plan without --scorers, or with all of {local}"
        )


def compute_estimate(plans: Record, timing_dir: Path, declaration: RunDeclaration) -> Record:
    check_estimate_scorers(list(plans), declaration)
    compute = declaration.source["compute"]
    budget = float(compute["budget_hours"])
    partners = {**declaration.twins, **{first: twin for twin, first in declaration.twins.items()}}
    rates: Record = {}
    for key in compute["local_scorers"]:
        path = score_report_file(timing_dir, key)
        if not path.exists():
            rates[key] = {"report": str(path), "seconds_per_input_token": None}
            continue
        with open(path) as f:
            report = json.load(f)
        rates[key] = {"report": str(path), **smoke_rate(report)}
    known = all(rates[key]["seconds_per_input_token"] for key in rates)
    devices = sorted({str(rates[key].get("device")) for key in rates})
    dropped: dict[str, set[tuple[str, str]]] = {key: set() for key in rates}

    def hours() -> dict[str, float | None]:
        return {
            key: (
                rates[key]["seconds_per_input_token"]
                * scorer_tokens(plans[key], dropped[key])
                / 3600
                if rates[key]["seconds_per_input_token"]
                else None
            )
            for key in rates
        }

    first = hours()
    steps = []
    total = sum(v for v in first.values() if v is not None) if known else None
    for name in compute["drop_order"]:
        if total is None or total <= budget:
            break
        step = compute["drop_steps"][name]
        named = list(step["scorers"])
        affected = list(dict.fromkeys([*named, *(partners[k] for k in named if k in partners)]))
        for key in affected:
            dropped[key] |= {(q, mode) for q in step["questions"] for mode in step["modes"]}
        after = hours()
        total = sum(v for v in after.values() if v is not None)
        steps.append(
            {
                "step": name,
                "text": step["text"],
                "scorers": affected,
                "total_hours_after": round(total, 2),
            }
        )
    last = hours()
    return {
        "budget_hours": budget,
        "timing_from": str(timing_dir),
        "devices": devices,
        "rates": rates,
        "hours_full": {k: None if v is None else round(v, 2) for k, v in first.items()},
        "total_hours_full": None
        if not known
        else round(sum(cast(Iterable[float], first.values())), 2),
        "steps_applied": steps,
        "hours_after_steps": {k: None if v is None else round(v, 2) for k, v in last.items()},
        "total_hours_after_steps": None if total is None else round(total, 2),
        "fits": None if total is None else total <= budget,
        "rule": compute["drop_rule"],
        "twin_rule": compute.get("twin_rule"),
    }


def command_plan(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = scored_splits(config, args.final_test)
    units, context = prepare_benchmark_units(config, splits, args.limit_per_split, args.subset_rule)
    specs = selected_scorers(config, args.scorers)
    declaration = config.declarations.get(args.run_name)
    if args.timing_from is not None:
        if declaration is None or "compute" not in declaration.source:
            raise SystemExit(f"--timing-from needs a compute table in declarations.{args.run_name}")
        check_estimate_scorers([spec.key for spec in specs], declaration)
    translations = open_translations(config, specs, args.translation_cache_dir)
    proxy = None
    plans: Record = {}
    for spec in specs:
        spec_units, texts = plan_units(spec, units, config, translations)
        if spec.kind == "cosine":
            plans[spec.key] = cosine_plan(spec_units)
        elif spec.kind == "nli":
            plans[spec.key] = nli_plan(spec, spec_units, config)
        elif spec.kind == "laya":
            plans[spec.key] = laya_plan(spec, spec_units, config)
        else:
            if proxy is None:
                proxy = proxy_tokenizer(config)
            plans[spec.key] = jev_plan(spec, spec_units, config, proxy)
        if spec.language == "en":
            plans[spec.key]["texts"] = texts
        print(f"planned {spec.key}", flush=True)
    estimate = None
    if args.timing_from is not None and declaration is not None:
        estimate = compute_estimate(plans, args.timing_from, declaration)
    report = {
        "created_at": utc_timestamp(),
        "splits": split_records(splits, args.final_test),
        "opinions": len(units),
        "plans": plans,
        "compute_estimate": estimate,
        "translation": (
            None
            if translations is None
            else {
                spec.key: translation_summary(config, translations, spec)
                for spec in specs
                if spec.language == "en"
            }
        ),
        **context,
        "code": code_hashes(),
        "environment": environment(),
        "config": config.source,
    }
    path = run_directory(config, args) / "plan.json"
    write_json(report, path)
    print(json.dumps({"plans": plans, "compute_estimate": estimate}, indent=2))
