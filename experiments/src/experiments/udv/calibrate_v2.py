import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl, sha256_of_file, write_json, write_jsonl
from sentence_transformers import SentenceTransformer

from experiments.common.provenance import source_hashes
from experiments.common.transcript import normalize_whitespace
from experiments.common.udv_run import (
    UdvConfig,
    load_config,
    load_encoder,
    load_lds_records,
    resolve_hearing_people,
    seed_everything,
    select_device,
)
from experiments.retrieval.data import SentenceSpan, locate_turn_sentences, window_units
from experiments.udv import calibrate_threshold as base

Record = dict[str, Any]

UNIT_SIZES = {"sentence": 1, "window2": 2, "window3": 3}
PAIR_ROLES = ("positive", "legacy_random", "hard_negative")
VERIFIER_RULES = ("legacy_random", "hard_negative")
ROLE_TIER = {role: f"calibration_{role}" for role in PAIR_ROLES}
LABEL_SEMANTICS_UNIT = (
    "candidate units are windows of consecutive candidate sentences of one matched turn "
    "(retrieval_data.window_units, stride 1); a unit is a target when it holds a sentence that "
    "agrees with the silver target sentence; the embedded text is the unit text (sentences joined "
    "by a space) and the premise text is the transcript span of the unit, whitespace normalized"
)


@dataclass(frozen=True)
class WindowUnits:
    texts: list[str]
    spans: list[str]
    positions: list[tuple[int, ...]]


def unit_size(udv_config: UdvConfig) -> tuple[str, int]:
    unit = udv_config.source["calibration"]["unit"]
    if unit not in UNIT_SIZES:
        raise SystemExit(f"calibration.unit must be one of {sorted(UNIT_SIZES)}")
    if udv_config.source["evidence"].get("semantic_unit", "sentence") != unit:
        raise SystemExit("calibration.unit differs from evidence.semantic_unit")
    return unit, UNIT_SIZES[unit]


def person_windows(person: Record, transcript: str, size: int, checks: Counter[str]) -> WindowUnits:
    texts: list[str] = []
    spans: list[str] = []
    positions: list[tuple[int, ...]] = []
    offset = 0
    for turn in person["matched_turns"]:
        turn_spans: list[SentenceSpan] = locate_turn_sentences(turn, transcript, checks)
        if not turn_spans:
            continue
        for unit in window_units(turn_spans, offset, size):
            texts.append(unit.text)
            spans.append(normalize_whitespace(transcript[unit.start_char : unit.end_char]))
            positions.append(unit.sentence_positions)
        offset += len(turn_spans)
    if offset != len(person["sentences"]):
        raise SystemExit(f"person {person['index']}: located sentences differ from the pipeline")
    return WindowUnits(texts, spans, positions)


def window_query(query: Record, windows: WindowUnits, window_offset: int) -> Record | None:
    targets = set(query["target_indices"])
    target_windows = [
        index for index, members in enumerate(windows.positions) if targets.intersection(members)
    ]
    if not target_windows:
        return None
    return {
        **query,
        "sentence_target_indices": query["target_indices"],
        "target_indices": target_windows,
        "candidates": windows.texts,
        "candidate_spans": windows.spans,
        "sentence_offset": window_offset,
    }


def collect_hearing(
    hearing: Record,
    rows: list[Record],
    encoder: SentenceTransformer,
    udv_config: UdvConfig,
    device: str,
    ledger: base.EncodingLedger,
    unit: str,
    size: int,
    checks: Counter[str],
) -> Record:
    hearing_id = hearing["id"]
    transcript = hearing["transcricao"]
    people = resolve_hearing_people(hearing)
    windows = {
        person["index"]: person_windows(person, transcript, size, checks) for person in people
    }
    offsets: dict[int, slice] = {}
    cursor = 0
    for person in people:
        count = len(windows[person["index"]].texts)
        offsets[person["index"]] = slice(cursor, cursor + count)
        cursor += count
    all_texts = [text for person in people for text in windows[person["index"]].texts]
    all_spans = [span for person in people for span in windows[person["index"]].spans]
    unit_embeddings = base.encode_guarded(
        encoder, all_texts, udv_config, device, f"{unit}s", hearing_id, ledger
    )
    base_queries, funnel = base.trusted_quote_queries(hearing, people)
    funnel["target_not_in_windows"] = 0
    queries: list[Record] = []
    for query in base_queries:
        converted = window_query(
            query, windows[query["person_index"]], offsets[query["person_index"]].start
        )
        if converted is None:
            funnel["target_not_in_windows"] += 1
            continue
        queries.append(converted)
    unmasked: list[Record] = []
    query_embeddings = np.empty((0, unit_embeddings.shape[1]), dtype=np.float32)
    if queries:
        opinion_texts = [text for person in people for text in person["participant"]["opinioes"]]
        opinion_embeddings = base.encode_guarded(
            encoder, opinion_texts, udv_config, device, "opinions", hearing_id, ledger
        )
        query_embeddings = opinion_embeddings[[q["opinion_position"] for q in queries]]
        unmasked = [
            base.score_query(
                query,
                "opinion",
                opinion_embeddings[query["opinion_position"]],
                unit_embeddings[offsets[query["person_index"]]],
            )
            for query in queries
        ]
    sentence_queries = {query["id"]: query for query in base_queries}
    by_id = {query["id"]: query for query in queries}
    problems = {row["id"]: base.masked_row_problems(row, sentence_queries) for row in rows}
    problems = {row_id: found for row_id, found in problems.items() if found}
    masked: list[Record] = []
    kept_rows = [row for row in rows if row["id"] in by_id]
    if kept_rows and not problems:
        masked_embeddings = base.encode_guarded(
            encoder,
            [row["masked_opinion"] for row in rows],
            udv_config,
            device,
            "masked",
            hearing_id,
            ledger,
        )
        masked = [
            base.score_query(
                {**by_id[row["id"]], "masked_opinion": row["masked_opinion"]},
                "masked_opinion",
                masked_embeddings[position],
                unit_embeddings[offsets[row["person"]["index"]]],
            )
            for position, row in enumerate(rows)
            if row["id"] in by_id
        ]
    return {
        "unmasked": unmasked,
        "query_embeddings": query_embeddings,
        "masked": masked,
        "unit_embeddings": unit_embeddings,
        "unit_spans": all_spans,
        "funnel": funnel,
        "problems": problems,
    }


def verifier_pairs(
    unmasked: list[Record],
    draws: dict[str, dict[str, list[Record]]],
    spans: dict[int, list[str]],
) -> list[Record]:
    rows: list[Record] = []
    for query in unmasked:
        best = max(query["target_indices"], key=lambda index: query["candidate_scores"][index])
        members: list[tuple[str, int, int, float]] = [
            (
                "positive",
                query["hearing_id"],
                query["sentence_offset"] + best,
                float(query["candidate_scores"][best]),
            )
        ]
        for rule in VERIFIER_RULES:
            for negative in draws[rule].get(query["id"], []):
                members.append(
                    (rule, negative["hearing_id"], negative["sentence_index"], negative["score"])
                )
        for role, unit_hearing, unit_index, score in members:
            rows.append(
                {
                    "id": f"{query['id']}:{role}",
                    "query_id": query["id"],
                    "hearing_id": query["hearing_id"],
                    "pair_role": role,
                    "unit_hearing_id": unit_hearing,
                    "unit_index": unit_index,
                    "tier": ROLE_TIER[role],
                    "proposition": query["opinion"],
                    "evidence": {
                        "text": spans[unit_hearing][unit_index],
                        "support_type": "semantic_similarity",
                        "score": round(score, base.QUERY_SCORE_DECIMALS),
                    },
                }
            )
    return rows


def command_cosine(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    udv_config = load_config(args.config)
    config = base.load_calibration_config(udv_config)
    unit, size = unit_size(udv_config)
    seed_everything(udv_config.seed)
    lds = load_lds_records(udv_config)
    split_of, split_source = base.load_split_lookup(
        config.manifest_path, udv_config.expected_sha256
    )
    hearings = base.select_calibration_hearings(lds, split_of, config.splits)
    ledger = base.EncodingLedger(allowed_hearing_ids=frozenset(h["id"] for h in hearings))
    masked_rows, masked_counts = base.load_masked_rows(
        config.masked_benchmark_path, split_of, config.splits
    )
    device = select_device(udv_config.device)
    print(f"{len(hearings)} calibration hearings | unit {unit} | {device}", flush=True)
    encoder = load_encoder(udv_config, device)
    unmasked: list[Record] = []
    masked: list[Record] = []
    query_embeddings: list[np.ndarray] = []
    unit_embeddings: dict[int, np.ndarray] = {}
    unit_spans: dict[int, list[str]] = {}
    funnels: list[Record] = []
    problems: dict[str, list[str]] = {}
    checks: Counter[str] = Counter()
    for number, hearing in enumerate(hearings, start=1):
        collected = collect_hearing(
            hearing,
            masked_rows.get(hearing["id"], []),
            encoder,
            udv_config,
            device,
            ledger,
            unit,
            size,
            checks,
        )
        unmasked.extend(collected["unmasked"])
        masked.extend(collected["masked"])
        query_embeddings.append(collected["query_embeddings"])
        unit_embeddings[hearing["id"]] = collected["unit_embeddings"]
        unit_spans[hearing["id"]] = collected["unit_spans"]
        funnels.append(collected["funnel"])
        problems.update(collected["problems"])
        print(
            f"[{number}/{len(hearings)}] hearing {hearing['id']}: "
            f"{len(collected['unmasked'])} quote queries, {len(collected['masked'])} masked",
            flush=True,
        )
    if problems:
        raise SystemExit(f"the masked benchmark does not match the pipeline: {problems}")
    pool = base.build_pool(np.vstack(query_embeddings), unit_embeddings)
    results, draws = base.run_rules(unmasked, masked, pool, config)
    query_hearings = {
        "legacy_random": {query["hearing_id"] for query in unmasked},
        "hard_negative": {
            query["hearing_id"] for query in unmasked if len(query["non_target_indices"])
        },
        "masked_hard_negative": {
            query["hearing_id"] for query in masked if len(query["non_target_indices"])
        },
        base.PRIMARY_RULE: {query["hearing_id"] for query in masked},
    }
    leak = base.leak_check(split_of, config, ledger, query_hearings, pool)
    points = base.operating_points(results, masked, udv_config.embedding_threshold)
    rows_path = config.output_dir / f"{config.version}_queries.jsonl"
    rows = base.query_rows(unmasked, draws) + base.query_rows(masked, draws)
    write_jsonl(rows, rows_path)
    pairs = verifier_pairs(unmasked, draws, unit_spans)
    pairs_path = config.output_dir / f"{config.version}_verifier_pairs.jsonl"
    write_jsonl(pairs, pairs_path)
    adopted = udv_config.source["calibration"]["adopted_rule"]
    report = base.build_report(
        results,
        points,
        leak,
        ledger,
        {
            **base.sum_counts(funnels),
            "used_by_legacy_random": len(unmasked),
            "used_by_hard_negative": results["hard_negative"]["pairs"]["queries"],
        },
        {**masked_counts, "used": len(masked)},
        {
            "lds": {"path": str(udv_config.lds_path), "sha256": udv_config.expected_sha256},
            "splits": split_source,
            "masked_benchmark": base.benchmark_source(config.masked_benchmark_path),
        },
        {
            "queries": {
                "path": str(rows_path),
                "rows": len(rows),
                "sha256": sha256_of_file(rows_path),
            },
            "verifier_pairs": {
                "path": str(pairs_path),
                "rows": len(pairs),
                "by_role": dict(Counter(pair["pair_role"] for pair in pairs)),
                "sha256": sha256_of_file(pairs_path),
            },
        },
        encoder,
        device,
        time.perf_counter() - started,
        udv_config,
        config,
    )
    report["unit"] = {
        "name": unit,
        "size": size,
        "semantics": LABEL_SEMANTICS_UNIT,
        "sentence_location_checks": dict(checks),
    }
    report["adopted_rule"] = adopted
    report["adopted_rule_declaration"] = udv_config.source["calibration"][
        "adopted_rule_declaration"
    ]
    report["adopted_threshold"] = results[adopted]["threshold"]
    report["adopted_threshold_rounded"] = results[adopted]["threshold_rounded"]
    report["code"].update(source_hashes(Path(__file__)))
    write_json(report, config.output_dir / f"{config.version}.json")
    base.print_summary(report)
    print(f"adopted ({adopted}): {report['adopted_threshold']}", flush=True)


def load_probabilities(path: Path) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for row in load_jsonl(path):
        if not row["scored"]:
            raise SystemExit(f"{row['udv_id']}: calibration pair without a verifier score")
        probabilities[row["udv_id"]] = float(row["primary_probability"])
    return probabilities


def verifier_rule(
    queries: list[str],
    hearings: dict[str, int],
    positive: dict[str, float],
    negative: dict[str, float],
    config: base.CalibrationConfig,
    stream: int,
) -> Record:
    kept = [query for query in queries if query in negative]
    positives = np.array([positive[query] for query in kept], dtype=np.float64)
    negatives = np.array([negative[query] for query in kept], dtype=np.float64)
    point = base.pair_threshold(positives, negatives, config)
    groups = base.unit_groups([hearings[query] for query in kept], config.bootstrap_unit)
    rng = base.rule_rng(config.seed, 100 + stream)
    replicates = []
    for _ in range(config.bootstrap_samples):
        rows = base.resample_rows(rng, groups)
        replicates.append(base.pair_threshold(positives[rows], negatives[rows], config))
    threshold = point["threshold"]
    return {
        "pairs": {
            "queries": len(kept),
            "hearings": len({hearings[query] for query in kept}),
            "positives": len(positives),
            "negatives": len(negatives),
        },
        "threshold": base.rounded(threshold),
        "threshold_exact": threshold,
        "threshold_rounded": round(threshold, 4),
        "positive_quantile_value": base.rounded(point["positive_quantile_value"]),
        "negative_quantile_value": base.rounded(point["negative_quantile_value"]),
        "positives_below_threshold": int((positives < threshold).sum()),
        "negatives_at_or_above_threshold": int((negatives >= threshold).sum()),
        "positive_scores": base.describe(positives),
        "negative_scores": base.describe(negatives),
        "bootstrap": {
            key: base.interval(
                [replicate[key] for replicate in replicates], config.confidence_level
            )
            for key in ("threshold", "positive_quantile_value", "negative_quantile_value")
        },
    }


def at_threshold(values: list[float], threshold: float) -> Record:
    array = np.array(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "at_or_above": int((array >= threshold).sum()),
        "share": base.rounded(float((array >= threshold).mean())) if len(array) else None,
    }


def command_verifier(args: argparse.Namespace) -> None:
    udv_config = load_config(args.config)
    config = base.load_calibration_config(udv_config)
    raw = udv_config.source["verifier_calibration"]
    cosine_path = config.output_dir / f"{config.version}.json"
    with open(cosine_path) as f:
        cosine = json.load(f)
    if not cosine["leak_check"]["passed"]:
        raise SystemExit(f"{cosine_path}: the leak check did not pass")
    pairs_path = Path(cosine["artifacts"]["verifier_pairs"]["path"])
    if sha256_of_file(pairs_path) != cosine["artifacts"]["verifier_pairs"]["sha256"]:
        raise SystemExit(f"{pairs_path}: sha256 differs from the calibration report")
    scores_path = Path(raw["scores_path"])
    probabilities = load_probabilities(scores_path)
    pairs = load_jsonl(pairs_path)
    split_of, split_source = base.load_split_lookup(
        config.manifest_path, udv_config.expected_sha256
    )
    used = {pair["hearing_id"] for pair in pairs} | {pair["unit_hearing_id"] for pair in pairs}
    outside = sorted(h for h in used if split_of[h] not in config.splits)
    if outside:
        raise SystemExit(f"calibration pairs outside the calibration splits: {outside}")
    by_role: dict[str, dict[str, float]] = {role: {} for role in PAIR_ROLES}
    hearings: dict[str, int] = {}
    for pair in pairs:
        by_role[pair["pair_role"]][pair["query_id"]] = probabilities[pair["id"]]
        hearings[pair["query_id"]] = pair["hearing_id"]
    queries = sorted(by_role["positive"])
    rules = {
        rule: verifier_rule(queries, hearings, by_role["positive"], by_role[rule], config, stream)
        for stream, rule in enumerate(VERIFIER_RULES)
    }
    train_threshold = float(raw["train_threshold"])
    adopted = raw["adopted_rule"]
    report = {
        "calibration_version": raw["version"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "score": raw["score"],
        "declaration": raw["declaration"],
        "adopted_rule": adopted,
        "udv_threshold": rules[adopted]["threshold"],
        "udv_threshold_exact": rules[adopted]["threshold_exact"],
        "train_threshold": train_threshold,
        "rules": rules,
        "pairs_at_train_threshold": {
            role: at_threshold(list(by_role[role].values()), train_threshold) for role in PAIR_ROLES
        },
        "bootstrap": {
            "samples": config.bootstrap_samples,
            "unit": config.bootstrap_unit,
            "confidence_level": config.confidence_level,
            "method": "percentile over hearing resamples of the fixed pairs drawn by the cosine "
            "calibration; negatives are not redrawn, because every negative needs a translation "
            "and a verifier score",
            "seed": config.seed,
        },
        "leak_check": {
            "calibration_splits": list(config.splits),
            "hearings_used": len(used),
            "hearings_outside_calibration_splits": outside,
            "passed": not outside,
        },
        "sources": {
            "cosine_calibration": {"path": str(cosine_path), "sha256": sha256_of_file(cosine_path)},
            "pairs": {"path": str(pairs_path), "sha256": sha256_of_file(pairs_path)},
            "scores": {"path": str(scores_path), "sha256": sha256_of_file(scores_path)},
            "splits": split_source,
        },
        "code": {
            **source_hashes(Path(__file__)),
            **source_hashes(base),
        },
    }
    output = Path(raw["output_path"])
    write_json(report, output)
    for rule, result in rules.items():
        ci = result["bootstrap"]["threshold"]
        print(f"{rule}: {result['threshold']} [{ci['low']}, {ci['high']}]", flush=True)
    print(f"adopted ({adopted}): {report['udv_threshold']} -> {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the udv_v2 cuts on the calibration splits only: the cosine cut "
        "on window units, then the verifier cut on the same pairs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv_v2.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cosine", help="window cosine cut and the verifier pairs")
    commands.add_parser("verifier", help="verifier cut from the scored verifier pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    {"cosine": command_cosine, "verifier": command_verifier}[args.command](args)


if __name__ == "__main__":
    main()
