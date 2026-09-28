import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.build_udvs import (
    UdvConfig,
    encode_with_cache,
    load_config,
    load_encoder,
    load_lds_records,
    resolve_hearing_people,
    seed_everything,
    select_device,
    sentence_slices_by_person,
    write_json,
    write_jsonl,
)
from utils.udv_pipeline import (
    QUOTE_PREFIX_LENGTHS,
    SENTENCE_BOUNDARY_PATTERN,
    TRUSTED_PREFIX_WORDS,
    extract_quotes,
    normalize_whitespace,
    quote_prefix_pattern,
)

Record = dict[str, Any]

RELAXED_MODES = ("first_letter", "ignore_case")
ALL_MODES = ("exact", *RELAXED_MODES)
CALIBRATION_RECORDS = 20
COVERAGE_BUCKETS = ((1.0, "full"), (0.75, "at_least_75"), (0.5, "at_least_50"), (0.0, "below_50"))


def case_sensitive_prefix_pattern(prefix: str) -> re.Pattern[str]:
    left_boundary = "" if prefix[:1].isupper() else r"(?<!\w)"
    return re.compile(rf"{left_boundary}{re.escape(prefix)}(?!\w)")


def relaxed_prefix_pattern(prefix: str, mode: str) -> re.Pattern[str]:
    if mode == "exact":
        return case_sensitive_prefix_pattern(prefix)
    if mode == "ignore_case":
        return quote_prefix_pattern(prefix)
    first, rest = prefix[0], prefix[1:]
    lower, upper = first.lower(), first.upper()
    if lower == upper:
        head = rf"(?<!\w){re.escape(first)}"
    else:
        head = rf"(?:(?<!\w){re.escape(lower)}|{re.escape(upper)})"
    return re.compile(rf"{head}{re.escape(rest)}(?!\w)")


def find_quote_match(quote: str, speech: str, mode: str) -> Record | None:
    words = quote.split()
    for length in QUOTE_PREFIX_LENGTHS:
        prefix = " ".join(words[:length])
        if len(prefix) <= 5:
            continue
        pattern = relaxed_prefix_pattern(prefix, mode)
        hit = pattern.search(speech)
        if hit is not None:
            return {
                "quote": quote,
                "prefix": prefix,
                "prefix_words": min(length, len(words)),
                "start": hit.start(),
                "end": hit.end(),
                "hit_text": hit.group(0),
                "occurrences": len(pattern.findall(speech)),
            }
    return None


def find_opinion_match(opinion_text: str, speech: str, mode: str) -> Record | None:
    for quote in extract_quotes(opinion_text):
        found = find_quote_match(quote, speech, mode)
        if found is not None:
            return found
    return None


def enclosing_sentence_at(start: int, end: int, speech: str) -> str:
    covered = []
    position = 0
    for part in SENTENCE_BOUNDARY_PATTERN.split(speech):
        part_end = position + len(part)
        if position < end and start < part_end:
            covered.append(part)
        position = part_end + 1
    return normalize_whitespace(" ".join(covered))


def comparable_tokens(text: str) -> list[str]:
    return [token for token in (re.sub(r"\W", "", word.lower()) for word in text.split()) if token]


def matched_quote_words(quote: str, speech: str, start: int) -> tuple[int, int]:
    quote_tokens = comparable_tokens(quote)
    speech_tokens = comparable_tokens(speech[start:])[: len(quote_tokens)]
    count = 0
    for quote_token, speech_token in zip(quote_tokens, speech_tokens, strict=False):
        if quote_token != speech_token:
            break
        count += 1
    return count, len(quote_tokens)


def case_difference(prefix: str, hit_text: str) -> str:
    if hit_text == prefix:
        return "none"
    if hit_text[1:] == prefix[1:]:
        return "first_letter"
    return "other"


def describe_match(found: Record, speech: str) -> Record:
    matched, total = matched_quote_words(found["quote"], speech, found["start"])
    return {
        "prefix": found["prefix"],
        "prefix_words": found["prefix_words"],
        "hit_text": found["hit_text"],
        "case_difference": case_difference(found["prefix"], found["hit_text"]),
        "occurrences": found["occurrences"],
        "sentence": enclosing_sentence_at(found["start"], found["end"], speech),
        "matched_words": matched,
        "quote_words": total,
        "coverage": round(matched / total, 3) if total else 0.0,
    }


def match_status(exact: Record | None, relaxed: Record | None) -> str:
    if relaxed is None:
        return "none"
    if exact is None:
        return "new"
    if exact["prefix"] == relaxed["prefix"] and exact["sentence"] == relaxed["sentence"]:
        return "unchanged"
    return "changed"


def sentence_relation(sentence: str, current_text: str | None) -> str:
    if current_text is None:
        return "no_current_evidence"
    if sentence == current_text:
        return "same"
    if sentence in current_text or current_text in sentence:
        return "overlap"
    return "different"


def current_evidence_summary(record: Record | None) -> Record:
    if record is None:
        return {"tier": None, "support_type": None, "score": None, "text": None}
    evidence = record["evidence"] or {}
    return {
        "tier": record["tier"],
        "support_type": evidence.get("support_type"),
        "score": evidence.get("score"),
        "text": evidence.get("text"),
    }


def collect_candidates(hearings: list[Record], current_by_id: dict[str, Record]) -> list[Record]:
    candidates = []
    for hearing in hearings:
        for person in resolve_hearing_people(hearing):
            if not person["matched_turns"]:
                continue
            speech = person["speech"]
            for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"]):
                if not extract_quotes(opinion_text):
                    continue
                matches = {
                    mode: find_opinion_match(opinion_text, speech, mode) for mode in ALL_MODES
                }
                described = {
                    mode: describe_match(found, speech) if found is not None else None
                    for mode, found in matches.items()
                }
                statuses = {
                    mode: match_status(described["exact"], described[mode])
                    for mode in RELAXED_MODES
                }
                if all(status in ("none", "unchanged") for status in statuses.values()):
                    continue
                udv_id = f"udv-{hearing['id']}-{person['index']}-{opinion_index}"
                current = current_evidence_summary(current_by_id.get(udv_id))
                candidates.append(
                    {
                        "id": udv_id,
                        "hearing_id": hearing["id"],
                        "actor": person["participant"]["nome"],
                        "opinion": opinion_text,
                        "quote": cast(Record, matches["ignore_case"])["quote"],
                        "exact": described["exact"],
                        "first_letter": described["first_letter"],
                        "ignore_case": described["ignore_case"],
                        "status": statuses,
                        "current": current,
                        "sentence_vs_current": sentence_relation(
                            cast(Record, described["ignore_case"])["sentence"], current["text"]
                        ),
                    }
                )
    return candidates


def coverage_bucket(coverage: float) -> str:
    for floor, label in COVERAGE_BUCKETS:
        if coverage >= floor:
            return label
    return "below_50"


def summarize_mode(candidates: list[Record], mode: str) -> Record:
    new = [c for c in candidates if c["status"][mode] == "new"]
    changed = [c for c in candidates if c["status"][mode] == "changed"]
    return {
        "new": len(new),
        "changed": len(changed),
        "changed_with_new_sentence": sum(
            1 for c in changed if c[mode]["sentence"] != c["exact"]["sentence"]
        ),
        "new_by_prefix_words": dict(Counter(c[mode]["prefix_words"] for c in new)),
        "new_by_case_difference": dict(Counter(c[mode]["case_difference"] for c in new)),
        "new_by_coverage": dict(Counter(coverage_bucket(c[mode]["coverage"]) for c in new)),
        "new_with_multiple_occurrences": sum(1 for c in new if c[mode]["occurrences"] > 1),
        "new_by_current_tier": dict(Counter(c["current"]["tier"] for c in new)),
        "new_by_sentence_vs_current": dict(
            Counter(sentence_relation(c[mode]["sentence"], c["current"]["text"]) for c in new)
        ),
        "new_low_coverage_and_different_sentence": sum(
            1
            for c in new
            if c[mode]["coverage"] < 0.75
            and sentence_relation(c[mode]["sentence"], c["current"]["text"]) == "different"
        ),
    }


def calibration_pairs(hearings: list[Record], mode: str, seed: int) -> list[tuple[str, str, str]]:
    random.seed(seed)
    pairs = []
    pool: list[str] = []
    per_hearing = []
    for hearing in hearings:
        people = [p for p in resolve_hearing_people(hearing) if p["sentences"]]
        per_hearing.append(people)
        pool.extend(sentence for person in people for sentence in person["sentences"])
    for people in per_hearing:
        for person in people:
            for opinion_text in person["participant"]["opinioes"]:
                found = find_opinion_match(opinion_text, " ".join(person["sentences"]), mode)
                if found is None:
                    continue
                pattern = relaxed_prefix_pattern(found["prefix"], mode)
                true_sentence = next((s for s in person["sentences"] if pattern.search(s)), None)
                if true_sentence is None:
                    continue
                other_sentence = random.choice([s for s in pool if s != true_sentence])
                pairs.append((opinion_text, true_sentence, other_sentence))
    return pairs


def pair_similarities(encoder: Any, pairs: list[tuple[str, str, str]], config: UdvConfig) -> Record:
    opinions = [pair[0] for pair in pairs]
    positives = [pair[1] for pair in pairs]
    negatives = [pair[2] for pair in pairs]
    embeddings = encoder.encode(
        opinions + positives + negatives,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    count = len(pairs)
    opinion_vectors = embeddings[:count]
    positive_scores = np.diag(cosine_similarity(opinion_vectors, embeddings[count : 2 * count]))
    negative_scores = np.diag(cosine_similarity(opinion_vectors, embeddings[2 * count :]))
    return {"positive": positive_scores, "negative": negative_scores}


def calibration_report(scores: Record) -> Record:
    positive = scores["positive"]
    negative = scores["negative"]
    threshold = round(
        (float(np.quantile(positive, 0.25)) + float(np.quantile(negative, 0.75))) / 2, 2
    )
    return {
        "pairs": int(len(positive)),
        "positive_q25": round(float(np.quantile(positive, 0.25)), 3),
        "positive_median": round(float(np.median(positive)), 3),
        "positive_min": round(float(positive.min()), 3),
        "negative_q75": round(float(np.quantile(negative, 0.75)), 3),
        "negative_max": round(float(negative.max()), 3),
        "threshold": threshold,
        "positives_below_threshold": int((positive < threshold).sum()),
        "negatives_above_threshold": int((negative >= threshold).sum()),
    }


def accepted_by_prefix(prefix_words: int, _relation: str) -> bool:
    return prefix_words >= TRUSTED_PREFIX_WORDS


def accepted_by_prefix_or_encoder(prefix_words: int, relation: str) -> bool:
    return prefix_words >= TRUSTED_PREFIX_WORDS or relation in ("same", "overlap")


def accepted_always(_prefix_words: int, _relation: str) -> bool:
    return True


POLICIES = {
    "accept_all": accepted_always,
    "prefix_rule": accepted_by_prefix,
    "prefix_or_encoder_rule": accepted_by_prefix_or_encoder,
}


def embedding_top_matches(
    hearings: list[Record],
    quote_records: list[Record],
    encoder: Any,
    config: UdvConfig,
    device: str,
) -> dict[str, Record]:
    wanted: dict[int, list[Record]] = {}
    for record in quote_records:
        wanted.setdefault(record["hearing_id"], []).append(record)
    result = {}
    for hearing in hearings:
        if hearing["id"] not in wanted:
            continue
        people = resolve_hearing_people(hearing)
        all_sentences = [sentence for person in people for sentence in person["sentences"]]
        sentence_embeddings = encode_with_cache(
            encoder, all_sentences, config, device, f"sentences_{hearing['id']}"
        )
        opinions = [
            (person["index"], opinion_index, opinion_text)
            for person in people
            for opinion_index, opinion_text in enumerate(person["participant"]["opinioes"])
        ]
        opinion_embeddings = encode_with_cache(
            encoder, [text for _, _, text in opinions], config, device, f"opinions_{hearing['id']}"
        )
        positions = {(p, o): i for i, (p, o, _) in enumerate(opinions)}
        slices = sentence_slices_by_person(people)
        people_by_index = {person["index"]: person for person in people}
        for record in wanted[hearing["id"]]:
            _, _, person_index, opinion_index = record["id"].split("-")
            person = people_by_index[int(person_index)]
            if not person["sentences"]:
                continue
            similarities = cosine_similarity(
                opinion_embeddings[positions[(int(person_index), int(opinion_index))]].reshape(
                    1, -1
                ),
                sentence_embeddings[slices[int(person_index)]],
            ).flatten()
            best = int(similarities.argmax())
            result[record["id"]] = {
                "sentence": person["sentences"][best],
                "score": float(similarities[best]),
            }
    return result


def add_encoder_relation(candidates: list[Record], top_matches: dict[str, Record]) -> None:
    for candidate in candidates:
        for mode in RELAXED_MODES:
            if candidate["status"][mode] == "none":
                candidate[mode + "_vs_embedding_top"] = None
                continue
            top = top_matches.get(candidate["id"])
            reference = top["sentence"] if top else candidate["current"]["text"]
            candidate[mode + "_vs_embedding_top"] = sentence_relation(
                candidate[mode]["sentence"], reference
            )


def describe_existing_quotes(
    quote_records: list[Record], top_matches: dict[str, Record]
) -> list[Record]:
    rows = []
    for record in quote_records:
        top = top_matches.get(record["id"])
        rows.append(
            {
                "id": record["id"],
                "prefix_words": len(record["evidence"]["quote_prefix"].split()),
                "sentence_vs_embedding_top": sentence_relation(
                    record["evidence"]["text"], top["sentence"] if top else None
                ),
                "embedding_top_score": round(top["score"], 4) if top else None,
            }
        )
    return rows


def summarize_existing_quotes(existing: list[Record]) -> Record:
    short = [row for row in existing if row["prefix_words"] < TRUSTED_PREFIX_WORDS]
    return {
        "total": len(existing),
        "by_prefix_words": dict(Counter(row["prefix_words"] for row in existing)),
        "short_prefix_total": len(short),
        "short_prefix_agreeing_with_embedding_top": sum(
            1 for row in short if row["sentence_vs_embedding_top"] in ("same", "overlap")
        ),
    }


def project_tiers_policy(
    current_records: list[Record],
    candidates: list[Record],
    existing: list[Record],
    mode: str,
    threshold: float,
    accepted: Any,
) -> Record:
    existing_by_id = {row["id"]: row for row in existing}
    candidates_by_id = {c["id"]: c for c in candidates}
    counter: Counter[str] = Counter()

    def semantic_tier(score: float | None) -> str:
        if score is None:
            return "no_evidence"
        return "semantic_match_high" if score >= threshold else "semantic_match_weak"

    for record in current_records:
        candidate = candidates_by_id.get(record["id"])
        status = candidate["status"][mode] if candidate else "none"
        if status in ("new", "changed"):
            found = cast(Record, candidate)
            keep = accepted(found[mode]["prefix_words"], found[mode + "_vs_embedding_top"])
            if keep:
                counter["quote_found"] += 1
            elif status == "new":
                counter[semantic_tier(record["evidence"]["score"])] += 1
            else:
                counter[semantic_tier(existing_by_id[record["id"]]["embedding_top_score"])] += 1
        elif record["tier"] == "quote_found":
            row = existing_by_id[record["id"]]
            keep = accepted(row["prefix_words"], row["sentence_vs_embedding_top"])
            counter["quote_found" if keep else semantic_tier(row["embedding_top_score"])] += 1
        elif record["tier"] in ("semantic_match_high", "semantic_match_weak"):
            counter[semantic_tier(record["evidence"]["score"])] += 1
        else:
            counter[record["tier"]] += 1
    return dict(counter)


def review_template(
    candidates: list[Record], mode: str, sample_size: int, seed: int
) -> list[Record]:
    new = [c for c in candidates if c["status"][mode] == "new"]
    random.seed(seed)
    sampled = random.sample(new, min(sample_size, len(new)))
    return [
        {
            "id": c["id"],
            "hearing_id": c["hearing_id"],
            "actor": c["actor"],
            "opinion": c["opinion"],
            "quote": c["quote"],
            "prefix": c[mode]["prefix"],
            "sentence": c[mode]["sentence"],
            "current_evidence": c["current"]["text"],
            "judgment": None,
            "note": None,
        }
        for c in sorted(sampled, key=lambda c: c["id"])
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure what relaxing letter case in quote matching would change in the UDVs."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/udv.toml"))
    parser.add_argument("--run-name", default="udv_v0", help="baseline run to compare against")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--skip-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.seed)
    hearings = load_lds_records(config)
    with open(config.output_dir / f"{args.run_name}.jsonl") as f:
        current_records = [json.loads(line) for line in f]
    current_by_id = {record["id"]: record for record in current_records}

    candidates = collect_candidates(hearings, current_by_id)
    summary: Record = {
        "baseline_run": args.run_name,
        "hearings": len(hearings),
        "candidates": len(candidates),
        "modes": {mode: summarize_mode(candidates, mode) for mode in RELAXED_MODES},
        "current_tiers": dict(Counter(record["tier"] for record in current_records)),
    }
    output_dir = config.output_dir
    if not args.skip_encoder:
        device = select_device(config.device)
        encoder = load_encoder(config, device)
        calibration = {}
        for mode in ALL_MODES:
            pairs = calibration_pairs(hearings[:CALIBRATION_RECORDS], mode, config.seed)
            calibration[mode] = calibration_report(pair_similarities(encoder, pairs, config))
        summary["calibration"] = calibration
        quote_records = [r for r in current_records if r["tier"] == "quote_found"]
        top_matches = embedding_top_matches(hearings, quote_records, encoder, config, device)
        add_encoder_relation(candidates, top_matches)
        existing = describe_existing_quotes(quote_records, top_matches)
        summary["existing_quote_found"] = summarize_existing_quotes(existing)
        summary["policies"] = {
            "trusted_prefix_words": TRUSTED_PREFIX_WORDS,
            "encoder_condition": "quote sentence equals or overlaps the encoder top-1 sentence",
        }
        summary["projected_tiers"] = {
            mode: {
                name: project_tiers_policy(
                    current_records, candidates, existing, mode, config.embedding_threshold, rule
                )
                for name, rule in POLICIES.items()
            }
            for mode in RELAXED_MODES
        }
        summary["projected_tiers_recalibrated_accept_all"] = {
            mode: project_tiers_policy(
                current_records,
                candidates,
                existing,
                mode,
                calibration[mode]["threshold"],
                accepted_always,
            )
            for mode in RELAXED_MODES
        }
        write_jsonl(existing, output_dir / "case_insensitive_existing_quotes.jsonl")

    write_jsonl(candidates, output_dir / "case_insensitive_quotes.jsonl")
    write_json(summary, output_dir / "case_insensitive_quotes_summary.json")
    write_json(
        review_template(candidates, "ignore_case", args.sample_size, config.seed),
        output_dir / "case_insensitive_review_template.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
