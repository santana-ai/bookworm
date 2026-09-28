import argparse
import dataclasses
import json
import logging
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from utils.actor_simulation import (
    CHAIR_ROLE,
    DEFAULT_CONFIG,
    NO_BASIS,
    Excerpt,
    Generation,
    Material,
    SimulationConfig,
    SimulationModel,
    SimulationPrompts,
    SpeechRetriever,
    chat_messages,
    check_disjoint,
    clean_role,
    file_info,
    fingerprint,
    linked_udvs,
    load_config,
    load_profiles,
    load_prompts,
    load_rows,
    number_profile,
    parse_level,
    profile_ids,
    render,
    split_hearings,
    turn_owners,
    turn_sentences,
)
from utils.dataset_io import load_jsonl, write_json, write_jsonl
from utils.generate_actor_profiles import hearing_metadata
from utils.udv_pipeline import (
    TRUSTED_PREFIX_WORDS,
    find_quote_match,
    is_trusted_quote,
    load_encoder_spec,
    normalize_whitespace,
)

Record = dict[str, Any]

REFUSAL = "o material não permite estimar"
APPROACHES = ("1", "2")
QUOTE_MARKS = '"“”'
TRAILING_PUNCTUATION = ".!?…;:,\"'”) "
CHECK_RULE = (
    "JSON objects are read anywhere in the justification (one per line is asked, but objects"
    " spread over several lines or inside code fences are read too; fragments that do not parse"
    " are counted in parse_failures). A sentence of the simulated speech is grounded when at"
    " least one object's 'frase' contains it (whitespace-normalized, ignoring trailing"
    " punctuation and leading quote marks) and every such object has a non-empty 'apoio', only"
    " ids that exist in the material of the call, and, for each cited example or excerpt, a"
    " 'copias' entry that,"
    " without enclosing quote marks, has at least 6 words and whose first 6 to 10 words are"
    " found in it by the UDV quote-prefix rule (otherwise copy_missing, copy_too_short or"
    " copy_not_found). This shows that the cited item exists and that the start of the copy is"
    " literal, not that the item supports the sentence."
)


def leading_sentences(sentences: list[str], max_words: int) -> str:
    chosen: list[str] = []
    words = 0
    for sentence in sentences:
        words += len(sentence.split())
        if words > max_words:
            break
        chosen.append(sentence)
    return " ".join(chosen)


def example_candidates(
    record: Record, metadata: dict[int, Record], config: SimulationConfig
) -> list[Excerpt]:
    candidates = []
    for hearing in record["hearings"]:
        info = metadata[hearing["hearing_id"]]
        for turn in hearing["turns"]:
            if turn["role"] == CHAIR_ROLE:
                continue
            text = leading_sentences(turn_sentences(turn["text"])[1:], config.example_max_words)
            if len(text.split()) >= config.example_min_words:
                candidates.append(
                    Excerpt(
                        date=info["date_br"],
                        assunto=info["assunto"],
                        text=text,
                        chair=False,
                        source={
                            "hearing_id": hearing["hearing_id"],
                            "turn_index": turn["turn_index"],
                        },
                    )
                )
    return candidates


def speech_examples(
    record: Record, metadata: dict[int, Record], config: SimulationConfig
) -> tuple[Excerpt, ...]:
    by_hearing: dict[int, list[Excerpt]] = defaultdict(list)
    for candidate in example_candidates(record, metadata, config):
        by_hearing[candidate.source["hearing_id"]].append(candidate)
    rng = random.Random(f"{config.generation_seed}:{record['actor']}")
    if len(by_hearing) >= config.examples:
        hearings = rng.sample(sorted(by_hearing), config.examples)
        chosen = [rng.choice(by_hearing[hearing_id]) for hearing_id in hearings]
    else:
        pool = [
            candidate for hearing_id in sorted(by_hearing) for candidate in by_hearing[hearing_id]
        ]
        chosen = rng.sample(pool, min(config.examples, len(pool)))
    return tuple(
        sorted(
            chosen,
            key=lambda example: (
                metadata[example.source["hearing_id"]]["date"],
                example.source["hearing_id"],
                example.source["turn_index"],
            ),
        )
    )


def latest_role(
    record: Record,
    udvs: list[Record],
    owners: dict[tuple[int, int], str],
    metadata: dict[int, Record],
    parties: tuple[str, ...],
) -> str | None:
    hearings = {hearing["hearing_id"] for hearing in record["hearings"]}
    linked = [udv for udv, _ in linked_udvs(udvs, owners, {record["actor"]}, hearings)]
    if not linked:
        return None
    latest = max(linked, key=lambda udv: (metadata[udv["hearing_id"]]["date"], udv["hearing_id"]))
    return clean_role(latest["actor"]["role"], parties)


def copy_problem(copy: Any, source_id: str, source: str) -> str | None:
    if not isinstance(copy, str):
        return f"copy_missing:{source_id}"
    text = normalize_whitespace(copy).strip(QUOTE_MARKS).strip()
    if len(text.split()) < TRUSTED_PREFIX_WORDS:
        return f"copy_too_short:{source_id}"
    if not is_trusted_quote(find_quote_match(text, source)):
        return f"copy_not_found:{source_id}"
    return None


def check_line(item: Record, known_ids: set[str], sources: dict[str, str]) -> Record:
    support = item.get("apoio")
    support = (
        [value for value in support if isinstance(value, str)] if isinstance(support, list) else []
    )
    copies = cast(Record, item.get("copias")) if isinstance(item.get("copias"), dict) else {}
    problems = [] if support else ["empty_support"]
    for source_id in support:
        if source_id not in known_ids:
            problems.append(f"unknown_id:{source_id}")
        elif source_id in sources:
            problem = copy_problem(copies.get(source_id), source_id, sources[source_id])
            if problem is not None:
                problems.append(problem)
    return {
        "frase": normalize_whitespace(item["frase"]),
        "apoio": support,
        "copias": copies,
        "problems": problems,
    }


def json_objects(text: str) -> tuple[list[Any], int]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    failures = 0
    position = 0
    while (start := text.find("{", position)) != -1:
        try:
            item, position = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            failures += 1
            position = start + 1
            continue
        objects.append(item)
    return objects, failures


def sentence_key(text: str) -> str:
    return normalize_whitespace(text).rstrip(TRAILING_PUNCTUATION).lstrip(QUOTE_MARKS)


def check_justification(speech: str, justification: str, material: Material) -> Record:
    sources = {f"E{i}": example.text for i, example in enumerate(material.examples, start=1)}
    sources |= {f"T{i}": excerpt.text for i, excerpt in enumerate(material.excerpts, start=1)}
    known_ids = profile_ids(material.profile) | set(sources)
    objects, parse_failures = json_objects(justification)
    lines: list[Record] = []
    invalid: list[Any] = []
    for item in objects:
        if isinstance(item, dict) and isinstance(item.get("frase"), str) and item["frase"].strip():
            lines.append(check_line(item, known_ids, sources))
        else:
            invalid.append(item)
    sentences = []
    for sentence in turn_sentences(speech):
        key = sentence_key(sentence)
        matched = [line for line in lines if key and key in line["frase"]]
        sentences.append(
            {
                "sentence": sentence,
                "lines": len(matched),
                "grounded": bool(matched) and all(not line["problems"] for line in matched),
            }
        )
    speech_text = normalize_whitespace(speech)
    return {
        "sentences": sentences,
        "lines": lines,
        "invalid_items": invalid,
        "parse_failures": parse_failures,
        "lines_not_in_speech": sum(
            sentence_key(line["frase"]) not in speech_text for line in lines
        ),
        "n_sentences": len(sentences),
        "n_ungrounded": sum(not sentence["grounded"] for sentence in sentences),
    }


def excerpt_record(excerpt: Excerpt) -> Record:
    return {
        **excerpt.source,
        "date": excerpt.date,
        "assunto": excerpt.assunto,
        "text": excerpt.text,
    }


@dataclass(frozen=True)
class Simulator:
    config: SimulationConfig
    prompts: SimulationPrompts
    model: SimulationModel
    retriever: SpeechRetriever
    k: int

    def messages(self, material: Material, request: Record, ask: str) -> list[Record]:
        return chat_messages(self.prompts, material, request["date"], request["topic"], ask)

    def level(self, material: Material, request: Record) -> Record:
        ask = render(self.prompts.evidence, name=material.name)
        generation = self.model.generate(
            self.messages(material, request, ask), self.config.evidence_max_tokens
        )
        return {"label": parse_level(generation.text), "text": generation.text}

    def speak(self, material: Material, request: Record) -> Generation:
        ask = render(self.prompts.speech, name=material.name)
        return self.model.generate(
            self.messages(material, request, ask), self.config.speech_max_tokens
        )

    def justify(self, material: Material, request: Record, speech: str) -> Generation:
        ask = render(self.prompts.justification, speech=speech)
        return self.model.generate(
            self.messages(material, request, ask), self.config.justification_max_tokens
        )

    def run(
        self, request: Record, profile: Record, role: str | None, examples: tuple[Excerpt, ...]
    ) -> Record:
        base = Material(
            name=request["actor"],
            role=role,
            profile=number_profile(profile["profile"]),
            examples=examples,
        )
        with_excerpts = base.with_excerpts(
            self.retriever.retrieve(request["actor"], request["topic"], self.k)
        )
        plans = (("1", base), ("2", with_excerpts))
        approaches: Record = {}
        for approach, material in plans:
            level = self.level(material, request)
            if level["label"] == NO_BASIS:
                approaches[approach] = {"level": level, "output": REFUSAL}
                continue
            speech = self.speak(material, request)
            justification = self.justify(material, request, speech.text)
            approaches[approach] = {
                "level": level,
                "output": speech.text,
                "speech": dataclasses.asdict(speech),
                "justification": dataclasses.asdict(justification),
                "check": check_justification(speech.text, justification.text, material),
            }
        return {
            **request,
            "role": role,
            "profile_prompt_version": profile["prompt_version"],
            "k": self.k,
            "examples": [excerpt_record(example) for example in examples],
            "excerpts": [excerpt_record(excerpt) for excerpt in with_excerpts.excerpts],
            "baseline_speech": dataclasses.asdict(self.speak(base.baseline(), request)),
            "approaches": approaches,
        }


def split_requests(
    config: SimulationConfig,
    profiles: dict[str, Record],
    udvs: list[Record],
    owners: dict[tuple[int, int], str],
    metadata: dict[int, Record],
) -> list[Record]:
    hearings = split_hearings(config, config.requests_split)
    pairs = sorted(
        {
            (actor, udv["hearing_id"])
            for udv, actor in linked_udvs(udvs, owners, set(profiles), hearings)
        }
    )
    return [
        {
            "request_id": f"{actor}|{hearing_id}",
            "actor": actor,
            "hearing_id": hearing_id,
            "date": metadata[hearing_id]["date_br"],
            "topic": metadata[hearing_id]["assunto"],
        }
        for actor, hearing_id in pairs
    ]


def file_requests(path: Path, profiles: dict[str, Record]) -> list[Record]:
    requests = [
        {
            "request_id": f"{row['actor']}|{row['date']}|{row['topic']}",
            "actor": row["actor"],
            "hearing_id": None,
            "date": row["date"],
            "topic": row["topic"],
        }
        for row in load_jsonl(path)
    ]
    missing = sorted({request["actor"] for request in requests} - set(profiles))
    if missing:
        raise SystemExit(f"requests for actors without profile: {missing}")
    return requests


def selected_k(config: SimulationConfig) -> int | None:
    path = config.output_dir / "evaluation.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["selection"]["k"]


def summarize(rows: list[Record]) -> Record:
    summary: Record = {"requests": len(rows)}
    for approach in APPROACHES:
        results = [row["approaches"][approach] for row in rows]
        checks = [result["check"] for result in results if "check" in result]
        sentences = sum(check["n_sentences"] for check in checks)
        ungrounded = sum(check["n_ungrounded"] for check in checks)
        summary[approach] = {
            "levels": dict(Counter(str(result["level"]["label"]) for result in results)),
            "refusals": sum(result["output"] == REFUSAL for result in results),
            "speeches": len(checks),
            "truncated_speeches": sum(
                result["speech"]["truncated"] for result in results if "speech" in result
            ),
            "truncated_justifications": sum(
                result["justification"]["truncated"]
                for result in results
                if "justification" in result
            ),
            "sentences": sentences,
            "ungrounded_sentences": ungrounded,
            "ungrounded_fraction": ungrounded / sentences if sentences else None,
            "invalid_justification_items": sum(len(check["invalid_items"]) for check in checks),
            "justification_parse_failures": sum(check["parse_failures"] for check in checks),
            "lines_not_in_speech": sum(check["lines_not_in_speech"] for check in checks),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Open generation of simulated speeches (profile, profile + excerpts), with evidence"
            " level and a checked justification, plus the speech from the baseline material."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Hugging Face model id or local path (overrides config)")
    parser.add_argument("--actors", nargs="*", help="only these profiled actors, by exact name")
    parser.add_argument(
        "--requests",
        type=Path,
        help="JSONL with actor, date (DD/MM/YYYY) and topic; default: one request per actor and"
        " hearing of the requests split with a linked UDV",
    )
    parser.add_argument("--k", type=int, help="excerpts per request (default: evaluation.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.model is not None:
        config = dataclasses.replace(config, model=args.model)
    if not config.model:
        parser.error("set [model] name in the config, or pass --model")
    k = args.k if args.k is not None else selected_k(config)
    if k is None:
        parser.error("run utils.evaluate_actor_simulation first, or pass --k")
    prompts = load_prompts(config.prompts_dir)
    profiles = load_profiles(config.profiles_path, args.actors)
    train_records = {
        record["actor"]: record
        for record in load_jsonl(config.train_speeches_path)
        if record["actor"] in profiles
    }
    metadata = hearing_metadata(config.lds_path, config.lds_sha256)
    udvs = load_jsonl(config.udv_path)
    owners = turn_owners(load_jsonl(config.speeches_path))
    if args.requests is not None:
        requests = file_requests(args.requests, profiles)
        check_disjoint(profiles, list(train_records.values()), set())
    else:
        requests = split_requests(config, profiles, udvs, owners, metadata)
        check_disjoint(
            profiles, list(train_records.values()), split_hearings(config, config.requests_split)
        )
    if not requests:
        raise SystemExit("no requests to simulate")
    inputs = {
        "profiles": file_info(config.profiles_path),
        "train_speeches": file_info(config.train_speeches_path),
        "udv": file_info(config.udv_path),
    }
    run_digest = fingerprint(
        config.model,
        config.evidence_max_tokens,
        config.speech_max_tokens,
        config.justification_max_tokens,
        load_encoder_spec(),
        prompts.version,
        inputs["train_speeches"],
        k,
    )
    logging.info(
        "%d requests, k=%d, prompts %s, loading %s",
        len(requests),
        k,
        prompts.version,
        config.model,
    )
    simulator = Simulator(
        config=config,
        prompts=prompts,
        model=SimulationModel(config.model, config.device_map),
        retriever=SpeechRetriever(list(train_records.values()), metadata),
        k=k,
    )
    path = config.output_dir / "simulations.jsonl"
    done = load_rows(path, "request_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(path, "a") as output:
        for index, request in enumerate(requests, start=1):
            actor = request["actor"]
            record = train_records[actor]
            role = latest_role(record, udvs, owners, metadata, config.parties)
            examples = speech_examples(record, metadata, config)
            key = fingerprint(
                run_digest,
                profiles[actor]["profile"],
                request,
                role,
                [excerpt_record(example) for example in examples],
            )
            row = done.get(request["request_id"])
            if row is None or row["fingerprint"] != key:
                started = time.monotonic()
                simulated = simulator.run(request, profiles[actor], role, examples)
                row = {
                    **simulated,
                    "model": config.model,
                    "prompt_version": prompts.version,
                    "fingerprint": key,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                logging.info(
                    "[%d/%d] %s: %.1fs",
                    index,
                    len(requests),
                    request["request_id"],
                    time.monotonic() - started,
                )
            rows.append(row)
    write_jsonl(rows, path)
    summary = {
        "model": config.model,
        "prompt_version": prompts.version,
        "k": k,
        "inputs": inputs,
        "requests_source": str(args.requests) if args.requests else config.requests_split,
        "check_rule": CHECK_RULE,
        **summarize(rows),
    }
    write_json(summary, config.output_dir / "simulations_summary.json")
    for approach in APPROACHES:
        result = summary[approach]
        print(
            f"approach {approach}: {result['speeches']} speeches, {result['refusals']} refusals,"
            f" {result['ungrounded_sentences']}/{result['sentences']} ungrounded sentences"
        )
    print(f"wrote {path} and {config.output_dir / 'simulations_summary.json'}")


if __name__ == "__main__":
    main()
