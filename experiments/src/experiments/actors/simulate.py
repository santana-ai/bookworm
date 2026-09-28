"""Open generation of simulated speeches: for each request (actor, date, topic), an evidence
level, a speech and its checked justification from the profile (approach 1) and from the
profile plus retrieved train excerpts (approach 2), and a speech from the baseline material."""

import argparse
import dataclasses
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, write_json

from experiments.actors.backend import SimulationModel
from experiments.actors.build_speeches import CHAIR_ROLE
from experiments.actors.chat import Generation, SimulationBackend
from experiments.actors.cli import configure_logging, require_model, write_report
from experiments.actors.io import Job, canonical_sha256, file_info, fingerprint, run_resumable
from experiments.actors.justification import CHECK_RULE, check_justification
from experiments.actors.simulation import (
    NO_BASIS,
    Excerpt,
    Material,
    SimulationConfig,
    SimulationInputs,
    SimulationPrompts,
    SpeechRetriever,
    TurnKey,
    add_run_arguments,
    ask_evidence_level,
    chat_messages,
    check_disjoint,
    clean_role,
    config_from_args,
    linked_udvs,
    load_inputs,
    number_profile,
    render,
    split_hearings,
    turn_sentences,
)
from experiments.common.transcript import load_encoder_spec

Record = dict[str, Any]

REFUSAL = "o material não permite estimar"
APPROACHES = ("1", "2")
EVALUATION_FILE = "evaluation.json"
SIMULATIONS_FILE = "simulations.jsonl"
SUMMARY_FILE = "simulations_summary.json"
K_RULE = "k comes from evaluation.json of the model run, or --k; null before that run"
DRY_RUN_RULE = (
    "requests, roles and speech examples are built as in a model run; an equal hash means"
    " the model receives the same requests, roles and examples (the excerpts depend on k)"
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
    """The non-chair turns of the actor, cut to their leading sentences after the first, that
    keep enough words to show how the actor speaks."""
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
    """Seeded draw of the speech examples, from distinct hearings when there are enough."""
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
    owners: dict[TurnKey, str],
    metadata: dict[int, Record],
    parties: tuple[str, ...],
) -> str | None:
    """The cleaned role of the actor's most recent linked UDV in the train hearings."""
    hearings = {hearing["hearing_id"] for hearing in record["hearings"]}
    linked = [udv for udv, _ in linked_udvs(udvs, owners, {record["actor"]}, hearings)]
    if not linked:
        return None
    latest = max(linked, key=lambda udv: (metadata[udv["hearing_id"]]["date"], udv["hearing_id"]))
    return clean_role(latest["actor"]["role"], parties)


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
    model: SimulationBackend
    retriever: SpeechRetriever
    k: int

    def _generate(self, material: Material, request: Record, ask: str, tokens: int) -> Generation:
        messages = chat_messages(self.prompts, material, request["date"], request["topic"], ask)
        return self.model.generate(messages, tokens)

    def level(self, material: Material, request: Record) -> Record:
        return ask_evidence_level(
            self.model,
            self.prompts,
            material,
            request["date"],
            request["topic"],
            self.config.evidence_max_tokens,
        )

    def speak(self, material: Material, request: Record) -> Generation:
        ask = render(self.prompts.speech, name=material.name)
        return self._generate(material, request, ask, self.config.speech_max_tokens)

    def justify(self, material: Material, request: Record, speech: str) -> Generation:
        ask = render(self.prompts.justification, speech=speech)
        return self._generate(material, request, ask, self.config.justification_max_tokens)

    def approach(self, material: Material, request: Record) -> Record:
        """Evidence level, then, unless it is SEM BASE, a speech and its checked justification."""
        level = self.level(material, request)
        if level["label"] == NO_BASIS:
            return {"level": level, "output": REFUSAL}
        speech = self.speak(material, request)
        justification = self.justify(material, request, speech.text)
        return {
            "level": level,
            "output": speech.text,
            "speech": dataclasses.asdict(speech),
            "justification": dataclasses.asdict(justification),
            "check": check_justification(speech.text, justification.text, material),
        }

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
        approaches = {
            approach: self.approach(material, request)
            for approach, material in zip(APPROACHES, (base, with_excerpts), strict=True)
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

    def row(
        self, request: Record, profile: Record, role: str | None, examples: tuple[Excerpt, ...]
    ) -> Record:
        return {
            **self.run(request, profile, role, examples),
            "model": self.config.model,
            "prompt_version": self.prompts.version,
        }


def split_requests(
    config: SimulationConfig,
    profiles: dict[str, Record],
    udvs: list[Record],
    owners: dict[TurnKey, str],
    metadata: dict[int, Record],
) -> list[Record]:
    """One request per profiled actor and hearing of the requests split with a linked UDV."""
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


def load_requests(
    config: SimulationConfig, inputs: SimulationInputs, path: Path | None
) -> list[Record]:
    """The requests of the file, or of the requests split, after checking that no profile or
    train speech comes from a hearing that is asked about."""
    if path is not None:
        requests = file_requests(path, inputs.profiles)
        check_disjoint(inputs.profiles, inputs.train_records, set())
    else:
        requests = split_requests(
            config, inputs.profiles, inputs.udvs, inputs.owners, inputs.metadata
        )
        check_disjoint(
            inputs.profiles,
            inputs.train_records,
            split_hearings(config, config.requests_split),
        )
    return requests


def selected_k(config: SimulationConfig) -> int | None:
    path = config.output_dir / EVALUATION_FILE
    if not path.exists():
        return None
    with open(path) as f:
        k: int = json.load(f)["selection"]["k"]
    return k


def approach_summary(results: list[Record]) -> Record:
    checks = [result["check"] for result in results if "check" in result]
    sentences = sum(check["n_sentences"] for check in checks)
    ungrounded = sum(check["n_ungrounded"] for check in checks)
    return {
        "levels": dict(Counter(str(result["level"]["label"]) for result in results)),
        "refusals": sum(result["output"] == REFUSAL for result in results),
        "speeches": len(checks),
        "truncated_speeches": sum(
            result["speech"]["truncated"] for result in results if "speech" in result
        ),
        "truncated_justifications": sum(
            result["justification"]["truncated"] for result in results if "justification" in result
        ),
        "sentences": sentences,
        "ungrounded_sentences": ungrounded,
        "ungrounded_fraction": ungrounded / sentences if sentences else None,
        "invalid_justification_items": sum(len(check["invalid_items"]) for check in checks),
        "justification_parse_failures": sum(check["parse_failures"] for check in checks),
        "lines_not_in_speech": sum(check["lines_not_in_speech"] for check in checks),
    }


def summarize(rows: list[Record]) -> Record:
    summary: Record = {"requests": len(rows)}
    for approach in APPROACHES:
        summary[approach] = approach_summary([row["approaches"][approach] for row in rows])
    return summary


def request_dry_run(
    config: SimulationConfig,
    inputs: SimulationInputs,
    input_files: Record,
    requests: list[Record],
    k: int | None,
) -> Record:
    records = {record["actor"]: record for record in inputs.train_records}
    actors = sorted({request["actor"] for request in requests})
    roles = {
        actor: latest_role(
            records[actor], inputs.udvs, inputs.owners, inputs.metadata, config.parties
        )
        for actor in actors
    }
    examples = {
        actor: [
            excerpt_record(example)
            for example in speech_examples(records[actor], inputs.metadata, config)
        ]
        for actor in actors
    }
    return {
        "dry_run": True,
        "model": config.model or None,
        "prompt_version": inputs.prompts.version,
        "k": k,
        "k_rule": K_RULE,
        "inputs": input_files,
        "requests_split": config.requests_split,
        "requests": len(requests),
        "request_actors": len(actors),
        "request_hearings": len({request["hearing_id"] for request in requests}),
        "requests_sha256": canonical_sha256(requests),
        "roles_sha256": canonical_sha256(roles),
        "actors_without_role": sum(role is None for role in roles.values()),
        "examples_sha256": canonical_sha256(examples),
        "rule": DRY_RUN_RULE,
    }


def simulate_requests(
    config: SimulationConfig,
    inputs: SimulationInputs,
    input_files: Record,
    requests: list[Record],
    k: int,
) -> list[Record]:
    """Simulate every request whose row in simulations.jsonl is missing or stale."""
    prompts = inputs.prompts
    run_digest = fingerprint(
        config.model,
        config.evidence_max_tokens,
        config.speech_max_tokens,
        config.justification_max_tokens,
        load_encoder_spec(),
        prompts.version,
        input_files["train_speeches"],
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
        retriever=SpeechRetriever(inputs.train_records, inputs.metadata),
        k=k,
    )
    records = {record["actor"]: record for record in inputs.train_records}

    def jobs() -> Iterator[Job]:
        for request in requests:
            actor = request["actor"]
            record = records[actor]
            role = latest_role(record, inputs.udvs, inputs.owners, inputs.metadata, config.parties)
            examples = speech_examples(record, inputs.metadata, config)
            key = fingerprint(
                run_digest,
                inputs.profiles[actor]["profile"],
                request,
                role,
                [excerpt_record(example) for example in examples],
            )
            yield Job(
                id=request["request_id"],
                label=request["request_id"],
                fingerprint=key,
                compute=partial(simulator.row, request, inputs.profiles[actor], role, examples),
            )

    path = config.output_dir / SIMULATIONS_FILE
    return run_resumable(path, "request_id", jobs(), len(requests))


def write_summary(
    config: SimulationConfig,
    prompts: SimulationPrompts,
    input_files: Record,
    requests_source: str,
    k: int,
    rows: list[Record],
) -> None:
    summary = {
        "model": config.model,
        "prompt_version": prompts.version,
        "k": k,
        "inputs": input_files,
        "requests_source": requests_source,
        "check_rule": CHECK_RULE,
        **summarize(rows),
    }
    write_json(summary, config.output_dir / SUMMARY_FILE)
    for approach in APPROACHES:
        result = summary[approach]
        print(
            f"approach {approach}: {result['speeches']} speeches, {result['refusals']} refusals,"
            f" {result['ungrounded_sentences']}/{result['sentences']} ungrounded sentences"
        )
    print(f"wrote {config.output_dir / SIMULATIONS_FILE} and {config.output_dir / SUMMARY_FILE}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open generation of simulated speeches (profile, profile + excerpts), with evidence"
            " level and a checked justification, plus the speech from the baseline material."
        )
    )
    add_run_arguments(parser)
    parser.add_argument(
        "--requests",
        type=Path,
        help="JSONL with actor, date (DD/MM/YYYY) and topic; default: one request per actor and"
        " hearing of the requests split with a linked UDV",
    )
    parser.add_argument("--k", type=int, help="excerpts per request (default: evaluation.json)")
    parser.add_argument(
        "--dry-run",
        type=Path,
        help="build the requests, roles and examples without a model and write their counts and"
        " hashes to this JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()
    config = config_from_args(args)
    dry_run = args.dry_run is not None
    require_model(parser, config.model, dry_run)
    k = args.k if args.k is not None else selected_k(config)
    if k is None and not dry_run:
        parser.error("run experiments.actors.evaluate_simulation first, or pass --k")
    inputs = load_inputs(config, args.actors)
    requests = load_requests(config, inputs, args.requests)
    if not requests:
        raise SystemExit("no requests to simulate")
    input_files = {
        "profiles": file_info(config.profiles_path),
        "train_speeches": file_info(config.train_speeches_path),
        "udv": file_info(config.udv_path),
    }
    if dry_run:
        write_report(request_dry_run(config, inputs, input_files, requests, k), args.dry_run)
        return
    assert k is not None
    rows = simulate_requests(config, inputs, input_files, requests, k)
    requests_source = str(args.requests) if args.requests else config.requests_split
    write_summary(config, inputs.prompts, input_files, requests_source, k, rows)


if __name__ == "__main__":
    main()
