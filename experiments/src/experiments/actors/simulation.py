"""Shared core of the actor simulation: configuration, prompts, the material given to the
model, the UDV-to-actor link and the retrieval of train excerpts. Used by
`experiments.actors.evaluate_simulation` and `experiments.actors.simulate`."""

import argparse
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import load_jsonl
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from experiments.actors.build_speeches import CHAIR_ROLE
from experiments.actors.chat import SimulationBackend, system_user_messages
from experiments.actors.cli import MODEL_HELP
from experiments.actors.io import (
    hearing_metadata,
    read_jsonl_rows,
    read_split_hearings,
    read_toml,
)
from experiments.common.transcript import (
    SENTENCE_BOUNDARY_PATTERN,
    STAGE_DIRECTION_PATTERN,
    get_embedding_model,
    is_sentence,
    normalize_whitespace,
    strip_accents,
)

Record = dict[str, Any]
TurnKey = tuple[int, int]

DEFAULT_CONFIG = Path("configs/actor_simulation.toml")
NO_PROFILE = "Nenhum."
PROFILE_BLOCKS = {
    "Posições": "P",
    "Critérios e valores": "C",
    "Alinhamentos declarados": "A",
    "Forma de argumentar": "F",
}
PROFILE_ID_PATTERN = re.compile(r"^([PCAF]\d+)\. ", re.MULTILINE)
LEVELS = ("DIRETA", "INDIRETA", "ESPECULATIVA", "SEM BASE")
NO_BASIS = "SEM BASE"
PROMPT_FILES = {
    "system": "system.md.j2",
    "user": "user.md.j2",
    "choice": "ask_choice.md.j2",
    "evidence": "ask_evidence.md.j2",
    "speech": "ask_speech.md.j2",
    "justification": "ask_justification.md.j2",
}
PROMPT_VERSION_LENGTH = 12
EXCERPT_CONTEXT_SENTENCES = 1


@dataclass(frozen=True)
class SimulationConfig:
    profiles_path: Path
    train_speeches_path: Path
    speeches_path: Path
    udv_path: Path
    manifest_path: Path
    lds_path: Path
    lds_sha256: str
    output_dir: Path
    prompts_dir: Path
    model: str
    device_map: str
    evidence_max_tokens: int
    speech_max_tokens: int
    justification_max_tokens: int
    parties: tuple[str, ...]
    selection_split: str
    eval_split: str
    k_grid: tuple[int, ...]
    guidance_grid: tuple[float, ...]
    bootstrap_samples: int
    eval_seed: int
    requests_split: str
    examples: int
    example_min_words: int
    example_max_words: int
    generation_seed: int


def load_config(path: Path) -> SimulationConfig:
    raw = read_toml(path)
    inputs = raw["input"]
    model = raw["model"]
    evaluation = raw["evaluation"]
    generation = raw["generation"]
    return SimulationConfig(
        profiles_path=Path(inputs["profiles_path"]),
        train_speeches_path=Path(inputs["train_speeches_path"]),
        speeches_path=Path(inputs["speeches_path"]),
        udv_path=Path(inputs["udv_path"]),
        manifest_path=Path(inputs["manifest_path"]),
        lds_path=Path(inputs["lds_path"]),
        lds_sha256=inputs["lds_sha256"],
        output_dir=Path(raw["output"]["dir"]),
        prompts_dir=Path(raw["prompts"]["dir"]),
        model=model["name"],
        device_map=model["device_map"],
        evidence_max_tokens=model["evidence_max_tokens"],
        speech_max_tokens=model["speech_max_tokens"],
        justification_max_tokens=model["justification_max_tokens"],
        parties=tuple(raw["roles"]["parties"]),
        selection_split=evaluation["selection_split"],
        eval_split=evaluation["eval_split"],
        k_grid=tuple(evaluation["k_grid"]),
        guidance_grid=tuple(float(value) for value in evaluation["guidance_grid"]),
        bootstrap_samples=evaluation["bootstrap_samples"],
        eval_seed=evaluation["seed"],
        requests_split=generation["requests_split"],
        examples=generation["examples"],
        example_min_words=generation["example_min_words"],
        example_max_words=generation["example_max_words"],
        generation_seed=generation["seed"],
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The options shared by the evaluation and the open generation."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help=MODEL_HELP)
    parser.add_argument("--actors", nargs="*", help="only these profiled actors, by exact name")
    parser.add_argument("--profiles", type=Path, help="profiles JSONL (overrides config)")


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    config = load_config(args.config)
    if args.model is not None:
        config = replace(config, model=args.model)
    if args.profiles is not None:
        config = replace(config, profiles_path=args.profiles)
    return config


@dataclass(frozen=True)
class SimulationPrompts:
    version: str
    system: Template
    user: Template
    choice: Template
    evidence: Template
    speech: Template
    justification: Template


def load_prompts(prompts_dir: Path) -> SimulationPrompts:
    """The prompt templates, versioned by the hash of their file names and contents."""
    environment = Environment(
        loader=FileSystemLoader(prompts_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    digest = hashlib.sha256()
    for name in sorted(PROMPT_FILES.values()):
        digest.update(name.encode())
        digest.update((prompts_dir / name).read_bytes())
    templates = {key: environment.get_template(name) for key, name in PROMPT_FILES.items()}
    return SimulationPrompts(version=digest.hexdigest()[:PROMPT_VERSION_LENGTH], **templates)


@dataclass(frozen=True)
class Excerpt:
    date: str
    assunto: str
    text: str
    chair: bool
    source: Record


@dataclass(frozen=True)
class Material:
    """What the model knows about the simulated actor in one call."""

    name: str
    role: str | None
    profile: str | None
    examples: tuple[Excerpt, ...] = ()
    excerpts: tuple[Excerpt, ...] = ()

    def baseline(self) -> "Material":
        return Material(name=self.name, role=self.role, profile=None)

    def with_excerpts(self, excerpts: tuple[Excerpt, ...]) -> "Material":
        return replace(self, excerpts=excerpts)


def render(template: Template, **values: Any) -> str:
    return template.render(**values).strip()


def chat_messages(
    prompts: SimulationPrompts, material: Material, date: str, assunto: str, request: str
) -> list[Record]:
    system = render(
        prompts.system,
        name=material.name,
        role=material.role,
        profile=material.profile or NO_PROFILE,
        examples=material.examples,
    )
    user = render(
        prompts.user,
        date=date,
        assunto=assunto,
        name=material.name,
        excerpts=material.excerpts,
        request=request,
    )
    return system_user_messages(system, user)


def ask_evidence_level(
    model: SimulationBackend,
    prompts: SimulationPrompts,
    material: Material,
    date: str,
    assunto: str,
    max_tokens: int,
) -> Record:
    """Ask how much the material supports speaking on this topic, and parse the level."""
    request = render(prompts.evidence, name=material.name)
    messages = chat_messages(prompts, material, date, assunto, request)
    generation = model.generate(messages, max_tokens)
    return {"label": parse_level(generation.text), "text": generation.text}


def number_profile(profile: str) -> str:
    """Number the bullets of each profile block (P1, C1, A1, F1, ...) so they can be cited."""
    lines = []
    prefix: str | None = None
    count = 0
    for line in profile.splitlines():
        if line.startswith("## "):
            prefix = PROFILE_BLOCKS.get(line[3:].strip())
            count = 0
        elif prefix is not None and line.startswith("- "):
            count += 1
            line = f"{prefix}{count}. {line[2:]}"
        lines.append(line)
    return "\n".join(lines).strip()


def profile_ids(profile: str | None) -> set[str]:
    return set(PROFILE_ID_PATTERN.findall(profile or ""))


def clean_role(role: str | None, parties: tuple[str, ...]) -> str | None:
    """The role without a '(PARTY)' or '(PARTY-UF)' parenthesis, so no party reaches a prompt."""
    if not role:
        return None
    names = "|".join(re.escape(party) for party in parties)
    cleaned = re.sub(rf"\s*\((?:{names})(?:-[A-Z]{{2}})?\)", "", role)
    return normalize_whitespace(cleaned) or None


def parse_level(text: str) -> str | None:
    head = re.sub(r"^\W+", "", strip_accents(text).upper())
    return next((level for level in LEVELS if head.startswith(level)), None)


def load_profiles(path: Path, actors: list[str] | None) -> dict[str, Record]:
    """Profiles by actor, restricted to `actors` when given, each of which must have one."""
    profiles = {row["actor"]: row for _, row in read_jsonl_rows(path)}
    if not actors:
        return profiles
    missing = [name for name in actors if name not in profiles]
    if missing:
        raise SystemExit(f"actors without profile in {path}: {missing}")
    return {name: profiles[name] for name in actors}


def split_hearings(config: SimulationConfig, split: str) -> set[int]:
    return set(read_split_hearings(config.manifest_path, config.lds_sha256)[split])


def check_disjoint(
    profiles: dict[str, Record], train_records: list[Record], hearings: set[int]
) -> None:
    """Every profiled actor has train speeches, and no profile or train speech comes from an
    evaluated hearing."""
    missing = sorted(set(profiles) - {record["actor"] for record in train_records})
    if missing:
        raise SystemExit(f"profiled actors missing from the train speeches: {missing}")
    leaked = {actor for actor, row in profiles.items() if set(row["hearing_ids"]) & hearings}
    leaked |= {
        record["actor"]
        for record in train_records
        if {hearing["hearing_id"] for hearing in record["hearings"]} & hearings
    }
    if leaked:
        raise SystemExit(
            f"profiles or train speeches include evaluated hearings for: {sorted(leaked)}"
        )


def turn_owners(records: list[Record]) -> dict[TurnKey, str]:
    """The actor who speaks each (hearing_id, turn_index) of a per-actor speech file."""
    return {
        (hearing["hearing_id"], turn["turn_index"]): record["actor"]
        for record in records
        for hearing in record["hearings"]
        for turn in hearing["turns"]
    }


def udv_owner(udv: Record, owners: dict[TurnKey, str]) -> str | None:
    """The actor who owns the evidence turn of a UDV, if it has one."""
    if not udv["evidence"]:
        return None
    return owners.get((udv["hearing_id"], udv["evidence"]["speaker_turn"]))


def linked_udvs(
    udvs: list[Record],
    owners: dict[TurnKey, str],
    actors: set[str],
    hearings: set[int],
) -> list[tuple[Record, str]]:
    """The UDVs of `hearings` whose evidence turn belongs to one of `actors`, with that actor."""
    linked = []
    for udv in udvs:
        if udv["hearing_id"] not in hearings:
            continue
        actor = udv_owner(udv, owners)
        if actor in actors:
            linked.append((udv, actor))
    return linked


def turn_sentences(text: str) -> list[str]:
    parts = (normalize_whitespace(part) for part in SENTENCE_BOUNDARY_PATTERN.split(text))
    return [part for part in parts if part and not STAGE_DIRECTION_PATTERN.match(part)]


@dataclass(frozen=True)
class SimulationInputs:
    """The files read by both the evaluation and the open generation."""

    prompts: SimulationPrompts
    profiles: dict[str, Record]
    train_records: list[Record]
    metadata: dict[int, Record]
    udvs: list[Record]
    owners: dict[TurnKey, str]


def load_inputs(config: SimulationConfig, actors: list[str] | None) -> SimulationInputs:
    prompts = load_prompts(config.prompts_dir)
    profiles = load_profiles(config.profiles_path, actors)
    return SimulationInputs(
        prompts=prompts,
        profiles=profiles,
        train_records=[
            record
            for record in load_jsonl(config.train_speeches_path)
            if record["actor"] in profiles
        ],
        metadata=hearing_metadata(config.lds_path, config.lds_sha256),
        udvs=load_jsonl(config.udv_path),
        owners=turn_owners(load_jsonl(config.speeches_path)),
    )


def sentence_units(record: Record) -> list[Record]:
    """One retrieval unit per sentence of the actor's turns, with the turn's sentences around it."""
    units = []
    for hearing in record["hearings"]:
        for turn in hearing["turns"]:
            sentences = turn_sentences(turn["text"])
            for position, sentence in enumerate(sentences):
                if is_sentence(sentence):
                    units.append(
                        {
                            "hearing_id": hearing["hearing_id"],
                            "turn_index": turn["turn_index"],
                            "chair": turn["role"] == CHAIR_ROLE,
                            "sentences": sentences,
                            "position": position,
                        }
                    )
    return units


class SpeechRetriever:
    """Retrieves, for an actor and a topic, the k train sentences closest to the topic, at most
    one per turn, each with its neighbour sentences, in chronological order."""

    def __init__(self, records: list[Record], metadata: dict[int, Record]) -> None:
        self._records = {record["actor"]: record for record in records}
        self._metadata = metadata
        self._encoder = get_embedding_model()
        self._units: dict[str, list[Record]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._queries: dict[str, np.ndarray] = {}

    def _index(self, actor: str) -> tuple[list[Record], np.ndarray]:
        if actor not in self._units:
            units = sentence_units(self._records[actor])
            self._units[actor] = units
            self._embeddings[actor] = self._encode(
                [unit["sentences"][unit["position"]] for unit in units]
            )
        return self._units[actor], self._embeddings[actor]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self._encoder.encode(texts, normalize_embeddings=True))

    def _query(self, text: str) -> np.ndarray:
        if text not in self._queries:
            self._queries[text] = self._encode([text])[0]
        return self._queries[text]

    def retrieve(self, actor: str, query: str, k: int) -> tuple[Excerpt, ...]:
        units, embeddings = self._index(actor)
        if not units or k <= 0:
            return ()
        scores = embeddings @ self._query(query)
        chosen = best_unit_per_turn(units, scores, k)
        chosen.sort(key=lambda item: self._chronological_key(item[0]))
        return tuple(self._excerpt(unit, score) for unit, score in chosen)

    def _chronological_key(self, unit: Record) -> tuple[Any, ...]:
        return (
            self._metadata[unit["hearing_id"]]["date"],
            unit["hearing_id"],
            unit["turn_index"],
            unit["position"],
        )

    def _excerpt(self, unit: Record, score: float) -> Excerpt:
        position = unit["position"]
        hearing = self._metadata[unit["hearing_id"]]
        start = max(position - EXCERPT_CONTEXT_SENTENCES, 0)
        end = position + EXCERPT_CONTEXT_SENTENCES + 1
        return Excerpt(
            date=hearing["date_br"],
            assunto=hearing["assunto"],
            text=" ".join(unit["sentences"][start:end]),
            chair=unit["chair"],
            source={
                "hearing_id": unit["hearing_id"],
                "turn_index": unit["turn_index"],
                "sentence_index": position,
                "cosine": round(score, 4),
            },
        )


def best_unit_per_turn(
    units: list[Record], scores: np.ndarray, k: int
) -> list[tuple[Record, float]]:
    """The k best-scored units, skipping any unit whose turn already has a better one."""
    chosen: list[tuple[Record, float]] = []
    used_turns: set[TurnKey] = set()
    for row in np.argsort(-scores, kind="stable"):
        unit = units[row]
        turn = (unit["hearing_id"], unit["turn_index"])
        if turn in used_turns:
            continue
        used_turns.add(turn)
        chosen.append((unit, float(scores[row])))
        if len(chosen) == k:
            break
    return chosen
