import hashlib
import json
import logging
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataset_io import sha256_of_file
from utils.udv_pipeline import (
    SENTENCE_BOUNDARY_PATTERN,
    STAGE_DIRECTION_PATTERN,
    get_embedding_model,
    is_sentence,
    normalize_whitespace,
    strip_accents,
)

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/actor_simulation.toml")
LETTERS = ("A", "B", "C", "D")
NO_PROFILE = "Nenhum."
CHAIR_ROLE = "chair"
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
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    evaluation = raw["evaluation"]
    generation = raw["generation"]
    return SimulationConfig(
        profiles_path=Path(raw["input"]["profiles_path"]),
        train_speeches_path=Path(raw["input"]["train_speeches_path"]),
        speeches_path=Path(raw["input"]["speeches_path"]),
        udv_path=Path(raw["input"]["udv_path"]),
        manifest_path=Path(raw["input"]["manifest_path"]),
        lds_path=Path(raw["input"]["lds_path"]),
        lds_sha256=raw["input"]["lds_sha256"],
        output_dir=Path(raw["output"]["dir"]),
        prompts_dir=Path(raw["prompts"]["dir"]),
        model=raw["model"]["name"],
        device_map=raw["model"]["device_map"],
        evidence_max_tokens=raw["model"]["evidence_max_tokens"],
        speech_max_tokens=raw["model"]["speech_max_tokens"],
        justification_max_tokens=raw["model"]["justification_max_tokens"],
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
    return SimulationPrompts(version=digest.hexdigest()[:12], **templates)


@dataclass(frozen=True)
class Excerpt:
    date: str
    assunto: str
    text: str
    chair: bool
    source: Record


@dataclass(frozen=True)
class Material:
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
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def number_profile(profile: str) -> str:
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
    if not role:
        return None
    names = "|".join(re.escape(party) for party in parties)
    cleaned = re.sub(rf"\s*\((?:{names})(?:-[A-Z]{{2}})?\)", "", role)
    return normalize_whitespace(cleaned) or None


def parse_level(text: str) -> str | None:
    head = re.sub(r"^\W+", "", strip_accents(text).upper())
    return next((level for level in LEVELS if head.startswith(level)), None)


def load_profiles(path: Path, actors: list[str] | None) -> dict[str, Record]:
    profiles: dict[str, Record] = {}
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("ignoring invalid line %d in %s", line_number, path)
                continue
            profiles[row["actor"]] = row
    if not actors:
        return profiles
    missing = [name for name in actors if name not in profiles]
    if missing:
        raise SystemExit(f"actors without profile in {path}: {missing}")
    return {name: profiles[name] for name in actors}


def split_hearings(config: SimulationConfig, split: str) -> set[int]:
    with open(config.manifest_path) as f:
        manifest = json.load(f)
    if manifest["dataset"]["sha256"] != config.lds_sha256:
        raise SystemExit(f"{config.manifest_path} was built from another LDS file")
    return set(manifest[split])


def check_disjoint(
    profiles: dict[str, Record], train_records: list[Record], hearings: set[int]
) -> None:
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


def turn_owners(records: list[Record]) -> dict[tuple[int, int], str]:
    return {
        (hearing["hearing_id"], turn["turn_index"]): record["actor"]
        for record in records
        for hearing in record["hearings"]
        for turn in hearing["turns"]
    }


def udv_owner(udv: Record, owners: dict[tuple[int, int], str]) -> str | None:
    if not udv["evidence"]:
        return None
    return owners.get((udv["hearing_id"], udv["evidence"]["speaker_turn"]))


def linked_udvs(
    udvs: list[Record],
    owners: dict[tuple[int, int], str],
    actors: set[str],
    hearings: set[int],
) -> list[tuple[Record, str]]:
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


def file_info(path: Path) -> Record:
    return {"path": str(path), "sha256": sha256_of_file(path)}


def fingerprint(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_rows(path: Path, key: str) -> dict[str, Record]:
    if not path.exists():
        return {}
    rows: dict[str, Record] = {}
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("ignoring invalid line %d in %s", line_number, path)
                continue
            rows[row[key]] = row
    return rows


class SpeechRetriever:
    def __init__(self, records: list[Record], metadata: dict[int, Record]) -> None:
        self._records = {record["actor"]: record for record in records}
        self._metadata = metadata
        self._encoder = get_embedding_model()
        self._units: dict[str, list[Record]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._queries: dict[str, np.ndarray] = {}

    def _index(self, actor: str) -> tuple[list[Record], np.ndarray]:
        if actor not in self._units:
            units = []
            for hearing in self._records[actor]["hearings"]:
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
            texts = [unit["sentences"][unit["position"]] for unit in units]
            self._units[actor] = units
            self._embeddings[actor] = self._encode(texts)
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
        chosen: list[tuple[Record, float]] = []
        used_turns: set[tuple[int, int]] = set()
        for row in np.argsort(-scores, kind="stable"):
            unit = units[row]
            turn = (unit["hearing_id"], unit["turn_index"])
            if turn in used_turns:
                continue
            used_turns.add(turn)
            chosen.append((unit, float(scores[row])))
            if len(chosen) == k:
                break
        chosen.sort(
            key=lambda item: (
                self._metadata[item[0]["hearing_id"]]["date"],
                item[0]["hearing_id"],
                item[0]["turn_index"],
                item[0]["position"],
            )
        )
        return tuple(self._excerpt(unit, score) for unit, score in chosen)

    def _excerpt(self, unit: Record, score: float) -> Excerpt:
        position = unit["position"]
        hearing = self._metadata[unit["hearing_id"]]
        return Excerpt(
            date=hearing["date_br"],
            assunto=hearing["assunto"],
            text=" ".join(unit["sentences"][max(position - 1, 0) : position + 2]),
            chair=unit["chair"],
            source={
                "hearing_id": unit["hearing_id"],
                "turn_index": unit["turn_index"],
                "sentence_index": position,
                "cosine": round(score, 4),
            },
        )


def letter_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded).strip() != letter:
            raise SystemExit(
                f"letter {letter!r} is not a single token in this tokenizer: {encoded}"
            )
        ids.append(encoded[0])
    return ids


@dataclass(frozen=True)
class Generation:
    text: str
    output_tokens: int
    truncated: bool


class SimulationModel:
    def __init__(self, name: str, device_map: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map)
        self.letter_ids = letter_token_ids(self.tokenizer)

    def _encode(self, messages: list[Record]) -> Any:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

    @torch.inference_mode()
    def letter_logprobs(self, messages: list[Record]) -> list[float]:
        logits = self.model(**self._encode(messages), logits_to_keep=1).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)[self.letter_ids].tolist()

    @torch.inference_mode()
    def generate(self, messages: list[Record], max_new_tokens: int) -> Generation:
        inputs = self._encode(messages)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return Generation(
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            output_tokens=len(generated),
            truncated=len(generated) >= max_new_tokens,
        )
