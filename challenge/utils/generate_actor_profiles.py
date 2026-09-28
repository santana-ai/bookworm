import argparse
import dataclasses
import hashlib
import json
import logging
import re
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from utils.dataset_io import load_gated_jsonl
from utils.hearing_dates import article_date

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/actor_profiles.toml")
THINK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass(frozen=True)
class ProfilesConfig:
    speeches_path: Path
    lds_path: Path
    lds_sha256: str
    profiles_path: Path
    prompts_dir: Path
    system_profile_file: str
    user_profile_file: str
    model: str
    device_map: str
    temperature: float
    top_p: float
    max_output_tokens: int
    prefill_chunk_size: int | None
    seed: int | None


def load_config(path: Path) -> ProfilesConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return ProfilesConfig(
        speeches_path=Path(raw["input"]["speeches_path"]),
        lds_path=Path(raw["input"]["lds_path"]),
        lds_sha256=raw["input"]["lds_sha256"],
        profiles_path=Path(raw["output"]["profiles_path"]),
        prompts_dir=Path(raw["prompts"]["dir"]),
        system_profile_file=raw["prompts"]["system_profile"],
        user_profile_file=raw["prompts"]["user_profile"],
        model=raw["model"]["name"],
        device_map=raw["model"]["device_map"],
        temperature=raw["model"]["temperature"],
        top_p=raw["model"]["top_p"],
        max_output_tokens=raw["model"]["max_output_tokens"],
        prefill_chunk_size=raw["model"].get("prefill_chunk_size"),
        seed=raw["model"].get("seed"),
    )


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int


class ChatClient(Protocol):
    def chat(self, system: str, user: str) -> ChatResult: ...


class TransformersChatClient:
    def __init__(self, config: ProfilesConfig) -> None:
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model, device_map=config.device_map
        )

    def chat(self, system: str, user: str) -> ChatResult:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)
        if self._config.seed is not None:
            set_seed(self._config.seed)
        output = self._model.generate(
            **inputs,
            do_sample=True,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_new_tokens=self._config.max_output_tokens,
            prefill_chunk_size=self._config.prefill_chunk_size,
        )
        input_tokens = inputs["input_ids"].shape[1]
        generated = output[0, input_tokens:]
        if len(generated) >= self._config.max_output_tokens:
            raise RuntimeError(
                f"generation reached max_output_tokens={self._config.max_output_tokens}"
            )
        content = self._tokenizer.decode(generated, skip_special_tokens=True)
        if "<think>" in content and "</think>" not in content:
            raise RuntimeError("response contains an unterminated think block")
        return ChatResult(
            text=THINK_PATTERN.sub("", content).strip(),
            input_tokens=input_tokens,
            output_tokens=len(generated),
        )


@dataclass(frozen=True)
class PromptSet:
    version: str
    system_profile: str
    user_profile: Template


def load_prompts(config: ProfilesConfig) -> PromptSet:
    environment = Environment(
        loader=FileSystemLoader(config.prompts_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    digest = hashlib.sha256()
    for name in sorted((config.system_profile_file, config.user_profile_file)):
        digest.update(name.encode())
        digest.update((config.prompts_dir / name).read_bytes())
    return PromptSet(
        version=digest.hexdigest()[:12],
        system_profile=(config.prompts_dir / config.system_profile_file).read_text(),
        user_profile=environment.get_template(config.user_profile_file),
    )


def load_hearing_metadata(config: ProfilesConfig) -> dict[int, Record]:
    return hearing_metadata(config.lds_path, config.lds_sha256)


def hearing_metadata(lds_path: Path, lds_sha256: str) -> dict[int, Record]:
    metadata: dict[int, Record] = {}
    for hearing in load_gated_jsonl(lds_path, lds_sha256):
        published = article_date(hearing["materia"])
        if published is None:
            raise ValueError(f"hearing {hearing['id']} has no article date")
        metadata[hearing["id"]] = {
            "date": published,
            "date_br": published.strftime("%d/%m/%Y"),
            "assunto": hearing["metadados"]["assunto"],
        }
    return metadata


def actor_hearings(record: Record, metadata: dict[int, Record]) -> list[Record]:
    hearings = sorted(
        record["hearings"],
        key=lambda hearing: (metadata[hearing["hearing_id"]]["date"], hearing["hearing_id"]),
    )
    return [
        {
            "date": metadata[hearing["hearing_id"]]["date_br"],
            "assunto": metadata[hearing["hearing_id"]]["assunto"],
            "turns": [{"role": turn["role"], "text": turn["text"]} for turn in hearing["turns"]],
        }
        for hearing in hearings
    ]


@dataclass(frozen=True)
class ProfileRunner:
    config: ProfilesConfig
    prompts: PromptSet
    client: ChatClient

    def profile_prompt(self, record: Record, metadata: dict[int, Record]) -> tuple[str, str]:
        user = self.prompts.user_profile.render(
            actor_label=record["actor"],
            hearings=actor_hearings(record, metadata),
        )
        return self.prompts.system_profile, user

    def generate(self, record: Record, metadata: dict[int, Record]) -> Record:
        started = time.monotonic()
        result = self.client.chat(*self.profile_prompt(record, metadata))
        if not result.text:
            raise ValueError(f"empty profile for {record['actor']}")
        return {
            "actor": record["actor"],
            "profile": result.text,
            "model": self.config.model,
            "prompt_version": self.prompts.version,
            "n_statements": sum(len(hearing["turns"]) for hearing in record["hearings"]),
            "n_hearings": len(record["hearings"]),
            "hearing_ids": sorted(hearing["hearing_id"] for hearing in record["hearings"]),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started, 1),
        }


def load_done_actors(path: Path, prompt_version: str, model: str) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    stale: set[tuple[str, str]] = set()
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                actor, version = row["actor"], (row["prompt_version"], row["model"])
            except (json.JSONDecodeError, KeyError):
                logging.warning("ignoring invalid line %d in %s", line_number, path)
                continue
            if version != (prompt_version, model):
                stale.add(version)
            done.add(actor)
    if stale:
        raise SystemExit(
            f"{path} has profiles from other (prompt_version, model) pairs {sorted(stale)},"
            f" current run is {(prompt_version, model)}; move the file away to regenerate"
            " every profile, or pass a new --output"
        )
    return done


def select_records(records: list[Record], actors: list[str] | None) -> list[Record]:
    if not actors:
        return records
    by_name = {record["actor"]: record for record in records}
    missing = [name for name in actors if name not in by_name]
    if missing:
        raise SystemExit(f"actors not found in input: {missing}")
    return [by_name[name] for name in actors]


def run(
    runner: ProfileRunner,
    records: list[Record],
    metadata: dict[int, Record],
    done: set[str],
    limit: int | None,
) -> int:
    config = runner.config
    skipped = [record["actor"] for record in records if record["actor"] in done]
    todo = [record for record in records if record["actor"] not in done]
    if skipped:
        logging.info("resuming: %d actors already in %s", len(skipped), config.profiles_path)
    if limit is not None:
        todo = todo[:limit]
    config.profiles_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures: list[str] = []
    total_input = 0
    total_output = 0
    successes = 0
    with open(config.profiles_path, "a") as output:
        for index, record in enumerate(todo, start=1):
            try:
                row = runner.generate(record, metadata)
            except Exception as error:
                failures.append(record["actor"])
                logging.error("[%d/%d] %s failed: %s", index, len(todo), record["actor"], error)
                continue
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            successes += 1
            total_input += row["input_tokens"]
            total_output += row["output_tokens"]
            logging.info(
                "[%d/%d] %s: %s in / %s out tokens, %.1fs",
                index,
                len(todo),
                record["actor"],
                f"{row['input_tokens']:,}",
                f"{row['output_tokens']:,}",
                row["duration_seconds"],
            )
    elapsed = time.monotonic() - started
    print(f"done: {successes} profiles, {len(failures)} failures, {len(skipped)} skipped (resume)")
    print(f"tokens: {total_input:,} in / {total_output:,} out, elapsed {elapsed / 60:.1f} min")
    if failures:
        print(f"failed actors: {failures}")
    return 1 if failures else 0


def apply_overrides(config: ProfilesConfig, args: argparse.Namespace) -> ProfilesConfig:
    overrides: Record = {}
    if args.input is not None:
        overrides["speeches_path"] = args.input
    if args.output is not None:
        overrides["profiles_path"] = args.output
    if args.model is not None:
        overrides["model"] = args.model
    return dataclasses.replace(config, **overrides) if overrides else config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one LLM-written profile per actor from the per-actor speech file,"
            " with a Hugging Face model loaded through from_pretrained."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="per-actor speeches JSONL (overrides config)")
    parser.add_argument("--output", type=Path, help="profiles JSONL (overrides config)")
    parser.add_argument("--model", help="Hugging Face model id or local path (overrides config)")
    parser.add_argument("--actors", nargs="*", help="only these actors, by exact name")
    parser.add_argument("--limit", type=int, help="process at most this many actors this run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = apply_overrides(load_config(args.config), args)
    if not config.model:
        parser.error("set [model] name in the config, or pass --model")
    prompts = load_prompts(config)
    done = load_done_actors(config.profiles_path, prompts.version, config.model)
    with open(config.speeches_path) as f:
        records = [json.loads(line) for line in f]
    records = select_records(records, args.actors)
    metadata = load_hearing_metadata(config)
    logging.info(
        "%d actors selected, prompts %s, loading %s", len(records), prompts.version, config.model
    )
    runner = ProfileRunner(
        config=config,
        prompts=prompts,
        client=TransformersChatClient(config),
    )
    raise SystemExit(run(runner, records, metadata, done, args.limit))


if __name__ == "__main__":
    main()
