"""One LLM-written profile per actor from the actor's train speeches: positions, criteria and
values, declared alignments and way of arguing."""

import argparse
import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import load_jsonl
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from experiments.actors.backend import TransformersChatClient
from experiments.actors.chat import ChatClient
from experiments.actors.cli import MODEL_HELP, configure_logging, require_model, write_report
from experiments.actors.io import (
    canonical_sha256,
    file_info,
    hearing_metadata,
    read_jsonl_rows,
    read_toml,
    warn_invalid_line,
)

Record = dict[str, Any]

DEFAULT_CONFIG = Path("configs/actor_profiles.toml")
PROMPT_VERSION_LENGTH = 12
DRY_RUN_RULE = (
    "rendered_prompts_sha256 is the sha256 of the JSON list of [actor, system prompt, user"
    " prompt] of every selected actor, in input order; the same value means the model"
    " receives the same prompts"
)


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
    seed: int | None


def load_config(path: Path) -> ProfilesConfig:
    raw = read_toml(path)
    prompts = raw["prompts"]
    model = raw["model"]
    return ProfilesConfig(
        speeches_path=Path(raw["input"]["speeches_path"]),
        lds_path=Path(raw["input"]["lds_path"]),
        lds_sha256=raw["input"]["lds_sha256"],
        profiles_path=Path(raw["output"]["profiles_path"]),
        prompts_dir=Path(prompts["dir"]),
        system_profile_file=prompts["system_profile"],
        user_profile_file=prompts["user_profile"],
        model=model["name"],
        device_map=model["device_map"],
        temperature=model["temperature"],
        top_p=model["top_p"],
        max_output_tokens=model["max_output_tokens"],
        seed=model.get("seed"),
    )


@dataclass(frozen=True)
class PromptSet:
    version: str
    system_profile: str
    user_profile: Template


def load_prompts(config: ProfilesConfig) -> PromptSet:
    """The system prompt and the user template, versioned by the hash of their files."""
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
        version=digest.hexdigest()[:PROMPT_VERSION_LENGTH],
        system_profile=(config.prompts_dir / config.system_profile_file).read_text(),
        user_profile=environment.get_template(config.user_profile_file),
    )


def load_hearing_metadata(config: ProfilesConfig) -> dict[int, Record]:
    return hearing_metadata(config.lds_path, config.lds_sha256)


def actor_hearings(record: Record, metadata: dict[int, Record]) -> list[Record]:
    """The actor's hearings in chronological order, with date, subject and turns."""
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


def render_profile_prompt(
    prompts: PromptSet, record: Record, metadata: dict[int, Record]
) -> tuple[str, str]:
    user = prompts.user_profile.render(
        actor_label=record["actor"],
        hearings=actor_hearings(record, metadata),
    )
    return prompts.system_profile, user


def dry_run_report(
    config: ProfilesConfig, prompts: PromptSet, records: list[Record], metadata: dict[int, Record]
) -> Record:
    rendered = [
        [record["actor"], *render_profile_prompt(prompts, record, metadata)] for record in records
    ]
    sizes = [len(user) for _, _, user in rendered]
    return {
        "dry_run": True,
        "model": config.model or None,
        "prompt_version": prompts.version,
        "prompts_dir": str(config.prompts_dir),
        "inputs": {
            "speeches": file_info(config.speeches_path),
            "lds": {"path": str(config.lds_path), "sha256": config.lds_sha256},
        },
        "actors": len(rendered),
        "actors_sha256": canonical_sha256(sorted(record["actor"] for record in records)),
        "rendered_prompts_sha256": canonical_sha256(rendered),
        "user_prompt_chars": {
            "total": sum(sizes),
            "min": min(sizes, default=0),
            "max": max(sizes, default=0),
        },
        "rule": DRY_RUN_RULE,
    }


@dataclass(frozen=True)
class ProfileRunner:
    config: ProfilesConfig
    prompts: PromptSet
    client: ChatClient

    def profile_prompt(self, record: Record, metadata: dict[int, Record]) -> tuple[str, str]:
        return render_profile_prompt(self.prompts, record, metadata)

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
    """Actors already profiled in `path`, which must hold only this prompt version and model."""
    if not path.exists():
        return set()
    done: set[str] = set()
    stale: set[tuple[str, str]] = set()
    for line_number, row in read_jsonl_rows(path):
        try:
            actor, version = row["actor"], (row["prompt_version"], row["model"])
        except KeyError:
            warn_invalid_line(line_number, path)
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


@dataclass
class RunTally:
    """Counts of one profile run, printed at its end."""

    profiles: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[str] = dataclasses.field(default_factory=list)

    def add(self, row: Record) -> None:
        self.profiles += 1
        self.input_tokens += row["input_tokens"]
        self.output_tokens += row["output_tokens"]

    def report(self, skipped: int, elapsed: float) -> None:
        print(
            f"done: {self.profiles} profiles, {len(self.failures)} failures,"
            f" {skipped} skipped (resume)"
        )
        print(
            f"tokens: {self.input_tokens:,} in / {self.output_tokens:,} out,"
            f" elapsed {elapsed / 60:.1f} min"
        )
        if self.failures:
            print(f"failed actors: {self.failures}")


def run(
    runner: ProfileRunner,
    records: list[Record],
    metadata: dict[int, Record],
    done: set[str],
    limit: int | None,
) -> int:
    """Profile the actors not yet in the output file, appending each profile as it is written;
    a failed actor is logged and skipped. Returns the exit code."""
    config = runner.config
    skipped = [record["actor"] for record in records if record["actor"] in done]
    todo = [record for record in records if record["actor"] not in done]
    if skipped:
        logging.info("resuming: %d actors already in %s", len(skipped), config.profiles_path)
    if limit is not None:
        todo = todo[:limit]
    config.profiles_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tally = RunTally()
    with open(config.profiles_path, "a") as output:
        for index, record in enumerate(todo, start=1):
            try:
                row = runner.generate(record, metadata)
            except Exception as error:
                tally.failures.append(record["actor"])
                logging.error("[%d/%d] %s failed: %s", index, len(todo), record["actor"], error)
                continue
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            tally.add(row)
            logging.info(
                "[%d/%d] %s: %s in / %s out tokens, %.1fs",
                index,
                len(todo),
                record["actor"],
                f"{row['input_tokens']:,}",
                f"{row['output_tokens']:,}",
                row["duration_seconds"],
            )
    tally.report(len(skipped), time.monotonic() - started)
    return 1 if tally.failures else 0


def apply_overrides(config: ProfilesConfig, args: argparse.Namespace) -> ProfilesConfig:
    overrides: Record = {}
    if args.input is not None:
        overrides["speeches_path"] = args.input
    if args.output is not None:
        overrides["profiles_path"] = args.output
    if args.model is not None:
        overrides["model"] = args.model
    return dataclasses.replace(config, **overrides) if overrides else config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one LLM-written profile per actor from the per-actor speech file,"
            " with a Hugging Face model loaded through from_pretrained."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, help="per-actor speeches JSONL (overrides config)")
    parser.add_argument("--output", type=Path, help="profiles JSONL (overrides config)")
    parser.add_argument("--model", help=MODEL_HELP)
    parser.add_argument("--actors", nargs="*", help="only these actors, by exact name")
    parser.add_argument("--limit", type=int, help="process at most this many actors this run")
    parser.add_argument(
        "--dry-run",
        type=Path,
        help="render every prompt without a model and write their sizes and hashes to this JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be zero or positive")
    configure_logging()
    config = apply_overrides(load_config(args.config), args)
    require_model(parser, config.model, args.dry_run is not None)
    prompts = load_prompts(config)
    records = select_records(load_jsonl(config.speeches_path), args.actors)
    metadata = load_hearing_metadata(config)
    if args.dry_run is not None:
        write_report(dry_run_report(config, prompts, records, metadata), args.dry_run)
        return
    done = load_done_actors(config.profiles_path, prompts.version, config.model)
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
