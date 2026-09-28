"""Run the actor profile and simulation stages of experiments.actors with an MLX model: the
stages are called unchanged, with the MLX classes installed in place of the transformers ones,
and the time of each stage is appended to the run's timings file."""

import argparse
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx

import experiments.actors.evaluate_simulation as evaluation_module
import experiments.actors.generate_profiles as profiles_module
import experiments.actors.simulate as simulation_module
from experiments.actors.chat import LETTERS, letter_token_ids
from experiments.actors.io import load_rows
from experiments.mlx.backend import MLXChatClient, MLXSimulationModel, engine_for
from experiments.mlx.probe import (
    PT_BR_QUESTION,
    SMOKE_OPTIONS,
    choice_messages,
    probe_material,
)
from experiments.mlx.settings import (
    DEFAULT_SETTINGS,
    ModelSpec,
    RunPaths,
    activate_model,
    append_timing,
    call_main,
    download_model,
    load_settings,
    model_spec,
    prepare_speeches,
    run_paths,
    train_speeches_path,
    write_derived_configs,
    write_json_file,
)

Record = dict[str, Any]

STAGES = ("download", "smoke", "profiles", "evaluate", "simulate", "all")
GB = 1e9
PROMPT_TAIL_TOKENS = 16
CHOICE_GREEDY_TOKENS = 8
PT_BR_SAMPLE_TOKENS = 96


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def elapsed(started: float) -> float:
    return round(time.monotonic() - started, 1)


def install_backend(spec: ModelSpec) -> None:
    """Replace the transformers classes of the actor modules by the MLX ones."""
    profiles_module.TransformersChatClient = partial(  # type: ignore[misc,assignment]
        MLXChatClient, options=spec.options
    )
    evaluation_module.SimulationModel = partial(  # type: ignore[misc,assignment]
        MLXSimulationModel, options=spec.options
    )
    simulation_module.SimulationModel = partial(  # type: ignore[misc,assignment]
        MLXSimulationModel, options=spec.options
    )


def row_fingerprints(path: Path, key: str) -> dict[str, str]:
    """The fingerprint of each row of a resumable JSONL file, by row id."""
    return {name: row["fingerprint"] for name, row in load_rows(path, key).items()}


def install_timed_score_split(spec: ModelSpec, paths: RunPaths) -> None:
    """Wrap score_split so that each split's scoring time is appended to the timings."""
    original = evaluation_module.score_split

    def timed(*args: Any, **kwargs: Any) -> list[Record]:
        path: Path = args[-1] if args else kwargs["path"]
        before = load_rows(path, "udv_id")
        started = time.monotonic()
        rows = original(*args, **kwargs)
        scored = sum(
            before.get(row["udv_id"], {}).get("fingerprint") != row["fingerprint"] for row in rows
        )
        append_timing(
            paths,
            {
                "stage": "evaluate",
                "split": path.stem.removeprefix("choice_"),
                "model": spec.id,
                "questions": len(rows),
                "scored": scored,
                "seconds": elapsed(started),
                "finished_at": now(),
            },
        )
        return rows

    evaluation_module.score_split = timed


def actor_args(actors: list[str] | None) -> list[str]:
    return ["--actors", *actors] if actors else []


def letter_probe(spec: ModelSpec, messages: list[Record]) -> Record:
    """Letter token ids, letter log-probabilities and the greedy answer to the probe choice."""
    model = MLXSimulationModel(spec.repo, "auto", spec.options)
    logprobs = model.letter_logprobs(messages)
    return {
        "letter_ids": dict(zip(LETTERS, model.letter_ids, strict=True)),
        "letter_logprobs": dict(zip(LETTERS, logprobs, strict=True)),
        "letter_mass": float(sum(mx.exp(mx.array(logprobs)).tolist())),
        "choice_greedy": model.generate(messages, CHOICE_GREEDY_TOKENS).text,
    }


def smoke(spec: ModelSpec, paths: RunPaths) -> Record:
    """Load the model, check the letter tokens and the chat template on a probe choice, and
    time a short pt-BR answer."""
    mx.reset_peak_memory()
    started = time.monotonic()
    engine = engine_for(spec.repo, spec.options)
    load_seconds = time.monotonic() - started
    messages = choice_messages(probe_material(None), list(range(len(SMOKE_OPTIONS))))
    tokens = engine.encode(messages)
    report: Record = {
        "model": spec.id,
        "repo": spec.repo,
        "revision": spec.revision,
        "load_seconds": round(load_seconds, 1),
        "template_kwargs": dict(spec.options.template_kwargs),
        "prompt_tail": engine.tokenizer.decode(tokens[-PROMPT_TAIL_TOKENS:]),
    }
    try:
        letter_token_ids(engine.tokenizer)
    except SystemExit as error:
        report["letter_ids"] = None
        report["letter_error"] = str(error)
    else:
        report |= letter_probe(spec, messages)
    started = time.monotonic()
    generated = engine.generate_tokens(engine.encode(PT_BR_QUESTION), PT_BR_SAMPLE_TOKENS)
    seconds = time.monotonic() - started
    report["pt_br_sample"] = engine.decode(generated)
    report["pt_br_sample_tokens_per_second"] = round(len(generated) / seconds, 1)
    report["contains_think"] = "<think>" in report["pt_br_sample"]
    report["peak_memory_gb"] = round(mx.get_peak_memory() / GB, 1)
    write_json_file(report, paths.root / "smoke.json")
    return report


def run_profiles(spec: ModelSpec, paths: RunPaths, actors: list[str] | None) -> None:
    prepare_speeches(paths)
    started = time.monotonic()
    call_main(
        "experiments.actors.generate_profiles",
        profiles_module.main,
        [
            "--config",
            str(paths.profiles_config),
            "--input",
            str(train_speeches_path(paths)),
            *actor_args(actors),
        ],
    )
    append_timing(
        paths,
        {
            "stage": "profiles",
            "model": spec.id,
            "actors": actors,
            "seconds": elapsed(started),
            "finished_at": now(),
        },
    )


def run_evaluate(spec: ModelSpec, paths: RunPaths, actors: list[str] | None) -> None:
    install_timed_score_split(spec, paths)
    call_main(
        "experiments.actors.evaluate_simulation",
        evaluation_module.main,
        ["--config", str(paths.simulation_config), *actor_args(actors)],
    )


def run_simulate(
    spec: ModelSpec, paths: RunPaths, actors: list[str] | None, extra: list[str]
) -> None:
    output = paths.simulation_dir / "simulations.jsonl"
    before = row_fingerprints(output, "request_id")
    started = time.monotonic()
    call_main(
        "experiments.actors.simulate",
        simulation_module.main,
        ["--config", str(paths.simulation_config), *actor_args(actors), *extra],
    )
    after = row_fingerprints(output, "request_id")
    append_timing(
        paths,
        {
            "stage": "simulate",
            "model": spec.id,
            "requests": len(after),
            "generated": sum(before.get(key) != value for key, value in after.items()),
            "seconds": elapsed(started),
            "finished_at": now(),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the actor profile and simulation pipeline of experiments.actors unchanged, with an"
            " MLX model in place of the transformers backend."
        )
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--model", required=True, help="model id from the settings file")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--actors", nargs="+", help="actors, by exact name (default: short run)")
    group.add_argument("--all-actors", action="store_true", help="every profiled actor")
    parser.add_argument("--k", help="passed to experiments.actors.simulate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.settings)
    spec = model_spec(settings, args.model)
    if args.stage == "download":
        print(f"downloaded {spec.repo} to {download_model(settings, spec)}")
        return
    activate_model(spec)
    paths = run_paths(settings, spec)
    write_derived_configs(spec, paths)
    install_backend(spec)
    actors = None if args.all_actors else args.actors or list(settings.short_run_actors) or None
    extra = ["--k", args.k] if args.k is not None else []
    if args.stage == "smoke":
        print(json.dumps(smoke(spec, paths), ensure_ascii=False, indent=2))
        return
    if args.stage in ("profiles", "all"):
        run_profiles(spec, paths, actors)
    if args.stage in ("evaluate", "all"):
        run_evaluate(spec, paths, actors)
    if args.stage in ("simulate", "all"):
        run_simulate(spec, paths, actors, extra)


if __name__ == "__main__":
    main()
