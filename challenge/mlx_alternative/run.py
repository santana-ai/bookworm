import argparse
import json
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx

import utils.evaluate_actor_simulation as evaluation_module
import utils.generate_actor_profiles as profiles_module
import utils.simulate_actors as simulation_module
from mlx_alternative.backend import MLXChatClient, MLXSimulationModel, engine_for
from mlx_alternative.settings import (
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
)
from utils.actor_simulation import (
    LETTERS,
    Material,
    chat_messages,
    letter_token_ids,
    load_prompts,
    load_rows,
    render,
)

Record = dict[str, Any]

STAGES = ("download", "smoke", "profiles", "evaluate", "simulate", "all")
SMOKE_OPTIONS = (
    "Defendeu a ampliação do financiamento público para a educação básica.",
    "Criticou a demora do ministério na liberação de recursos para os municípios.",
    "Propôs a criação de uma comissão especial para acompanhar o programa.",
    "Afirmou que a proposta transfere custos para os estados sem compensação.",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def install_backend(spec: ModelSpec) -> None:
    profiles_module.TransformersChatClient = partial(MLXChatClient, options=spec.options)
    evaluation_module.SimulationModel = partial(MLXSimulationModel, options=spec.options)
    simulation_module.SimulationModel = partial(MLXSimulationModel, options=spec.options)


def install_timed_score_split(spec: ModelSpec, paths: RunPaths) -> None:
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
                "seconds": round(time.monotonic() - started, 1),
                "finished_at": now(),
            },
        )
        return rows

    evaluation_module.score_split = timed


def actor_args(actors: list[str] | None) -> list[str]:
    return ["--actors", *actors] if actors else []


def smoke(spec: ModelSpec, paths: RunPaths) -> Record:
    mx.reset_peak_memory()
    started = time.monotonic()
    engine = engine_for(spec.repo, spec.options)
    load_seconds = time.monotonic() - started
    prompts = load_prompts(Path("prompts/actor_simulation"))
    material = Material(name="Fulano de Tal", role="Deputado", profile=None)
    choice = render(prompts.choice, name=material.name, options=SMOKE_OPTIONS, letters=LETTERS)
    messages = chat_messages(prompts, material, "05/03/2024", "Financiamento da educação", choice)
    tokens = engine.encode(messages)
    report: Record = {
        "model": spec.id,
        "repo": spec.repo,
        "load_seconds": round(load_seconds, 1),
        "template_kwargs": dict(spec.options.template_kwargs),
        "prompt_tail": engine.tokenizer.decode(tokens[-16:]),
    }
    try:
        letter_ids = letter_token_ids(engine.tokenizer)
    except SystemExit as error:
        report["letter_ids"] = None
        report["letter_error"] = str(error)
    else:
        report["letter_ids"] = dict(zip(LETTERS, letter_ids, strict=True))
        model = MLXSimulationModel(spec.repo, "auto", spec.options)
        logprobs = model.letter_logprobs(messages)
        report["letter_logprobs"] = dict(zip(LETTERS, logprobs, strict=True))
        report["letter_mass"] = float(sum(mx.exp(mx.array(logprobs)).tolist()))
        report["choice_greedy"] = model.generate(messages, 8).text
    question = [
        {"role": "user", "content": "Em duas frases, o que é uma audiência pública na Câmara?"}
    ]
    started = time.monotonic()
    generated = engine.generate_tokens(engine.encode(question), 96)
    seconds = time.monotonic() - started
    report["pt_br_sample"] = engine.decode(generated)
    report["pt_br_sample_tokens_per_second"] = round(len(generated) / seconds, 1)
    report["contains_think"] = "<think>" in report["pt_br_sample"]
    report["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 1)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def count_rows(path: Path, key: str) -> dict[str, str]:
    return {name: row["fingerprint"] for name, row in load_rows(path, key).items()}


def run_profiles(spec: ModelSpec, paths: RunPaths, actors: list[str] | None) -> None:
    prepare_speeches(paths)
    started = time.monotonic()
    call_main(
        "utils.generate_actor_profiles",
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
            "seconds": round(time.monotonic() - started, 1),
            "finished_at": now(),
        },
    )


def run_evaluate(spec: ModelSpec, paths: RunPaths, actors: list[str] | None) -> None:
    install_timed_score_split(spec, paths)
    call_main(
        "utils.evaluate_actor_simulation",
        evaluation_module.main,
        ["--config", str(paths.simulation_config), *actor_args(actors)],
    )


def run_simulate(
    spec: ModelSpec, paths: RunPaths, actors: list[str] | None, extra: list[str]
) -> None:
    output = paths.simulation_dir / "simulations.jsonl"
    before = count_rows(output, "request_id")
    started = time.monotonic()
    call_main(
        "utils.simulate_actors",
        simulation_module.main,
        ["--config", str(paths.simulation_config), *actor_args(actors), *extra],
    )
    after = count_rows(output, "request_id")
    append_timing(
        paths,
        {
            "stage": "simulate",
            "model": spec.id,
            "requests": len(after),
            "generated": sum(before.get(key) != value for key, value in after.items()),
            "seconds": round(time.monotonic() - started, 1),
            "finished_at": now(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the actor profile and simulation pipeline of utils/ unchanged, with an MLX model"
            " in place of the transformers backend."
        )
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--model", required=True, help="model id from the settings file")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--actors", nargs="+", help="actors, by exact name (default: short run)")
    group.add_argument("--all-actors", action="store_true", help="every profiled actor")
    parser.add_argument("--k", help="passed to utils.simulate_actors")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    spec = model_spec(settings, args.model)
    if args.stage == "download":
        print(f"downloaded {spec.repo} to {download_model(settings, spec)}")
        return
    activate_model(spec)
    paths = run_paths(settings, spec)
    write_derived_configs(spec, paths)
    install_backend(spec)
    if args.all_actors:
        actors = None
    else:
        actors = args.actors or list(settings.short_run_actors) or None
    extra = []
    if args.k is not None:
        extra += ["--k", args.k]
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
