import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx

import utils.evaluate_actor_simulation as evaluation_module
import utils.generate_actor_profiles as profiles_module
import utils.simulate_actors as simulation_module
from mlx_alternative.backend import MLXEngine, engine_for, release_engines
from mlx_alternative.settings import (
    DEFAULT_SETTINGS,
    ModelSpec,
    RunPaths,
    Settings,
    activate_model,
    load_settings,
    load_timings,
    model_spec,
    prepare_speeches,
    run_paths,
    train_speeches_path,
    write_derived_configs,
)
from mlx_alternative.verify_backend import PROFILE, choice_messages
from utils.actor_simulation import LETTERS, Material, load_config, turn_owners
from utils.dataset_io import load_jsonl
from utils.evaluate_actor_simulation import rotations

Record = dict[str, Any]

GB = 1e9


def profile_messages(system: str, user: str) -> list[Record]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def timed_prefill(engine: MLXEngine, tokens: list[int]) -> tuple[float, mx.array, list[Any]]:
    engine.reset_prefix()
    started = time.monotonic()
    logits, cache = engine.prefill(tokens)
    mx.eval(logits)
    return time.monotonic() - started, logits, cache


def timed_generation(engine: MLXEngine, logits: mx.array, cache: list[Any], steps: int) -> float:
    token = int(mx.argmax(logits).item())
    started = time.monotonic()
    for _ in range(steps):
        token = int(mx.argmax(engine.step(token, cache)).item())
    return steps / (time.monotonic() - started)


def speed(engine: MLXEngine, tokens: list[int], steps: int) -> Record:
    seconds, logits, cache = timed_prefill(engine, tokens)
    generation = timed_generation(engine, logits, cache, steps)
    del cache
    mx.clear_cache()
    return {
        "prompt_tokens": len(tokens),
        "prefill_seconds": round(seconds, 2),
        "prefill_tokens_per_second": round(len(tokens) / seconds, 1),
        "generation_tokens_per_second": round(generation, 1),
    }


def letter_pass_seconds(engine: MLXEngine, prompts: list[list[int]], prefix_cache: bool) -> float:
    saved = engine.options
    engine.options = replace(saved, prefix_cache=prefix_cache)
    engine.reset_prefix()
    started = time.monotonic()
    for tokens in prompts:
        logits, _ = engine.prefill(tokens)
        mx.eval(logits)
    seconds = time.monotonic() - started
    engine.options = saved
    engine.reset_prefix()
    mx.clear_cache()
    return seconds / len(prompts)


def short_run_profile(paths: RunPaths) -> str | None:
    if not paths.profiles.exists():
        return None
    rows = load_jsonl(paths.profiles)
    return rows[0]["profile"] if rows else None


def mean_profile_output(paths: RunPaths) -> float | None:
    if not paths.profiles.exists():
        return None
    rows = load_jsonl(paths.profiles)
    return sum(row["output_tokens"] for row in rows) / len(rows) if rows else None


def per_item_seconds(timings: list[Record], stage: str, split: str | None = None) -> float | None:
    key = "scored" if stage == "evaluate" else "generated"
    entries = [
        entry
        for entry in timings
        if entry["stage"] == stage
        and (split is None or entry.get("split") == split)
        and entry.get(key, 0) > 0
    ]
    if not entries:
        return None
    latest = entries[-1]
    return latest["seconds"] / latest[key]


def full_run_counts(paths: RunPaths, actors: set[str]) -> Record:
    config = load_config(paths.simulation_config)
    metadata = profiles_module.hearing_metadata(config.lds_path, config.lds_sha256)
    udvs = load_jsonl(config.udv_path)
    owners = turn_owners(load_jsonl(config.speeches_path))
    profiles = {actor: {} for actor in actors}
    counts = {}
    for split in (config.selection_split, config.eval_split):
        questions, _ = evaluation_module.build_questions(
            config, split, profiles, udvs, owners, metadata
        )
        counts[split] = len(questions)
    counts["requests"] = len(
        simulation_module.split_requests(config, profiles, udvs, owners, metadata)
    )
    counts["selection_split"] = config.selection_split
    counts["eval_split"] = config.eval_split
    return counts


def short_run_results(paths: RunPaths) -> Record | None:
    path = paths.simulation_dir / "evaluation.json"
    if not path.exists():
        return None
    with open(path) as f:
        summary = json.load(f)
    evaluation = summary["evaluation"]
    return {
        "questions": evaluation["counts"]["questions"],
        "accuracy": {key: value["accuracy"] for key, value in evaluation["conditions"].items()},
        "letter_mass": evaluation["letter_mass"],
    }


def hours(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds / 3600, 2)


def measure(spec: ModelSpec, paths: RunPaths, settings: Settings) -> Record:
    options = settings.benchmark
    steps = int(options.get("generation_tokens", 128))
    mx.reset_peak_memory()
    started = time.monotonic()
    engine = engine_for(spec.repo, spec.options)
    load_seconds = time.monotonic() - started
    loaded_memory = mx.get_active_memory()

    profiles_config = profiles_module.load_config(paths.profiles_config)
    prompts = profiles_module.load_prompts(profiles_config)
    records = load_jsonl(train_speeches_path(paths))
    metadata = profiles_module.load_hearing_metadata(profiles_config)
    runner = profiles_module.ProfileRunner(config=profiles_config, prompts=prompts, client=None)
    prompt_tokens = {
        record["actor"]: engine.encode(profile_messages(*runner.profile_prompt(record, metadata)))
        for record in records
    }
    long_actor = options.get("long_context_actor") or max(
        prompt_tokens, key=lambda actor: len(prompt_tokens[actor])
    )
    long_context = speed(engine, prompt_tokens[long_actor], steps)
    long_context["actor"] = long_actor
    long_context["peak_memory_gb"] = round(mx.get_peak_memory() / GB, 1)

    material = Material(
        name="Fulano de Tal", role="Deputado", profile=short_run_profile(paths) or PROFILE
    )
    choice_prompts = [
        engine.encode(choice_messages(condition, order))
        for condition in (material.baseline(), material)
        for order in rotations(len(LETTERS))
    ]
    short_context = speed(engine, choice_prompts[-1], steps)
    without_prefix = letter_pass_seconds(engine, choice_prompts, False)
    with_prefix = letter_pass_seconds(engine, choice_prompts, True)

    actors = {record["actor"] for record in records}
    counts = full_run_counts(paths, actors)
    timings = load_timings(paths)
    output_mean = mean_profile_output(paths)
    output_tokens = output_mean or float(options.get("profile_output_tokens_guess", 1200))
    total_input = sum(len(tokens) for tokens in prompt_tokens.values())
    profile_seconds = (
        total_input / long_context["prefill_tokens_per_second"]
        + len(records) * output_tokens / long_context["generation_tokens_per_second"]
    )
    selection = per_item_seconds(timings, "evaluate", counts["selection_split"])
    evaluation = per_item_seconds(timings, "evaluate", counts["eval_split"])
    simulation = per_item_seconds(timings, "simulate")
    evaluate_seconds = (
        selection * counts[counts["selection_split"]] + evaluation * counts[counts["eval_split"]]
        if selection is not None and evaluation is not None
        else None
    )
    simulate_seconds = simulation * counts["requests"] if simulation is not None else None
    parts = [profile_seconds, evaluate_seconds, simulate_seconds]
    return {
        "model": spec.id,
        "repo": spec.repo,
        "prefix_cache_in_runs": spec.options.prefix_cache,
        "load_seconds": round(load_seconds, 1),
        "loaded_memory_gb": round(loaded_memory / GB, 1),
        "long_context": long_context,
        "short_context": short_context,
        "letter_pass_seconds": {
            "without_prefix_cache": round(without_prefix, 2),
            "with_prefix_cache": round(with_prefix, 2),
            "speedup": round(without_prefix / with_prefix, 2),
        },
        "full_run": {
            "actors": len(records),
            "profile_input_tokens": total_input,
            "profile_output_tokens_per_actor": round(output_tokens),
            "profile_output_tokens_source": "short run" if output_mean else "settings guess",
            "questions": {
                counts["selection_split"]: counts[counts["selection_split"]],
                counts["eval_split"]: counts[counts["eval_split"]],
            },
            "requests": counts["requests"],
            "seconds_per_question": {
                counts["selection_split"]: selection,
                counts["eval_split"]: evaluation,
            },
            "seconds_per_request": simulation,
            "hours": {
                "profiles": hours(profile_seconds),
                "evaluate": hours(evaluate_seconds),
                "simulate": hours(simulate_seconds),
                "total": hours(sum(parts)) if all(p is not None for p in parts) else None,
            },
        },
        "short_run": short_run_results(paths),
    }


def markdown(results: dict[str, Record]) -> str:
    header = (
        "| modelo | carga (s) | memória pico (GB) | prefill longo (tok/s) | geração curta (tok/s)"
        " | geração longa (tok/s) | ganho prefixo | letter mass 0 / 1 | horas perfis | horas"
        " avaliação | horas simulação | horas total |"
    )
    lines = [header, "|" + "---|" * 12]
    for result in results.values():
        full = result["full_run"]["hours"]
        run = result["short_run"]
        mass = (
            f"{run['letter_mass']['0']['mean']:.3f} / {run['letter_mass']['1']['mean']:.3f}"
            if run
            else "-"
        )
        cells = [
            result["model"],
            result["load_seconds"],
            result["long_context"]["peak_memory_gb"],
            result["long_context"]["prefill_tokens_per_second"],
            result["short_context"]["generation_tokens_per_second"],
            result["long_context"]["generation_tokens_per_second"],
            f"{result['letter_pass_seconds']['speedup']}x",
            mass,
            full["profiles"],
            full["evaluate"] if full["evaluate"] is not None else "-",
            full["simulate"] if full["simulate"] is not None else "-",
            full["total"] if full["total"] is not None else "-",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure speed and memory of each MLX candidate and project a full run."
    )
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--models", nargs="+", help="model ids (default: every model)")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    output = settings.runs_dir / "benchmark.json"
    results: dict[str, Record] = {}
    if output.exists():
        with open(output) as f:
            results = json.load(f)
    for model_id in args.models or list(settings.models):
        spec = model_spec(settings, model_id)
        activate_model(spec)
        paths = run_paths(settings, spec)
        write_derived_configs(spec, paths)
        prepare_speeches(paths)
        results[model_id] = measure(spec, paths, settings)
        release_engines()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        (settings.runs_dir / "benchmark.md").write_text(markdown(results))
        print(json.dumps(results[model_id], ensure_ascii=False, indent=2))
    print(f"wrote {output} and {settings.runs_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
