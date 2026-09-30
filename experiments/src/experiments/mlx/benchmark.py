"""Speed and memory of each MLX candidate model, and a projection of the time of a full run
from the long-context speed and the per-item times of the short run."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
from bookworm import load_jsonl

import experiments.actors.evaluate_simulation as evaluation_module
import experiments.actors.generate_profiles as profiles_module
import experiments.actors.simulate as simulation_module
from experiments.actors.chat import LETTERS, system_user_messages
from experiments.actors.evaluate_simulation import rotations
from experiments.actors.simulation import load_config, turn_owners
from experiments.mlx.backend import MLXEngine, engine_for, release_engines
from experiments.mlx.probe import PROFILE, choice_messages, probe_material
from experiments.mlx.settings import (
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
    write_json_file,
)

Record = dict[str, Any]

GB = 1e9
SECONDS_PER_HOUR = 3600
DEFAULT_GENERATION_TOKENS = 128
DEFAULT_PROFILE_OUTPUT_TOKENS = 1200


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


def short_run_profiles(paths: RunPaths) -> list[Record]:
    return load_jsonl(paths.profiles) if paths.profiles.exists() else []


def short_run_profile(paths: RunPaths) -> str | None:
    rows = short_run_profiles(paths)
    return rows[0]["profile"] if rows else None


def mean_profile_output(paths: RunPaths) -> float | None:
    rows = short_run_profiles(paths)
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
    profiles: dict[str, Record] = {actor: {} for actor in actors}
    counts: Record = {}
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
    return None if seconds is None else round(seconds / SECONDS_PER_HOUR, 2)


def profile_prompt_tokens(
    engine: MLXEngine, paths: RunPaths, records: list[Record]
) -> dict[str, list[int]]:
    """The tokens of every train actor's profile prompt, by actor."""
    config = profiles_module.load_config(paths.profiles_config)
    prompts = profiles_module.load_prompts(config)
    metadata = profiles_module.load_hearing_metadata(config)
    return {
        record["actor"]: engine.encode(
            system_user_messages(*profiles_module.render_profile_prompt(prompts, record, metadata))
        )
        for record in records
    }


def project_full_run(
    counts: Record,
    timings: list[Record],
    actors: int,
    input_tokens: int,
    output_tokens: float,
    long_context: Record,
) -> Record:
    """Hours of the profile stage from the measured speeds, and of the evaluation and
    simulation stages from the latest per-item times, when there are any."""
    profile_seconds = (
        input_tokens / long_context["prefill_tokens_per_second"]
        + actors * output_tokens / long_context["generation_tokens_per_second"]
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
    }


def measure(spec: ModelSpec, paths: RunPaths, settings: Settings) -> Record:
    options = settings.benchmark
    steps = int(options.get("generation_tokens", DEFAULT_GENERATION_TOKENS))
    mx.reset_peak_memory()
    started = time.monotonic()
    engine = engine_for(spec.repo, spec.options)
    load_seconds = time.monotonic() - started
    loaded_memory = mx.get_active_memory()

    records = load_jsonl(train_speeches_path(paths))
    prompt_tokens = profile_prompt_tokens(engine, paths, records)
    long_actor = options.get("long_context_actor") or max(
        prompt_tokens, key=lambda actor: len(prompt_tokens[actor])
    )
    long_context = speed(engine, prompt_tokens[long_actor], steps)
    long_context["actor"] = long_actor
    long_context["peak_memory_gb"] = round(mx.get_peak_memory() / GB, 1)

    material = probe_material(short_run_profile(paths) or PROFILE)
    choice_prompts = [
        engine.encode(choice_messages(condition, order))
        for condition in (material.baseline(), material)
        for order in rotations(len(LETTERS))
    ]
    short_context = speed(engine, choice_prompts[-1], steps)
    without_prefix = letter_pass_seconds(engine, choice_prompts, False)
    with_prefix = letter_pass_seconds(engine, choice_prompts, True)

    counts = full_run_counts(paths, {record["actor"] for record in records})
    timings = load_timings(paths)
    output_mean = mean_profile_output(paths)
    output_tokens = output_mean or float(
        options.get("profile_output_tokens_guess", DEFAULT_PROFILE_OUTPUT_TOKENS)
    )
    total_input = sum(len(tokens) for tokens in prompt_tokens.values())
    projection = project_full_run(
        counts, timings, len(records), total_input, output_tokens, long_context
    )
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
            **projection,
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
        write_json_file(results, output)
        (settings.runs_dir / "benchmark.md").write_text(markdown(results))
        print(json.dumps(results[model_id], ensure_ascii=False, indent=2))
    print(f"wrote {output} and {settings.runs_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
