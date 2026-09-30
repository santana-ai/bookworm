"""Check the MLX backend against direct forwards: the letter log-probabilities of the probe
choice against a standard prefill, and each greedy step against a full forward pass."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from experiments.actors.chat import LETTERS
from experiments.actors.evaluate_simulation import rotations
from experiments.actors.simulation import Material
from experiments.mlx.backend import MLXEngine, MLXSimulationModel, engine_for, log_softmax
from experiments.mlx.probe import PROFILE, choice_messages, probe_material, speech_messages
from experiments.mlx.settings import (
    DEFAULT_SETTINGS,
    activate_model,
    load_settings,
    model_spec,
    run_paths,
    write_json_file,
)

Record = dict[str, Any]

TOLERANCE = 1e-2


def full_last_logits(model: Any, tokens: list[int]) -> mx.array:
    return model(mx.array(tokens)[None], cache=make_prompt_cache(model))[0, -1].astype(mx.float32)


def compare_letters(
    engine: MLXEngine, simulation: MLXSimulationModel, condition: Material
) -> Record:
    """Backend letter log-probabilities against a prefill without prefix reuse, per rotation."""
    letters = mx.array(simulation.letter_ids)
    raw, renormalized, same_argmax = [], [], []
    for order in rotations(len(LETTERS)):
        messages = choice_messages(condition, order)
        backend = mx.array(simulation.letter_logprobs(messages))
        tokens = engine.encode(messages)
        standard = log_softmax(engine.last_logits(tokens, make_prompt_cache(engine.model)))
        standard = standard[letters]
        raw.append(float(mx.max(mx.abs(backend - standard)).item()))
        renormalized.append(
            float(mx.max(mx.abs(mx.softmax(backend) - mx.softmax(standard))).item())
        )
        same_argmax.append(int(mx.argmax(backend).item()) == int(mx.argmax(standard).item()))
    return {
        "max_abs_logprob_diff": max(raw),
        "max_abs_renormalized_prob_diff": max(renormalized),
        "same_argmax": f"{sum(same_argmax)}/{len(same_argmax)}",
    }


def compare_greedy(engine: MLXEngine, material: Material, steps: int) -> Record:
    """Each greedy token of the backend against the argmax of a full forward pass."""
    engine.reset_prefix()
    speech_tokens = engine.encode(speech_messages(material))
    greedy = engine.generate_tokens(speech_tokens, steps)
    agreements = [
        int(mx.argmax(full_last_logits(engine.model, speech_tokens + greedy[:index])).item())
        == token
        for index, token in enumerate(greedy)
    ]
    return {
        "greedy_steps_checked": len(agreements),
        "greedy_steps_matching_direct": sum(agreements),
        "greedy_text": engine.decode(greedy),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the MLX backend against direct forwards.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prefix-cache", action="store_true", help="force prefix reuse on")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.settings)
    spec = model_spec(settings, args.model)
    if args.prefix_cache:
        spec = replace(spec, options=replace(spec.options, prefix_cache=True))
    activate_model(spec)
    engine = engine_for(spec.repo, spec.options)
    simulation = MLXSimulationModel(spec.repo, "auto", spec.options)
    material = probe_material(PROFILE)
    report: Record = {"model": spec.id, "tolerance": TOLERANCE}
    engine.reset_prefix()
    by_condition = {
        label: compare_letters(engine, simulation, condition)
        for label, condition in (("0", material.baseline()), ("1", material))
    }
    report["prefix_cache"] = spec.options.prefix_cache
    report["letters_vs_standard_mlx_prefill"] = by_condition
    report["letters_ok"] = all(
        entry["max_abs_renormalized_prob_diff"] <= TOLERANCE for entry in by_condition.values()
    )
    report |= compare_greedy(engine, material, args.steps)
    suffix = "_prefix_cache" if spec.options.prefix_cache else ""
    write_json_file(report, run_paths(settings, spec).root / f"verify{suffix}.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
