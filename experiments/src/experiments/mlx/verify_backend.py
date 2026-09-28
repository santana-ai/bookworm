import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from experiments.actors.evaluate_simulation import rotations
from experiments.actors.simulation import LETTERS, Material, chat_messages, load_prompts, render
from experiments.mlx.backend import MLXSimulationModel, engine_for, log_softmax
from experiments.mlx.run import SMOKE_OPTIONS
from experiments.mlx.settings import (
    DEFAULT_SETTINGS,
    activate_model,
    load_settings,
    model_spec,
    run_paths,
)

Record = dict[str, Any]

PROFILE = (
    "## Posições\n"
    "P1. Defende a ampliação do financiamento público para a educação básica.\n"
    "P2. Cobra do ministério a liberação de recursos para os municípios.\n"
    "## Forma de argumentar\n"
    "F1. Cita dados do orçamento e pede resposta por escrito."
)
TOLERANCE = 1e-2


def choice_messages(material: Material, order: list[int]) -> list[Record]:
    prompts = load_prompts(Path("prompts/actor_simulation"))
    request = render(
        prompts.choice,
        name=material.name,
        options=[SMOKE_OPTIONS[index] for index in order],
        letters=LETTERS,
    )
    return chat_messages(prompts, material, "05/03/2024", "Financiamento da educação", request)


def speech_messages(material: Material) -> list[Record]:
    prompts = load_prompts(Path("prompts/actor_simulation"))
    request = render(prompts.speech, name=material.name)
    return chat_messages(prompts, material, "05/03/2024", "Financiamento da educação", request)


def full_last_logits(model: Any, tokens: list[int]) -> mx.array:
    return model(mx.array(tokens)[None], cache=make_prompt_cache(model))[0, -1].astype(mx.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the MLX backend against direct forwards.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prefix-cache", action="store_true", help="force prefix reuse on")
    args = parser.parse_args()
    settings = load_settings(args.settings)
    spec = model_spec(settings, args.model)
    if args.prefix_cache:
        spec = replace(spec, options=replace(spec.options, prefix_cache=True))
    activate_model(spec)
    engine = engine_for(spec.repo, spec.options)
    simulation = MLXSimulationModel(spec.repo, "auto", spec.options)
    material = Material(name="Fulano de Tal", role="Deputado", profile=PROFILE)
    report: Record = {"model": spec.id, "tolerance": TOLERANCE}

    engine.reset_prefix()
    letters = mx.array(simulation.letter_ids)
    by_condition: Record = {}
    for label, condition in (("0", material.baseline()), ("1", material)):
        raw, renormalized, same_argmax = [], [], []
        for order in rotations(len(LETTERS)):
            messages = choice_messages(condition, order)
            backend = mx.array(simulation.letter_logprobs(messages))
            tokens = engine.encode(messages)
            standard = log_softmax(engine._last_logits(tokens, make_prompt_cache(engine.model)))
            standard = standard[letters]
            raw.append(float(mx.max(mx.abs(backend - standard)).item()))
            renormalized.append(
                float(mx.max(mx.abs(mx.softmax(backend) - mx.softmax(standard))).item())
            )
            same_argmax.append(int(mx.argmax(backend).item()) == int(mx.argmax(standard).item()))
        by_condition[label] = {
            "max_abs_logprob_diff": max(raw),
            "max_abs_renormalized_prob_diff": max(renormalized),
            "same_argmax": f"{sum(same_argmax)}/{len(same_argmax)}",
        }
    report["prefix_cache"] = spec.options.prefix_cache
    report["letters_vs_standard_mlx_prefill"] = by_condition
    report["letters_ok"] = all(
        entry["max_abs_renormalized_prob_diff"] <= TOLERANCE for entry in by_condition.values()
    )

    engine.reset_prefix()
    speech_tokens = engine.encode(speech_messages(material))
    greedy = engine.generate_tokens(speech_tokens, args.steps)
    agreements = [
        int(mx.argmax(full_last_logits(engine.model, speech_tokens + greedy[:index])).item())
        == token
        for index, token in enumerate(greedy)
    ]
    report["greedy_steps_checked"] = len(agreements)
    report["greedy_steps_matching_direct"] = sum(agreements)
    report["greedy_text"] = engine.decode(greedy)

    suffix = "_prefix_cache" if spec.options.prefix_cache else ""
    path = run_paths(settings, spec).root / f"verify{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
