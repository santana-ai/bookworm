"""Fixed probe prompts for the MLX smoke test, the backend check and the benchmark: a fictitious
deputy, one hearing date and topic, four choice options and a short profile."""

from functools import cache
from pathlib import Path
from typing import Any

from experiments.actors.chat import LETTERS
from experiments.actors.simulation import (
    Material,
    SimulationPrompts,
    chat_messages,
    load_prompts,
    render,
)

Record = dict[str, Any]

PROMPTS_DIR = Path("prompts/actor_simulation")
PROBE_NAME = "Fulano de Tal"
PROBE_ROLE = "Deputado"
PROBE_DATE = "05/03/2024"
PROBE_TOPIC = "Financiamento da educação"
SMOKE_OPTIONS = (
    "Defendeu a ampliação do financiamento público para a educação básica.",
    "Criticou a demora do ministério na liberação de recursos para os municípios.",
    "Propôs a criação de uma comissão especial para acompanhar o programa.",
    "Afirmou que a proposta transfere custos para os estados sem compensação.",
)
PROFILE = (
    "## Posições\n"
    "P1. Defende a ampliação do financiamento público para a educação básica.\n"
    "P2. Cobra do ministério a liberação de recursos para os municípios.\n"
    "## Forma de argumentar\n"
    "F1. Cita dados do orçamento e pede resposta por escrito."
)
PT_BR_QUESTION = [
    {"role": "user", "content": "Em duas frases, o que é uma audiência pública na Câmara?"}
]


@cache
def probe_prompts() -> SimulationPrompts:
    return load_prompts(PROMPTS_DIR)


def probe_material(profile: str | None) -> Material:
    return Material(name=PROBE_NAME, role=PROBE_ROLE, profile=profile)


def probe_messages(material: Material, request: str) -> list[Record]:
    return chat_messages(probe_prompts(), material, PROBE_DATE, PROBE_TOPIC, request)


def choice_messages(material: Material, order: list[int]) -> list[Record]:
    """The multiple-choice call with the smoke options in this order."""
    request = render(
        probe_prompts().choice,
        name=material.name,
        options=[SMOKE_OPTIONS[index] for index in order],
        letters=LETTERS,
    )
    return probe_messages(material, request)


def speech_messages(material: Material) -> list[Record]:
    return probe_messages(material, render(probe_prompts().speech, name=material.name))
