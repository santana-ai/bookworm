"""Multiple-choice evaluation of the actor simulation: which of 4 propositions of the same
hearing the actor made, from no profile (condition 0), the profile (1), the profile plus
retrieved train excerpts (2) and conditions 2 and 0 combined by classifier-free guidance (3)."""

import argparse
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from bookworm import write_json

from experiments.actors.backend import SimulationModel
from experiments.actors.chat import LETTERS, SimulationBackend
from experiments.actors.cli import configure_logging, require_model, write_report
from experiments.actors.io import Job, canonical_sha256, file_info, fingerprint, run_resumable
from experiments.actors.simulation import (
    Material,
    SimulationConfig,
    SimulationInputs,
    SimulationPrompts,
    SpeechRetriever,
    TurnKey,
    add_run_arguments,
    ask_evidence_level,
    chat_messages,
    check_disjoint,
    clean_role,
    config_from_args,
    linked_udvs,
    load_inputs,
    number_profile,
    render,
    split_hearings,
    udv_owner,
)
from experiments.common.transcript import load_encoder_spec, normalize_name, normalize_whitespace

Record = dict[str, Any]

CONDITIONS = ("0", "1", "2", "3")
DISTRACTORS = len(LETTERS) - 1
CONDITION_NAMES = {
    "0": "no profile (name and role)",
    "1": "profile",
    "2": "profile + retrieved train excerpts",
    "3": "condition 2 combined with condition 0 by classifier-free guidance",
}
DISTRACTOR_RULE = (
    "the other options are propositions of 3 distinct other actors of the same hearing: UDVs whose"
    " actor name differs from the target after name normalization and whose evidence turn does not"
    " belong to the target speaker; a text (whitespace-normalized) given to the target in this"
    " hearing never enters the pool, and a text shared by several other actors enters it once,"
    " under the first of them in file order; the actors and one proposition of each are drawn"
    " with random.Random(f'{seed}:{udv_id}'), and questions with fewer than 3 candidate actors"
    " are dropped"
)
SCORING_RULE = (
    "the options are shown in the 4 cyclic rotations of a seeded shuffled order; in each rotation"
    " the log-probabilities of the letter tokens at the first assistant position (log-softmax over"
    " the whole vocabulary) are renormalized over the 4 letters; the option probabilities are"
    " averaged over rotations and the predicted option is the argmax"
)
GUIDANCE_RULE = (
    "condition 3 combines, per rotation, the whole-vocabulary letter log-probabilities of"
    " conditions 0 and 2 as logp0 + gamma * (logp2 - logp0) before renormalizing over the 4"
    " letters, the same formula as transformers' UnbatchedClassifierFreeGuidanceLogitsProcessor"
)
SELECTION_RULE = (
    "k maximizes the accuracy of condition 2 on the selection split, then gamma maximizes the"
    " accuracy of condition 3 with that k; ties go to the higher mean probability of the correct"
    " option, then to the first value in the config grid"
)
LETTER_MASS_RULE = (
    "letter_mass is the sum of the whole-vocabulary probabilities of the 4 letter tokens at the"
    " first assistant position, per rotation, summarized by mean and minimum over questions and"
    " rotations; condition 3 is a combination of log-probabilities, not a distribution, and is"
    " left out"
)
DRY_RUN_RULE = (
    "the questions are built as in a model run (linked UDVs, distractors, cleaned roles);"
    " questions_sha256 is the sha256 of their JSON list, so an equal value means the model"
    " is asked the same questions"
)
EVALUATION_FILE = "evaluation.json"
LEVEL_RULE = (
    "evidence levels are asked without the options, only on the evaluation split, for conditions"
    " 1 and 2 (condition 3 has the material of condition 2 and reuses its level); the label is"
    " the first of DIRETA, INDIRETA, ESPECULATIVA, SEM BASE that starts the greedy response"
)


def choose_options(
    udv: Record,
    actor: str,
    hearing_udvs: list[Record],
    owners: dict[TurnKey, str],
    config: SimulationConfig,
) -> list[Record] | None:
    """The target UDV and 3 seeded distractors from other actors, shuffled, or None."""
    target_name = normalize_name(udv["actor"]["name"])
    others = []
    seen = set()
    for other in hearing_udvs:
        if (
            normalize_name(other["actor"]["name"]) == target_name
            or udv_owner(other, owners) == actor
        ):
            seen.add(normalize_whitespace(other["proposition"]))
        else:
            others.append(other)
    pool: dict[str, list[Record]] = defaultdict(list)
    for other in others:
        text = normalize_whitespace(other["proposition"])
        if text in seen:
            continue
        seen.add(text)
        pool[normalize_name(other["actor"]["name"])].append(other)
    if len(pool) < DISTRACTORS:
        return None
    rng = random.Random(f"{config.eval_seed}:{udv['id']}")
    names = rng.sample(sorted(pool), DISTRACTORS)
    options = [udv] + [rng.choice(pool[name]) for name in names]
    rng.shuffle(options)
    return options


def build_questions(
    config: SimulationConfig,
    split: str,
    profiles: dict[str, Record],
    udvs: list[Record],
    owners: dict[TurnKey, str],
    metadata: dict[int, Record],
) -> tuple[list[Record], Record]:
    hearings = split_hearings(config, split)
    by_hearing: dict[int, list[Record]] = defaultdict(list)
    for udv in udvs:
        if udv["hearing_id"] in hearings:
            by_hearing[udv["hearing_id"]].append(udv)
    linked = linked_udvs(udvs, owners, set(profiles), hearings)
    questions: list[Record] = []
    dropped: list[str] = []
    for udv, actor in linked:
        options = choose_options(udv, actor, by_hearing[udv["hearing_id"]], owners, config)
        if options is None:
            dropped.append(udv["id"])
            continue
        hearing = metadata[udv["hearing_id"]]
        questions.append(
            {
                "udv_id": udv["id"],
                "hearing_id": udv["hearing_id"],
                "actor": actor,
                "tier": udv["tier"],
                "cargo": udv["actor"]["role"],
                "role": clean_role(udv["actor"]["role"], config.parties),
                "date": hearing["date_br"],
                "assunto": hearing["assunto"],
                "options": [option["proposition"] for option in options],
                "option_udv_ids": [option["id"] for option in options],
                "answer": options.index(udv),
            }
        )
    counts = {
        "split": split,
        "hearings": len(hearings),
        "linked_udvs": len(linked),
        "linked_actors": len({actor for _, actor in linked}),
        "dropped_without_distractors": dropped,
        "questions": len(questions),
        "question_actors": len({question["actor"] for question in questions}),
        "question_hearings": len({question["hearing_id"] for question in questions}),
        "questions_by_tier": dict(sorted(Counter(q["tier"] for q in questions).items())),
        "roles_with_party_removed": sum(
            question["role"] != normalize_whitespace(question["cargo"] or "")
            for question in questions
        ),
    }
    return questions, counts


def rotations(size: int) -> list[list[int]]:
    return [[(start + position) % size for position in range(size)] for start in range(size)]


@dataclass(frozen=True)
class Evaluator:
    config: SimulationConfig
    prompts: SimulationPrompts
    model: SimulationBackend
    retriever: SpeechRetriever

    def choice_logprobs(self, material: Material, question: Record) -> list[list[float]]:
        """Letter log-probabilities per rotation of the options, reordered by option."""
        options = question["options"]
        by_rotation = []
        for order in rotations(len(options)):
            request = render(
                self.prompts.choice,
                name=material.name,
                options=[options[index] for index in order],
                letters=LETTERS,
            )
            messages = chat_messages(
                self.prompts, material, question["date"], question["assunto"], request
            )
            letters = self.model.letter_logprobs(messages)
            by_option = [0.0] * len(options)
            for position, option in enumerate(order):
                by_option[option] = letters[position]
            by_rotation.append(by_option)
        return by_rotation

    def evidence_level(self, material: Material, question: Record) -> Record:
        return ask_evidence_level(
            self.model,
            self.prompts,
            material,
            question["date"],
            question["assunto"],
            self.config.evidence_max_tokens,
        )

    def score(
        self, question: Record, profile: Record, ks: tuple[int, ...], with_levels: bool
    ) -> Record:
        material = Material(
            name=question["actor"],
            role=question["role"],
            profile=number_profile(profile["profile"]),
        )
        with_excerpts = {
            k: material.with_excerpts(
                self.retriever.retrieve(question["actor"], question["assunto"], k)
            )
            for k in ks
        }
        row = {
            **question,
            "logprobs": {
                "0": self.choice_logprobs(material.baseline(), question),
                "1": self.choice_logprobs(material, question),
                "2": {str(k): self.choice_logprobs(m, question) for k, m in with_excerpts.items()},
            },
            "excerpts": {
                str(k): [excerpt.source for excerpt in m.excerpts] for k, m in with_excerpts.items()
            },
            "profile_prompt_version": profile["prompt_version"],
        }
        if with_levels:
            (k,) = ks
            row["levels"] = {
                "1": self.evidence_level(material, question),
                "2": self.evidence_level(with_excerpts[k], question),
            }
        return row


def score_split(
    evaluator: Evaluator,
    questions: list[Record],
    profiles: dict[str, Record],
    ks: tuple[int, ...],
    with_levels: bool,
    run_digest: str,
    path: Path,
) -> list[Record]:
    """Score every question whose row in `path` is missing or stale."""

    def jobs() -> Iterator[Job]:
        for question in questions:
            profile = profiles[question["actor"]]
            yield Job(
                id=question["udv_id"],
                label=f"{question['udv_id']} ({question['actor']})",
                fingerprint=fingerprint(run_digest, profile["profile"], question, ks, with_levels),
                compute=partial(evaluator.score, question, profile, ks, with_levels),
            )

    return run_resumable(path, "udv_id", jobs(), len(questions))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def condition_logprobs(row: Record, condition: str, k: int) -> np.ndarray:
    logprobs = row["logprobs"]
    return np.array(logprobs["2"][str(k)] if condition == "2" else logprobs[condition])


def option_probabilities(row: Record, condition: str, k: int, gamma: float) -> np.ndarray:
    if condition == "3":
        base = condition_logprobs(row, "0", k)
        values = base + gamma * (condition_logprobs(row, "2", k) - base)
    else:
        values = condition_logprobs(row, condition, k)
    return softmax(values).mean(axis=0)


def outcomes(
    rows: list[Record], condition: str, k: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = [option_probabilities(row, condition, k, gamma) for row in rows]
    answers = [row["answer"] for row in rows]
    correct = np.array(
        [float(np.argmax(p) == a) for p, a in zip(probabilities, answers, strict=True)]
    )
    p_correct = np.array([p[a] for p, a in zip(probabilities, answers, strict=True)])
    return correct, p_correct


def metrics(rows: list[Record], condition: str, k: int, gamma: float) -> Record:
    correct, p_correct = outcomes(rows, condition, k, gamma)
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "mean_p_correct": float(p_correct.mean()),
    }


def letter_mass(rows: list[Record], k: int) -> Record:
    mass: Record = {}
    for condition in ("0", "1", "2"):
        sums = np.concatenate(
            [np.exp(condition_logprobs(row, condition, k)).sum(axis=-1) for row in rows]
        )
        mass[condition] = {"mean": float(sums.mean()), "min": float(sums.min())}
    return mass


def best(grid: tuple[Any, ...], results: dict[Any, Record]) -> Any:
    return max(
        grid, key=lambda value: (results[value]["accuracy"], results[value]["mean_p_correct"])
    )


def select(rows: list[Record], config: SimulationConfig) -> Record:
    by_k = {k: metrics(rows, "2", k, 1.0) for k in config.k_grid}
    k = best(config.k_grid, by_k)
    by_gamma = {gamma: metrics(rows, "3", k, gamma) for gamma in config.guidance_grid}
    gamma = best(config.guidance_grid, by_gamma)
    return {
        "k": k,
        "guidance_scale": gamma,
        "rule": SELECTION_RULE,
        "condition_2_by_k": {str(value): result for value, result in by_k.items()},
        "condition_3_by_guidance_scale": {str(value): result for value, result in by_gamma.items()},
    }


def bootstrap_interval(
    delta: np.ndarray, hearing_ids: list[int], samples: int, seed: int
) -> list[float]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, hearing_id in enumerate(hearing_ids):
        groups[hearing_id].append(index)
    members = [np.array(indices) for _, indices in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        drawn = rng.integers(len(members), size=len(members))
        means.append(delta[np.concatenate([members[i] for i in drawn])].mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def accuracy_by_level(rows: list[Record], correct: dict[str, np.ndarray]) -> Record:
    by_level: Record = {}
    for condition in ("1", "2", "3"):
        source = "2" if condition == "3" else condition
        groups: dict[str, list[float]] = defaultdict(list)
        for row, hit in zip(rows, correct[condition], strict=True):
            groups[row["levels"][source]["label"] or "unparsed"].append(hit)
        by_level[condition] = {
            label: {"n": len(hits), "accuracy": float(np.mean(hits))}
            for label, hits in sorted(groups.items())
        }
    return by_level


def evaluate(rows: list[Record], k: int, gamma: float, config: SimulationConfig) -> Record:
    correct = {condition: outcomes(rows, condition, k, gamma)[0] for condition in CONDITIONS}
    hearing_ids = [row["hearing_id"] for row in rows]
    return {
        "k": k,
        "guidance_scale": gamma,
        "conditions": {
            condition: {"name": CONDITION_NAMES[condition], **metrics(rows, condition, k, gamma)}
            for condition in CONDITIONS
        },
        "letter_mass": letter_mass(rows, k),
        "differences": {
            f"{after}-{before}": {
                "accuracy": float((correct[after] - correct[before]).mean()),
                "ci95": bootstrap_interval(
                    correct[after] - correct[before],
                    hearing_ids,
                    config.bootstrap_samples,
                    config.eval_seed,
                ),
            }
            for before, after in pairwise(CONDITIONS)
        },
        "bootstrap": (
            f"paired percentile interval over {config.bootstrap_samples} resamples of whole"
            f" hearings, seed {config.eval_seed}"
        ),
        "accuracy_by_level": accuracy_by_level(rows, correct),
    }


def print_report(summary: Record) -> None:
    for key in ("selection", "evaluation"):
        counts = summary[key]["counts"]
        print(
            f"{counts['split']}: {counts['questions']} questions ({counts['question_actors']}"
            f" actors, {counts['question_hearings']} hearings), {counts['linked_udvs']} linked"
            f" UDVs, {len(counts['dropped_without_distractors'])} dropped without distractors"
        )
    selection = summary["selection"]
    split = selection["counts"]["split"]
    print(f"selected on {split}: k={selection['k']}, gamma={selection['guidance_scale']}")
    evaluation = summary["evaluation"]
    for condition, result in evaluation["conditions"].items():
        print(
            f"  condition {condition}: accuracy {result['accuracy']:.3f},"
            f" mean p(correct) {result['mean_p_correct']:.3f} (n={result['n']})"
        )
    for name, difference in evaluation["differences"].items():
        low, high = difference["ci95"]
        print(f"  {name}: {difference['accuracy']:+.3f} [{low:+.3f}, {high:+.3f}]")


def question_dry_run(
    config: SimulationConfig,
    prompt_version: str,
    input_files: Record,
    questions: dict[str, list[Record]],
    counts: dict[str, Record],
) -> Record:
    return {
        "dry_run": True,
        "model": config.model or None,
        "prompt_version": prompt_version,
        "inputs": input_files,
        "splits": {
            split: {
                "counts": counts[split],
                "questions_sha256": canonical_sha256(split_questions),
                "udv_ids_sha256": canonical_sha256([q["udv_id"] for q in split_questions]),
            }
            for split, split_questions in questions.items()
        },
        "rule": DRY_RUN_RULE,
    }


def evaluation_input_files(config: SimulationConfig, profiles: dict[str, Record]) -> Record:
    return {
        "profiles": {
            **file_info(config.profiles_path),
            "actors": len(profiles),
            "prompt_versions": sorted({row["prompt_version"] for row in profiles.values()}),
            "models": sorted({row["model"] for row in profiles.values()}),
        },
        "train_speeches": file_info(config.train_speeches_path),
        "speeches": file_info(config.speeches_path),
        "udv": file_info(config.udv_path),
        "split_manifest": file_info(config.manifest_path),
    }


def run_evaluation(
    config: SimulationConfig,
    inputs: SimulationInputs,
    input_files: Record,
    selection_questions: list[Record],
    eval_questions: list[Record],
) -> tuple[Evaluator, list[Record], Record, list[Record]]:
    """Score the selection split over the k grid, choose k and gamma, then score the
    evaluation split with that k and the evidence levels."""
    prompts = inputs.prompts
    run_digest = fingerprint(
        config.model,
        config.evidence_max_tokens,
        load_encoder_spec(),
        prompts.version,
        {name: info for name, info in input_files.items() if name != "profiles"},
    )
    logging.info(
        "%d + %d questions, prompts %s, loading %s",
        len(selection_questions),
        len(eval_questions),
        prompts.version,
        config.model,
    )
    evaluator = Evaluator(
        config=config,
        prompts=prompts,
        model=SimulationModel(config.model, config.device_map),
        retriever=SpeechRetriever(inputs.train_records, inputs.metadata),
    )
    selection_rows = score_split(
        evaluator,
        selection_questions,
        inputs.profiles,
        config.k_grid,
        False,
        run_digest,
        config.output_dir / f"choice_{config.selection_split}.jsonl",
    )
    selection = select(selection_rows, config)
    eval_rows = score_split(
        evaluator,
        eval_questions,
        inputs.profiles,
        (selection["k"],),
        True,
        run_digest,
        config.output_dir / f"choice_{config.eval_split}.jsonl",
    )
    return evaluator, selection_rows, selection, eval_rows


def evaluation_summary(
    config: SimulationConfig,
    evaluator: Evaluator,
    input_files: Record,
    counts: dict[str, Record],
    selection_rows: list[Record],
    selection: Record,
    eval_rows: list[Record],
) -> Record:
    return {
        "model": config.model,
        "prompt_version": evaluator.prompts.version,
        "letter_token_ids": dict(zip(LETTERS, evaluator.model.letter_ids, strict=True)),
        "encoder": dict(zip(("name", "revision"), load_encoder_spec(), strict=True)),
        "inputs": input_files,
        "rules": {
            "distractors": DISTRACTOR_RULE,
            "scoring": SCORING_RULE,
            "guidance": GUIDANCE_RULE,
            "levels": LEVEL_RULE,
            "letter_mass": LETTER_MASS_RULE,
        },
        "selection": {
            "counts": counts[config.selection_split],
            "conditions": {
                condition: metrics(selection_rows, condition, selection["k"], 1.0)
                for condition in ("0", "1")
            },
            "letter_mass": letter_mass(selection_rows, selection["k"]),
            **selection,
        },
        "evaluation": {
            "counts": counts[config.eval_split],
            **evaluate(eval_rows, selection["k"], selection["guidance_scale"], config),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multiple-choice evaluation of actor simulation: choose k and gamma on the selection"
            " split, then score conditions 0-3 and evidence levels on the evaluation split."
        )
    )
    add_run_arguments(parser)
    parser.add_argument(
        "--dry-run",
        type=Path,
        help="build the questions of both splits without a model and write their counts to this"
        " JSON",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()
    config = config_from_args(args)
    require_model(parser, config.model, args.dry_run is not None)
    if config.selection_split == config.eval_split:
        parser.error("selection_split and eval_split must differ")
    inputs = load_inputs(config, args.actors)
    evaluated = split_hearings(config, config.selection_split) | split_hearings(
        config, config.eval_split
    )
    check_disjoint(inputs.profiles, inputs.train_records, evaluated)
    questions: dict[str, list[Record]] = {}
    counts: dict[str, Record] = {}
    for split in (config.selection_split, config.eval_split):
        questions[split], counts[split] = build_questions(
            config, split, inputs.profiles, inputs.udvs, inputs.owners, inputs.metadata
        )
    selection_questions = questions[config.selection_split]
    eval_questions = questions[config.eval_split]
    if not selection_questions or not eval_questions:
        raise SystemExit(
            f"no questions to score: {len(selection_questions)} on {config.selection_split},"
            f" {len(eval_questions)} on {config.eval_split}; k and gamma are chosen on"
            f" {config.selection_split}, so the profiled actors need questions in both splits"
        )
    input_files = evaluation_input_files(config, inputs.profiles)
    if args.dry_run is not None:
        report = question_dry_run(config, inputs.prompts.version, input_files, questions, counts)
        write_report(report, args.dry_run)
        return
    evaluator, selection_rows, selection, eval_rows = run_evaluation(
        config, inputs, input_files, selection_questions, eval_questions
    )
    summary = evaluation_summary(
        config, evaluator, input_files, counts, selection_rows, selection, eval_rows
    )
    write_json(summary, config.output_dir / EVALUATION_FILE)
    print_report(summary)
    print(f"wrote {config.output_dir / EVALUATION_FILE}")


if __name__ == "__main__":
    main()
