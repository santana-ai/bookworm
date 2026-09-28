import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bookworm import load_jsonl, sha256_of_file, write_json

from experiments.actors import evaluate_simulation, simulation
from experiments.actors.generate_profiles import hearing_metadata
from experiments.common.provenance import source_hashes

Record = dict[str, Any]

SPEECH_FILES = (
    Path("artifacts/cache/hearing_actors/actors_single_hearing.jsonl"),
    Path("artifacts/cache/hearing_actors/actors_multi_hearing.jsonl"),
)
TRAIN_SPEECHES = Path("artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl")
MANIFEST = Path("artifacts/splits/temporal_v1.json")
SIMULATION_CONFIG = Path("configs/actor_simulation.toml")
PROFILES = Path("artifacts/mlx_runs/qwen38_27b/actor_profiles/actor_profiles_train.jsonl")
LINKS_V2 = Path("artifacts/udv/udv_v2_actor_links.jsonl")
OUTPUT = Path("artifacts/udv/udv_v2_downstream_report.json")
RUNS = {"udv_v1": Path("artifacts/udv/udv_v1.jsonl"), "udv_v2": Path("artifacts/udv/udv_v2.jsonl")}
SPLITS = ("train", "validation", "test")
TIERS = (
    "quote_found",
    "semantic_match_high",
    "semantic_match_weak",
    "no_evidence",
    "person_not_resolved",
)
LIBRARY_RULE = (
    "the links of udv_v2_actor_links.jsonl, written by bookworm build-udvs --actors-config: the"
    " actor owning the most kept turns among the turns matched to the participant; the rule reads"
    " no UDV evidence, so the same links apply to udv_v1"
)
EVIDENCE_RULE = (
    "the actor owning the kept turn (hearing_id, evidence.speaker_turn) in the single and multi"
    " hearing speech files (display names); it is the rule of summarize_evaluation and of the"
    " simulation questions"
)
PROFILED_RULE = (
    "UDVs of a split whose linked actor is in the train speech file (the multi hearing file"
    " filtered to train hearings), as counted for profile evaluation"
)
OVERLAP_RULE = (
    "simulation questions built by evaluate_actor_simulation.build_questions on each run with the"
    " same profiles, speeches and seeds; a question is identical when every field, options"
    " included, is equal"
)


@dataclass(frozen=True)
class Artifact:
    name: str
    paths: Mapping[str, str | None]
    fields: Sequence[str]
    note: str | None = None


def library_links(path: Path) -> dict[str, str]:
    return {row["udv_id"]: row["actor"] for row in load_jsonl(path) if row["actor"] is not None}


def evidence_turn_links(
    udvs: Iterable[Record], owners: dict[tuple[int, int], str]
) -> dict[str, str]:
    links: dict[str, str] = {}
    for udv in udvs:
        actor = simulation.udv_owner(udv, owners)
        if actor is not None:
            links[udv["id"]] = actor
    return links


def split_of(manifest: Mapping[str, Any]) -> dict[int, str]:
    return {hearing: split for split in SPLITS for hearing in manifest[split]}


def profiled_counts(
    udvs: Mapping[str, Record],
    links: Mapping[str, str],
    profiled: set[str],
    hearing_split: Mapping[int, str],
) -> dict[str, Record]:
    counts: dict[str, Record] = {}
    for split in SPLITS:
        linked = {
            uid: actor
            for uid, actor in links.items()
            if hearing_split.get(udvs[uid]["hearing_id"]) == split and actor in profiled
        }
        counts[split] = {
            "udvs": len(linked),
            "actors": len(set(linked.values())),
            "hearings": len({udvs[uid]["hearing_id"] for uid in linked}),
            "by_tier": dict(sorted(Counter(udvs[uid]["tier"] for uid in linked).items())),
        }
    return counts


def link_comparison(
    udvs: Mapping[str, Record],
    library: Mapping[str, str],
    evidence: Mapping[str, str],
    profiled: set[str],
    hearing_split: Mapping[int, str],
) -> Record:
    both = sorted(set(library) & set(evidence))
    disagreements = [uid for uid in both if library[uid] != evidence[uid]]
    library_only = sorted(set(library) - set(evidence))
    evidence_only = sorted(set(evidence) - set(library))
    return {
        "udvs": len(udvs),
        "library_linked": len(library),
        "evidence_turn_linked": len(evidence),
        "both_linked": len(both),
        "disagreements": len(disagreements),
        "disagreement_examples": [
            {"udv_id": uid, "library": library[uid], "evidence_turn": evidence[uid]}
            for uid in disagreements[:20]
        ],
        "library_only": {uid: library[uid] for uid in library_only},
        "evidence_turn_only": {uid: evidence[uid] for uid in evidence_only},
        "profiled_by_split": {
            "library": profiled_counts(udvs, library, profiled, hearing_split),
            "evidence_turn": profiled_counts(udvs, evidence, profiled, hearing_split),
        },
    }


def split_tier_counts(udvs: Iterable[Record], hearing_split: Mapping[int, str]) -> Record:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    hearings: dict[str, set[int]] = {split: set() for split in SPLITS}
    for udv in udvs:
        split = hearing_split[udv["hearing_id"]]
        counts[split][udv["tier"]] += 1
        hearings[split].add(udv["hearing_id"])
    return {
        split: {
            "udvs": sum(counts[split].values()),
            "hearings": len(hearings[split]),
            "by_tier": {tier: counts[split][tier] for tier in TIERS},
        }
        for split in SPLITS
    }


def simulation_questions(udv_path: Path, profiles_path: Path) -> dict[str, list[Record]]:
    config = simulation.load_config(SIMULATION_CONFIG)
    profiles = simulation.load_profiles(profiles_path, None)
    metadata = hearing_metadata(config.lds_path, config.lds_sha256)
    owners = simulation.turn_owners(load_jsonl(config.speeches_path))
    udvs = load_jsonl(udv_path)
    return {
        split: evaluate_simulation.build_questions(config, split, profiles, udvs, owners, metadata)[
            0
        ]
        for split in (config.selection_split, config.eval_split)
    }


def question_overlap(v1: Mapping[str, list[Record]], v2: Mapping[str, list[Record]]) -> Record:
    overlap: Record = {"rule": OVERLAP_RULE}
    for split, questions in v1.items():
        old = {question["udv_id"]: question for question in questions}
        new = {question["udv_id"]: question for question in v2[split]}
        shared = sorted(set(old) & set(new))
        overlap[split] = {
            "udv_v1_questions": len(old),
            "udv_v2_questions": len(new),
            "same_udv": len(shared),
            "identical": sum(old[uid] == new[uid] for uid in shared),
            "tier_changed": sum(old[uid]["tier"] != new[uid]["tier"] for uid in shared),
            "options_changed": sum(old[uid]["options"] != new[uid]["options"] for uid in shared),
            "only_udv_v1": sorted(set(old) - set(new)),
            "only_udv_v2": sorted(set(new) - set(old)),
        }
    return overlap


def pick(record: Any, dotted: str) -> Any:
    value = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def artifact_section(artifact: Artifact) -> Record:
    section: Record = {"artifacts": dict(artifact.paths)}
    for run, path in artifact.paths.items():
        if path is None:
            section[run] = None
            continue
        with open(path) as f:
            payload = json.load(f)
        section[run] = {field: pick(payload, field) for field in artifact.fields}
    if artifact.note is not None:
        section["note"] = artifact.note
    return section


ARTIFACTS = (
    Artifact(
        "udv_verifier",
        {
            "udv_v1": "artifacts/udv/udv_v1_verifier_report.json",
            "udv_v2": "artifacts/udv/udv_v2_verifier_report.json",
        },
        (
            "counts.scored_by_tier",
            "counts.scored_by_support_type",
            "supported_at_train_threshold.by_tier",
            "supported_at_train_threshold.by_support_type",
            "supported_at_train_threshold.by_hearing_split",
            "supported_at_udv_threshold.by_tier",
            "refit_check.checks",
            "pool_check.max_abs_gap_by_scorer",
            "evidence_score_check",
            "cosine_relation.spearman.semantic_all",
            "inputs.udv.sha256",
            "outputs.sha256",
        ),
        "evidence_score_check compares evidence.score with the cosine of the unit the verifier"
        " read: the sentence_max cosine for udv_v1, the cosine of the whole window for udv_v2;"
        " on udv_v2 the UDVs of encoded_text_differs have an evidence span that also holds"
        " parts the encoder skipped, so the verifier premise is longer than the encoded text",
    ),
    Artifact(
        "chair_evidence",
        {
            "udv_v1": "artifacts/hearing_actors/measurements.json",
            "udv_v2": "artifacts/hearing_actors/measurements_udv_v2.json",
        },
        (
            "udv_chair_evidence.located_evidence_by_tier",
            "udv_chair_evidence.chair_evidence",
            "udv_chair_evidence.chair_evidence_by_tier",
            "udv_chair_evidence.chair_evidence_actors",
            "udv_chair_evidence.chair_cuts",
        ),
    ),
    Artifact(
        "profile_evaluation_links",
        {
            "udv_v1": "artifacts/actor_profiles/train_speeches_stats.json",
            "udv_v2": "artifacts/actor_profiles/train_speeches_stats_udv_v2.json",
        },
        (
            "input.sha256",
            "output.sha256",
            "output.actors",
            "evaluation.udvs",
            "evaluation.linked_udvs",
            "evaluation.linked_actors",
            "evaluation.linked_hearings",
            "evaluation.linked_udvs_by_tier",
        ),
    ),
    Artifact(
        "profile_validation",
        {
            "udv_v1": "artifacts/profile_validation/profile_validation_udv_v1_report.json",
            "udv_v2": "artifacts/profile_validation/profile_validation_udv_v2_report.json",
        },
        (
            "inputs.profiles.sha256",
            "counts.skipped",
            "counts.pairs",
            "groups.in_prompt.chance",
            "groups.in_prompt.score",
            "groups.in_prompt.identification",
            "groups.in_prompt.bootstrap",
            "groups.held_out.score",
            "groups.held_out.identification",
            "groups.held_out.bootstrap",
        ),
        "both runs score the same qwen38_27b train profiles and the same library links (the link"
        " rule reads no UDV evidence); only the UDV tiers and propositions per run differ",
    ),
    Artifact(
        "profile_generation_dry_run",
        {"udv_v1": None, "udv_v2": "artifacts/actor_profiles/udv_v2_dry_run_profiles.json"},
        (
            "prompt_version",
            "inputs",
            "actors",
            "actors_sha256",
            "rendered_prompts_sha256",
            "user_prompt_chars",
        ),
        "profile prompts are built from the train speech file and the LDS only; the train speech"
        " file of the udv_v2 run has the same sha256 as the udv_v1 run, so the prompts and the"
        " qwen38_27b profiles do not depend on the UDV run",
    ),
    Artifact(
        "simulation_evaluation_dry_run",
        {
            "udv_v1": None,
            "udv_v2": "artifacts/actor_simulation/udv_v2_dry_run_evaluation.json",
        },
        ("prompt_version", "inputs", "splits"),
        "the udv_v1 counts are the evaluation.json counts of artifacts/mlx_runs/qwen38_27b; see"
        " question_overlap for the per question comparison",
    ),
    Artifact(
        "simulation_evaluation_model_run",
        {
            "udv_v1": "artifacts/mlx_runs/qwen38_27b/actor_simulation/evaluation.json",
            "udv_v2": None,
        },
        (
            "model",
            "selection.counts",
            "selection.k",
            "selection.guidance_scale",
            "evaluation.counts",
            "evaluation.conditions",
            "evaluation.differences",
        ),
        "the udv_v2 model run is the colleague command below; its runs_dir is"
        " artifacts/mlx_runs/udv_v2",
    ),
    Artifact(
        "simulation_requests_dry_run",
        {
            "udv_v1": None,
            "udv_v2": "artifacts/actor_simulation/udv_v2_dry_run_simulation.json",
        },
        (
            "prompt_version",
            "inputs",
            "requests",
            "request_actors",
            "request_hearings",
            "requests_sha256",
            "roles_sha256",
            "examples_sha256",
            "actors_without_role",
        ),
        "the same dry run on udv_v1 gives the same requests_sha256, roles_sha256 and"
        " examples_sha256",
    ),
    Artifact(
        "web_export",
        {"udv_v1": None, "udv_v2": "artifacts/web/export_site_udv_v2.json"},
        ("run", "hearings", "udvs", "bytes", "output", "profiles"),
        "summary printed by bookworm export-site on udv_v2 with --verifier-report and the"
        " qwen38_27b train profiles; the export checks the run against configs/udv_v2.toml",
    ),
    Artifact(
        "human_precision_interim",
        {
            "udv_v1": None,
            "udv_v2": "artifacts/udv/udv_v2_precision_interim_20260928T173107Z.json",
        },
        (
            "status",
            "judged",
            "annotation_csv",
            "udv_v1.strata",
            "udv_v1.criteria",
            "udv_v2_unchanged.items_in_sample",
            "udv_v2_unchanged.strata",
            "udv_v2_unchanged.criteria",
        ),
        "INTERIM: the udv_v1 sheet was read only, with 65 of 127 items judged; udv_v2 precision"
        " on the changed items needs the supplement sheet",
    ),
)

SUPPLEMENT = {
    "path": "artifacts/validation/human_validation_v1_udv_v2_supplement/annotation.csv",
    "key": "artifacts/validation/human_validation_v1_udv_v2_supplement/annotation_key.json",
    "score_command": (
        "uv run python -m experiments.udv.v2_analysis score-annotation --final-test --annotation"
        " artifacts/validation/human_validation_v1_udv_v1/annotation.csv --supplement-dir"
        " artifacts/validation/human_validation_v1_udv_v2_supplement"
    ),
}
COLLEAGUE_COMMAND = {
    "cwd": "experiments",
    "download": (
        'uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run download --model'
        " qwen38_27b --settings configs/mlx_udv_v2.yaml"
    ),
    "run": (
        'uv run --with "mlx-lm==0.31.3" python -m experiments.mlx.run all --model qwen38_27b'
        " --all-actors --settings configs/mlx_udv_v2.yaml"
    ),
    "model": "mlx-community/Qwen3.8-27B-8bit",
    "revision": None,
    "revision_note": "no local copy of the model was available, so no revision is pinned",
}
NOT_RUN = (
    {
        "stage": "profile generation and actor simulation with qwen38_27b",
        "command": COLLEAGUE_COMMAND["run"],
        "reason": "the 27B MLX model is run on another machine",
    },
    {
        "stage": "udv_v1 actor links file",
        "command": "bookworm build-udvs --config configs/udv.toml --actors-config ... --overwrite",
        "reason": (
            "rebuilding would overwrite udv_v1 artifacts; the udv_v1 profile validation uses the"
            " udv_v2 links file, whose links are equal to the recomputed library links"
        ),
    },
)


def supplement_section() -> Record:
    with open(SUPPLEMENT["key"]) as f:
        key = json.load(f)
    return {
        **SUPPLEMENT,
        "rows_to_judge": len(key["items"]),
        "inherited_items": len(key["inherited_items"]),
        "relation_counts": key["relation_counts"],
    }


def code_hashes() -> dict[str, str]:
    modules = (simulation, evaluate_simulation)
    return source_hashes(Path(__file__), *modules)


def build_report(profiles_path: Path) -> Record:
    with open(MANIFEST) as f:
        hearing_split = split_of(json.load(f))
    owners: dict[tuple[int, int], str] = {}
    for speech_file in SPEECH_FILES:
        owners.update(simulation.turn_owners(load_jsonl(speech_file)))
    profiled = {record["actor"] for record in load_jsonl(TRAIN_SPEECHES)}
    library = library_links(LINKS_V2)
    runs: Record = {}
    questions: dict[str, dict[str, list[Record]]] = {}
    for run, path in RUNS.items():
        udvs = {udv["id"]: udv for udv in load_jsonl(path)}
        evidence = evidence_turn_links(udvs.values(), owners)
        runs[run] = {
            "path": str(path),
            "sha256": sha256_of_file(path),
            "split_tier_counts": split_tier_counts(udvs.values(), hearing_split),
            "link_rules": link_comparison(udvs, library, evidence, profiled, hearing_split),
        }
        questions[run] = simulation_questions(path, profiles_path)
    return {
        "name": "udv_v2_downstream",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": (
            "every stage downstream of the UDV run, rerun or checked on udv_v2, next to its"
            " udv_v1 value"
        ),
        "inputs": {
            "split_manifest": {"path": str(MANIFEST), "sha256": sha256_of_file(MANIFEST)},
            "speech_files": {str(path): sha256_of_file(path) for path in SPEECH_FILES},
            "train_speeches": {
                "path": str(TRAIN_SPEECHES),
                "sha256": sha256_of_file(TRAIN_SPEECHES),
            },
            "profiles": {"path": str(profiles_path), "sha256": sha256_of_file(profiles_path)},
        },
        "link_rules": {
            "library_rule": LIBRARY_RULE,
            "evidence_turn_rule": EVIDENCE_RULE,
            "profiled_rule": PROFILED_RULE,
            "profiled_actors": len(profiled),
            "library_links_file": {"path": str(LINKS_V2), "sha256": sha256_of_file(LINKS_V2)},
        },
        "runs": runs,
        "question_overlap": question_overlap(questions["udv_v1"], questions["udv_v2"]),
        "results": {artifact.name: artifact_section(artifact) for artifact in ARTIFACTS},
        "supplement_annotation": supplement_section(),
        "colleague_command": COLLEAGUE_COMMAND,
        "not_run": list(NOT_RUN),
        "code": code_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare every stage downstream of the UDV run on udv_v1 and udv_v2."
    )
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build_report(args.profiles)
    write_json(report, args.output)
    summary = {
        run: {
            "link_rules": {
                key: value
                for key, value in section["link_rules"].items()
                if key not in {"library_only", "evidence_turn_only", "disagreement_examples"}
            },
        }
        for run, section in report["runs"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
