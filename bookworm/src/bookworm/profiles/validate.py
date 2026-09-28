"""Validation of actor profiles against the verified UDVs of each actor."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from bookworm.actors.schemas import UdvActorLink, read_udv_actor_links
from bookworm.config import EncoderSettings, TfidfSettings
from bookworm.data.io import (
    JsonObject,
    is_json_integer_list,
    read_json_object,
    sha256_of_file,
    write_json,
    write_jsonl,
)
from bookworm.data.splits import SPLIT_NAMES, SplitName
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, SentenceEncoder
from bookworm.features.loading import load_encoder
from bookworm.profiles.config import (
    ProfileValidationConfig,
    ReviewSettings,
    load_profile_validation_config,
)
from bookworm.profiles.schemas import (
    GROUPS,
    Group,
    ProfilePair,
    ProfileRecord,
    read_pairs,
    read_profiles,
)
from bookworm.profiles.validation_metrics import (
    ProfileScores,
    chance_mrr,
    group_metrics,
    identification_rank,
    round_value,
    rounded,
    score_profiles,
)
from bookworm.transcript.sentences import split_sentences
from bookworm.udv.schemas import UdvRecord, read_udv_run

__all__ = [
    "GROUPS",
    "SKIP_REASONS",
    "Group",
    "ProfilePair",
    "ProfileValidationConfig",
    "ReviewSettings",
    "SplitManifest",
    "chance_mrr",
    "check_held_out_pair",
    "check_new_outputs",
    "check_profile",
    "identification_rank",
    "load_profile_validation_config",
    "profile_sentences",
    "profiles_by_actor",
    "read_pairs",
    "rounded",
    "run_validate_profiles",
    "score_profiles",
    "validate_profiles",
]

SkipReason = Literal[
    "person_not_resolved",
    "tier_excluded",
    "no_link",
    "no_actor",
    "actor_without_profile",
    "split_not_evaluated",
    "not_in_prompt",
]
ProfileEncoderFactory = Callable[[EncoderSettings, Sequence[str], int], SentenceEncoder]

SKIP_REASONS: tuple[SkipReason, ...] = get_args(SkipReason)
PROPOSITION_CACHE_LABEL = "profile_validation_propositions"
SENTENCE_CACHE_LABEL = "profile_validation_sentences"
SENTENCE_SEGMENTATION = (
    "each non-empty line of the profile split by bookworm.transcript.sentences.split_sentences"
)
PAIR_SCORE = "max cosine between the UDV proposition and the sentences of one profile"
RANK_RULE = (
    "1 + number of other profiles whose score is greater than or equal to the true profile "
    "score (ties count against the true profile)"
)
TFIDF_CORPUS = "sentences of every profile; propositions are never part of the fitted corpus"
SCORE_NOTE = (
    "cosine measures content proximity, not entailment; whether a profile sentence supports "
    "the opinion is decided by the human review"
)


@dataclass(frozen=True)
class SplitManifest:
    split_version: str
    split_by_hearing: dict[int, SplitName]


def read_split_manifest(path: Path) -> SplitManifest:
    payload = read_json_object(path, "split manifest")
    if not isinstance(payload.get("split_version"), str):
        raise ConfigError(f"{path}: split manifest has no split_version")
    split_by_hearing: dict[int, SplitName] = {}
    for name in SPLIT_NAMES:
        ids = payload.get(name)
        if not is_json_integer_list(ids):
            raise ConfigError(f"{path}: {name} is missing or not a list of integers")
        for hearing_id in ids:
            if hearing_id in split_by_hearing:
                raise ConfigError(f"{path}: hearing {hearing_id} is listed in two splits")
            split_by_hearing[hearing_id] = name
    return SplitManifest(payload["split_version"], split_by_hearing)


def check_profile(
    profile: ProfileRecord, manifest: SplitManifest, generation_splits: Sequence[SplitName]
) -> None:
    for hearing_id in profile.hearing_ids:
        split = manifest.split_by_hearing.get(hearing_id)
        if split is None:
            raise ConfigError(
                f"profile {profile.actor!r}: hearing {hearing_id} is in no split of the manifest"
            )
        if split not in generation_splits:
            raise ConfigError(
                f"profile {profile.actor!r}: hearing {hearing_id} of split {split!r} entered the "
                f"prompt, outside the generation splits {list(generation_splits)}"
            )


def profiles_by_actor(profiles: Sequence[ProfileRecord]) -> dict[str, ProfileRecord]:
    counts = Counter(profile.actor for profile in profiles)
    repeated = sorted(actor for actor, count in counts.items() if count > 1)
    if repeated:
        raise ConfigError(f"actors with more than one profile: {repeated}")
    return {profile.actor: profile for profile in profiles}


def check_held_out_pair(udv: UdvRecord, profile: ProfileRecord) -> None:
    if udv.hearing_id in profile.hearing_ids:
        raise ConfigError(
            f"{udv.id}: hearing {udv.hearing_id} is held out but entered the prompt of "
            f"profile {profile.actor!r}"
        )


def profile_sentences(profile: str) -> list[str]:
    return [
        sentence
        for line in profile.splitlines()
        if line.strip()
        for sentence in split_sentences(line)
    ]


@dataclass(frozen=True)
class ProfileText:
    record: ProfileRecord
    sentences: list[str]


def segment_profiles(profiles: Sequence[ProfileRecord]) -> list[ProfileText]:
    texts = [ProfileText(profile, profile_sentences(profile.profile)) for profile in profiles]
    empty = [text.record.actor for text in texts if not text.sentences]
    if empty:
        raise ConfigError(f"profiles without any sentence to compare: {empty}")
    return texts


@dataclass(frozen=True)
class Candidate:
    udv: UdvRecord
    link: UdvActorLink
    split: SplitName
    group: Group


@dataclass
class Selection:
    candidates: list[Candidate]
    skipped: Counter[SkipReason]


def select_candidates(
    udvs: Sequence[UdvRecord],
    links: Mapping[str, UdvActorLink],
    profiles: Mapping[str, ProfileRecord],
    manifest: SplitManifest,
    config: ProfileValidationConfig,
) -> Selection:
    selection = Selection([], Counter())
    for udv in udvs:
        reason, candidate = classify_udv(udv, links, profiles, manifest, config)
        if candidate is not None:
            selection.candidates.append(candidate)
        elif reason is not None:
            selection.skipped[reason] += 1
    return selection


def classify_udv(
    udv: UdvRecord,
    links: Mapping[str, UdvActorLink],
    profiles: Mapping[str, ProfileRecord],
    manifest: SplitManifest,
    config: ProfileValidationConfig,
) -> tuple[SkipReason | None, Candidate | None]:
    if udv.tier == "person_not_resolved":
        return "person_not_resolved", None
    if udv.tier not in config.tiers:
        return "tier_excluded", None
    link = links.get(udv.id)
    if link is None:
        return "no_link", None
    if link.actor is None:
        return "no_actor", None
    profile = profiles.get(link.actor)
    if profile is None:
        return "actor_without_profile", None
    split = manifest.split_by_hearing.get(udv.hearing_id)
    if split is None:
        raise ConfigError(f"{udv.id}: hearing {udv.hearing_id} is in no split of the manifest")
    if split in config.held_out_splits:
        check_held_out_pair(udv, profile)
        return None, Candidate(udv, link, split, "held_out")
    if split not in config.generation_splits:
        return "split_not_evaluated", None
    if udv.hearing_id not in profile.hearing_ids:
        return "not_in_prompt", None
    return None, Candidate(udv, link, split, "in_prompt")


def default_profile_encoder(
    settings: EncoderSettings, corpus: Sequence[str], seed: int
) -> SentenceEncoder:
    return load_encoder(settings, seed, lambda: corpus)


def build_pairs(
    candidates: Sequence[Candidate],
    texts: Sequence[ProfileText],
    scores: ProfileScores,
) -> list[ProfilePair]:
    column_by_actor = {text.record.actor: column for column, text in enumerate(texts)}
    pairs: list[ProfilePair] = []
    for row, candidate in enumerate(candidates):
        column = column_by_actor[str(candidate.link.actor)]
        text = texts[column]
        row_scores = scores.scores[row]
        sentence_index = int(scores.best_sentence[row, column])
        others = [index for index in range(len(texts)) if index != column]
        best_other = max(others, key=lambda index: row_scores[index], default=None)
        udv = candidate.udv
        pairs.append(
            ProfilePair(
                udv_id=udv.id,
                hearing_id=udv.hearing_id,
                split=candidate.split,
                group=candidate.group,
                tier=udv.tier,
                actor_key=candidate.link.actor_key,
                actor=text.record.actor,
                udv_actor=udv.actor.name,
                proposition=udv.proposition,
                udv_evidence=None if udv.evidence is None else udv.evidence.text,
                profile_sentence=text.sentences[sentence_index],
                profile_sentence_index=sentence_index,
                score=round_value(row_scores[column]),
                rank=identification_rank(row_scores, column),
                n_candidates=len(texts),
                best_other_actor=None if best_other is None else texts[best_other].record.actor,
                best_other_score=None if best_other is None else rounded(row_scores[best_other]),
            )
        )
    return pairs


def input_digest(path: Path) -> JsonObject:
    return {"path": str(path), "sha256": sha256_of_file(path)}


def check_new_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ConfigError(f"{existing[0]}: output already exists; pass --overwrite to replace it")


@dataclass(frozen=True)
class ValidationInputs:
    udvs: list[UdvRecord]
    links: dict[str, UdvActorLink]
    profiles: list[ProfileRecord]
    manifest: SplitManifest


def read_links(path: Path) -> dict[str, UdvActorLink]:
    try:
        links = read_udv_actor_links(path)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: actor links file not found") from error
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read actor links: {error}") from error
    by_udv = {link.udv_id: link for link in links}
    if len(by_udv) != len(links):
        raise ConfigError(f"{path}: a UDV id is linked more than once")
    return by_udv


def load_inputs(config: ProfileValidationConfig) -> ValidationInputs:
    if not config.profiles_path.is_file():
        raise ConfigError(f"{config.profiles_path}: profiles file not found")
    return ValidationInputs(
        udvs=read_udv_run(config.udv_path, "UDV run"),
        links=read_links(config.links_path),
        profiles=read_profiles(config.profiles_path),
        manifest=read_split_manifest(config.split_manifest),
    )


def validate_profiles(
    config: ProfileValidationConfig,
    *,
    encoder_factory: ProfileEncoderFactory = default_profile_encoder,
    overwrite: bool = False,
    created_at: datetime | None = None,
) -> JsonObject:
    """Score each verified UDV against every profile and write the pairs and report."""
    check_new_outputs((config.pairs_path, config.report_path), overwrite)
    inputs = load_inputs(config)
    if not inputs.profiles:
        raise ConfigError(f"{config.profiles_path}: no profiles to validate")
    for profile in inputs.profiles:
        check_profile(profile, inputs.manifest, config.generation_splits)
    texts = segment_profiles(inputs.profiles)
    selection = select_candidates(
        inputs.udvs,
        inputs.links,
        profiles_by_actor(inputs.profiles),
        inputs.manifest,
        config,
    )
    corpus = [sentence for text in texts for sentence in text.sentences]
    encoder = encoder_factory(config.encoder, corpus, config.seed)
    cached = CachedEncoder(encoder, config.cache_dir)
    propositions = [candidate.udv.proposition for candidate in selection.candidates]
    scores = score_profiles(
        cached.encode(propositions, PROPOSITION_CACHE_LABEL),
        cached.encode(corpus, SENTENCE_CACHE_LABEL),
        [len(text.sentences) for text in texts],
    )
    pairs = build_pairs(selection.candidates, texts, scores)
    report = build_report(config, inputs, texts, selection, pairs, encoder, created_at)
    write_jsonl((pair.to_dict() for pair in pairs), config.pairs_path)
    write_json(report, config.report_path)
    return report


def build_report(
    config: ProfileValidationConfig,
    inputs: ValidationInputs,
    texts: Sequence[ProfileText],
    selection: Selection,
    pairs: Sequence[ProfilePair],
    encoder: SentenceEncoder,
    created_at: datetime | None,
) -> JsonObject:
    moment = datetime.now(UTC) if created_at is None else created_at
    n_candidates = len(texts)
    groups = {
        group: group_metrics(
            [pair for pair in pairs if pair.group == group],
            n_candidates,
            config,
            np.random.default_rng([config.seed, index]),
        )
        for index, group in enumerate(GROUPS)
    }
    return {
        "name": config.name,
        "created_at": moment.isoformat(timespec="seconds"),
        "config": config.source,
        "inputs": {
            "udvs": input_digest(config.udv_path),
            "links": input_digest(config.links_path),
            "profiles": input_digest(config.profiles_path),
            "split_manifest": input_digest(config.split_manifest),
        },
        "split_version": inputs.manifest.split_version,
        "encoder": {
            "name": encoder.name,
            "revision": encoder.revision,
            "cache_identity": encoder.cache_identity,
            "runtime": encoder.runtime_info(),
        },
        "method": {
            "tiers": list(config.tiers),
            "generation_splits": list(config.generation_splits),
            "held_out_splits": list(config.held_out_splits),
            "sentence_segmentation": SENTENCE_SEGMENTATION,
            "pair_score": PAIR_SCORE,
            "rank": RANK_RULE,
            "tfidf_corpus": TFIDF_CORPUS if isinstance(config.encoder, TfidfSettings) else None,
            "note": SCORE_NOTE,
        },
        "counts": {
            "udvs": len(inputs.udvs),
            "profiles": n_candidates,
            "profile_sentences": sum(len(text.sentences) for text in texts),
            "skipped": {reason: selection.skipped[reason] for reason in SKIP_REASONS},
            "pairs": {group: groups[group]["pairs"] for group in GROUPS},
        },
        "groups": groups,
    }


def run_validate_profiles(config_path: Path, *, overwrite: bool = False) -> JsonObject:
    """Load a profile validation config and run ``validate_profiles``."""
    config = load_profile_validation_config(config_path)
    report = validate_profiles(config, overwrite=overwrite)
    return {
        "name": config.name,
        "counts": report["counts"],
        "groups": {
            group: {
                key: report["groups"][group][key]
                for key in ("pairs", "hearings", "actors", "identification", "chance")
            }
            for group in GROUPS
        },
        "pairs": str(config.pairs_path),
        "report": str(config.report_path),
    }
