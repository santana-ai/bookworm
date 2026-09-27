import copy
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self, get_args

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bookworm.actors.schemas import UdvActorLink, read_udv_actor_links
from bookworm.config import (
    DEFAULT_ENCODER_KIND,
    SPLIT_VERSION_PATTERN,
    EncoderSettings,
    TfidfSettings,
    read_toml,
    required,
    section,
)
from bookworm.data.io import is_json_integer_list, sha256_of_file, write_json, write_jsonl
from bookworm.data.splits import SPLIT_NAMES, SplitName
from bookworm.errors import ConfigError
from bookworm.features.encoders import CachedEncoder, FloatMatrix, SentenceEncoder
from bookworm.features.tfidf import TfidfEncoder
from bookworm.profiles.schemas import ProfileRecord, read_profiles
from bookworm.transcript.sentences import split_sentences
from bookworm.udv.schemas import Tier, UdvRecord, load_udv_jsonl

Group = Literal["in_prompt", "held_out"]
SkipReason = Literal[
    "person_not_resolved",
    "tier_excluded",
    "no_link",
    "no_actor",
    "actor_without_profile",
    "split_not_evaluated",
    "not_in_prompt",
]
JsonObject = dict[str, Any]
ProfileEncoderFactory = Callable[[EncoderSettings, Sequence[str], int], SentenceEncoder]

GROUPS: tuple[Group, ...] = get_args(Group)
SKIP_REASONS: tuple[SkipReason, ...] = get_args(SkipReason)
DEFAULT_TIERS: tuple[Tier, ...] = ("quote_found", "semantic_match_high")
DEFAULT_GENERATION_SPLITS: tuple[SplitName, ...] = ("train",)
DEFAULT_HELD_OUT_SPLITS: tuple[SplitName, ...] = ("test",)
DEFAULT_CONFIDENCE_LEVEL = 0.95
PROPOSITION_CACHE_LABEL = "profile_validation_propositions"
SENTENCE_CACHE_LABEL = "profile_validation_sentences"
ROUND_DECIMALS = 4
SENTENCE_SEGMENTATION = (
    "each non-empty line of the profile split by bookworm.transcript.sentences.split_sentences"
)
PAIR_SCORE = "max cosine between the UDV proposition and the sentences of one profile"
RANK_RULE = (
    "1 + number of other profiles whose score is greater than or equal to the true profile "
    "score (ties count against the true profile)"
)
TFIDF_CORPUS = "sentences of every profile; propositions are never part of the fitted corpus"
BOOTSTRAP_INTERVAL = "percentile interval over replicates that resample whole hearings"
SCORE_NOTE = (
    "cosine measures content proximity, not entailment; whether a profile sentence supports "
    "the opinion is decided by the human review"
)


class _ValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReviewSettings(_ValidationModel):
    seed: int
    sizes: dict[Group, int]
    score_bands: list[float]
    output: Path

    @model_validator(mode="after")
    def check_review(self) -> Self:
        if any(size < 0 for size in self.sizes.values()):
            raise ValueError("review sizes must be >= 0")
        if any(high <= low for low, high in pairwise(self.score_bands)):
            raise ValueError("score_bands must be strictly increasing")
        return self


class ProfileValidationConfig(_ValidationModel):
    udv_path: Path
    links_path: Path
    profiles_path: Path
    split_manifest: Path
    tiers: list[Tier] = Field(min_length=1)
    generation_splits: list[SplitName] = Field(min_length=1)
    held_out_splits: list[SplitName] = Field(min_length=1)
    encoder: EncoderSettings
    cache_dir: Path
    output_dir: Path
    name: str = Field(pattern=SPLIT_VERSION_PATTERN)
    seed: int
    bootstrap_samples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    review: ReviewSettings
    source: dict[str, Any]

    @model_validator(mode="after")
    def check_splits(self) -> Self:
        for name, values in (
            ("tiers", self.tiers),
            ("generation_splits", self.generation_splits),
            ("held_out_splits", self.held_out_splits),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} has repeated values: {values}")
        shared = sorted(set(self.generation_splits) & set(self.held_out_splits))
        if shared:
            raise ValueError(f"splits {shared} are both generation and held-out splits")
        return self

    @property
    def pairs_path(self) -> Path:
        return self.output_dir / f"{self.name}_pairs.jsonl"

    @property
    def report_path(self) -> Path:
        return self.output_dir / f"{self.name}_report.json"

    @property
    def review_sample_path(self) -> Path:
        output = self.review.output
        return output.with_name(f"{output.stem}_sample.json")

    @property
    def review_report_path(self) -> Path:
        output = self.review.output
        return output.with_name(f"{output.stem}_report.json")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], origin: str = "<mapping>"
    ) -> "ProfileValidationConfig":
        source = copy.deepcopy(dict(raw))
        validation = section(source, "validation", origin)
        encoder = dict(section(validation, "encoder", f"{origin} [validation]"))
        encoder.setdefault("kind", DEFAULT_ENCODER_KIND)
        try:
            review = section(source, "review", origin)
            return cls.model_validate(
                {
                    "udv_path": Path(required(source, "inputs", "udv_path", origin)),
                    "links_path": Path(required(source, "inputs", "links_path", origin)),
                    "profiles_path": Path(required(source, "inputs", "profiles_path", origin)),
                    "split_manifest": Path(required(source, "inputs", "split_manifest", origin)),
                    "tiers": list(validation.get("tiers", DEFAULT_TIERS)),
                    "generation_splits": list(
                        validation.get("generation_splits", DEFAULT_GENERATION_SPLITS)
                    ),
                    "held_out_splits": list(
                        validation.get("held_out_splits", DEFAULT_HELD_OUT_SPLITS)
                    ),
                    "encoder": encoder,
                    "cache_dir": Path(required(source, "validation", "cache_dir", origin)),
                    "output_dir": Path(required(source, "validation", "output_dir", origin)),
                    "name": required(source, "validation", "name", origin),
                    "seed": required(source, "validation", "seed", origin),
                    "bootstrap_samples": required(
                        source, "validation", "bootstrap_samples", origin
                    ),
                    "confidence_level": float(
                        validation.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)
                    ),
                    "review": {
                        "seed": required(source, "review", "seed", origin),
                        "sizes": dict(required(source, "review", "sizes", origin)),
                        "score_bands": [float(edge) for edge in review.get("score_bands", [])],
                        "output": Path(required(source, "review", "output", origin)),
                    },
                    "source": source,
                }
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise ConfigError(f"{origin}: {error}") from error


def load_profile_validation_config(path: Path) -> ProfileValidationConfig:
    return ProfileValidationConfig.from_mapping(read_toml(path), origin=str(path))


class ProfilePair(_ValidationModel):
    udv_id: str
    hearing_id: int
    split: SplitName
    group: Group
    tier: Tier
    actor_key: str | None
    actor: str
    udv_actor: str
    proposition: str
    udv_evidence: str | None
    profile_sentence: str
    profile_sentence_index: int
    score: float
    rank: int
    n_candidates: int
    best_other_actor: str | None
    best_other_score: float | None

    @classmethod
    def from_json_line(cls, line: str) -> "ProfilePair":
        return cls.model_validate_json(line)

    def to_dict(self) -> JsonObject:
        return self.model_dump()


def read_pairs(path: Path) -> list[ProfilePair]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [ProfilePair.from_json_line(line) for line in handle if line.strip()]
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: pairs file not found; run validate-profiles first") from error
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read pairs file: {error}") from error


@dataclass(frozen=True)
class SplitManifest:
    split_version: str
    split_by_hearing: dict[int, SplitName]


def read_split_manifest(path: Path) -> SplitManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: split manifest not found") from error
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read split manifest: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("split_version"), str):
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


def unit_rows(matrix: FloatMatrix) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


@dataclass(frozen=True)
class ProfileScores:
    scores: NDArray[np.float64]
    best_sentence: NDArray[np.int64]


def score_profiles(
    propositions: FloatMatrix, sentences: FloatMatrix, sentence_counts: Sequence[int]
) -> ProfileScores:
    similarities = unit_rows(propositions) @ unit_rows(sentences).T
    bounds = np.cumsum([0, *sentence_counts])
    scores = np.empty((similarities.shape[0], len(sentence_counts)), dtype=np.float64)
    best = np.empty_like(scores, dtype=np.int64)
    for column, (start, end) in enumerate(pairwise(bounds)):
        block = similarities[:, start:end]
        best[:, column] = block.argmax(axis=1)
        scores[:, column] = block.max(axis=1)
    return ProfileScores(scores, best)


def identification_rank(scores: NDArray[np.float64], true_index: int) -> int:
    others = np.delete(scores, true_index)
    return 1 + int(np.count_nonzero(others >= scores[true_index]))


def chance_mrr(n_candidates: int) -> float:
    return sum(1 / rank for rank in range(1, n_candidates + 1)) / n_candidates


def default_profile_encoder(
    settings: EncoderSettings, corpus: Sequence[str], seed: int
) -> SentenceEncoder:
    if isinstance(settings, TfidfSettings):
        return TfidfEncoder.fit(corpus, max_features=settings.max_features)
    try:
        from bookworm.features import sentence_transformer
    except ModuleNotFoundError as error:
        raise ConfigError(
            f"encoder kind {settings.kind!r} needs the optional 'embeddings' extra: {error}"
        ) from error
    sentence_transformer.seed_torch(seed)
    return sentence_transformer.SentenceTransformerEncoder(
        settings.name, settings.revision, settings.device, settings.batch_size
    )


def round_value(value: float) -> float:
    return round(float(value), ROUND_DECIMALS)


def rounded(value: float | None) -> float | None:
    return None if value is None else round_value(value)


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


def percentile_interval(values: NDArray[np.float64], confidence_level: float) -> list[float]:
    alpha = 1 - confidence_level
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return [round_value(low), round_value(high)]


def bootstrap_by_hearing(
    pairs: Sequence[ProfilePair],
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> JsonObject:
    hearings = sorted({pair.hearing_id for pair in pairs})
    position = {hearing_id: index for index, hearing_id in enumerate(hearings)}
    counts = np.zeros(len(hearings))
    sums = np.zeros((3, len(hearings)))
    for pair in pairs:
        index = position[pair.hearing_id]
        counts[index] += 1
        sums[:, index] += (pair.score, pair.rank == 1, 1 / pair.rank)
    weights = np.stack(
        [
            np.bincount(rng.integers(0, len(hearings), len(hearings)), minlength=len(hearings))
            for _ in range(samples)
        ]
    ).astype(np.float64)
    replicates = (weights @ sums.T) / (weights @ counts)[:, None]
    return {
        "unit": "hearing",
        "method": BOOTSTRAP_INTERVAL,
        "samples": samples,
        "confidence_level": confidence_level,
        "mean_score": percentile_interval(replicates[:, 0], confidence_level),
        "acc_at_1": percentile_interval(replicates[:, 1], confidence_level),
        "mrr": percentile_interval(replicates[:, 2], confidence_level),
    }


def group_metrics(
    pairs: Sequence[ProfilePair],
    n_candidates: int,
    config: ProfileValidationConfig,
    rng: np.random.Generator,
) -> JsonObject:
    chance = {"acc_at_1": rounded(1 / n_candidates), "mrr": rounded(chance_mrr(n_candidates))}
    summary: JsonObject = {
        "pairs": len(pairs),
        "hearings": len({pair.hearing_id for pair in pairs}),
        "actors": len({pair.actor for pair in pairs}),
        "splits": sorted({pair.split for pair in pairs}),
        "n_candidates": n_candidates,
        "chance": chance,
    }
    if not pairs:
        return {**summary, "score": None, "identification": None, "bootstrap": None}
    scores = np.array([pair.score for pair in pairs])
    ranks = np.array([pair.rank for pair in pairs])
    q25, median, q75 = np.quantile(scores, [0.25, 0.5, 0.75])
    return {
        **summary,
        "score": {
            "mean": rounded(scores.mean()),
            "median": rounded(median),
            "q25": rounded(q25),
            "q75": rounded(q75),
            "min": rounded(scores.min()),
            "max": rounded(scores.max()),
        },
        "identification": {
            "acc_at_1": rounded(float(np.mean(ranks == 1))),
            "mrr": rounded(float(np.mean(1 / ranks))),
        },
        "bootstrap": bootstrap_by_hearing(
            pairs, config.bootstrap_samples, config.confidence_level, rng
        ),
    }


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


def read_udvs(path: Path) -> list[UdvRecord]:
    try:
        return load_udv_jsonl(path)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: UDV run file not found") from error
    except (OSError, ValueError) as error:
        raise ConfigError(f"{path}: cannot read UDV run: {error}") from error


def load_inputs(config: ProfileValidationConfig) -> ValidationInputs:
    if not config.profiles_path.is_file():
        raise ConfigError(f"{config.profiles_path}: profiles file not found")
    return ValidationInputs(
        udvs=read_udvs(config.udv_path),
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
