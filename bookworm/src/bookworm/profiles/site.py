"""Actor pages of the demo: ``actors.json`` and one ``profiles/<slug>.json`` per profiled actor."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bookworm.actors.schemas import ActorHearing, ActorSpeechRecord
from bookworm.data.io import JsonObject, write_json
from bookworm.errors import ConfigError
from bookworm.profiles.claim_matching import (
    MATCH_METHOD,
    PASSAGE_MATCH_MIN,
    PASSAGE_MIN_WORDS,
    SAME_SENTENCE_RULE,
    SIMILAR_TEXT_RULE,
    UDV_MATCH_MIN,
    MatchThresholds,
    Sentence,
    match_claims,
)
from bookworm.profiles.profile_text import (
    UNTITLED_SECTION,
    ProfileSection,
    actor_slug,
    assign_slugs,
    parse_profile,
    trim_passage,
)
from bookworm.profiles.schemas import ProfileRecord
from bookworm.udv.export import sentence_text

__all__ = [
    "ACTORS_FILE_NAME",
    "PROFILES_DIR_NAME",
    "SAME_SENTENCE_RULE",
    "SIMILAR_TEXT_RULE",
    "UNTITLED_SECTION",
    "ProfileSection",
    "ProfileSiteBuilder",
    "ProfileSiteExport",
    "actor_slug",
    "assign_slugs",
    "clear_profile_site",
    "display_name_of",
    "page_verifier_cuts",
    "page_verifier_threshold",
    "parse_profile",
    "trim_passage",
    "verifier_of",
]

ACTORS_FILE_NAME = "actors.json"
PROFILES_DIR_NAME = "profiles"
PAGE_HEARING_KEYS = ("split", "article_date", "title", "assunto")


@dataclass
class ActorTrail:
    turns_by_hearing: dict[int, int] = field(default_factory=dict)
    sentences: list[Sentence] = field(default_factory=list)
    udvs: list[JsonObject] = field(default_factory=list)
    roles: Counter[str] = field(default_factory=Counter)
    article_names: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class ProfileSiteExport:
    actors: JsonObject
    actors_bytes: int
    profile_bytes: dict[str, int]

    @property
    def total_bytes(self) -> int:
        return self.actors_bytes + sum(self.profile_bytes.values())


def most_common(counts: Counter[str]) -> str | None:
    """Most frequent key, the smallest one on ties."""
    if not counts:
        return None
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def display_name_of(article_names: Counter[str], fallback: str) -> str:
    """Most frequent article name, the longest one on ties, then the smallest."""
    if not article_names:
        return fallback
    return min(article_names.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0]


def majority_owner(turns: Sequence[int], owners: Mapping[int, str]) -> str | None:
    return most_common(Counter(owners[turn] for turn in turns if turn in owners))


def verifier_of(udv: Mapping[str, Any]) -> JsonObject | None:
    signals = udv.get("signals")
    if not isinstance(signals, Mapping) or signals.get("verifier") is None:
        return None
    verifier = signals["verifier"]
    supported = verifier.get("supported_at_udv_threshold", verifier["supported"])
    return {"probability": verifier["probability"], "supported": supported}


def page_verifier_threshold(signals: Mapping[str, Any]) -> float:
    verifier = signals["verifier"]
    udv_threshold = verifier.get("udv_threshold")
    if isinstance(udv_threshold, Mapping):
        return float(udv_threshold["value"])
    return float(verifier["threshold"])


def page_verifier_cuts(signals: Mapping[str, Any]) -> JsonObject:
    """Cuts of the support bands: below ``low`` weak, from ``high`` on strong."""
    verifier = signals["verifier"]
    udv_threshold = verifier.get("udv_threshold")
    if isinstance(udv_threshold, Mapping):
        return {"low": float(udv_threshold["value"]), "high": float(verifier["threshold"])}
    return {"low": float(verifier["threshold"]), "high": None}


def evidence_of(udv: Mapping[str, Any]) -> JsonObject | None:
    evidence = udv["evidence"]
    if evidence is None:
        return None
    return {
        "text": trim_passage(evidence["text"]),
        "turn": evidence["speaker_turn"],
        "start": evidence["start_char"],
        "end": evidence["end_char"],
    }


def udv_text(udv: Mapping[str, Any]) -> str:
    evidence = udv["evidence"]
    return udv["proposition"] if evidence is None else f"{udv['proposition']} {evidence['text']}"


def trail_udv(udv: Mapping[str, Any], hearing_id: int, number: int) -> JsonObject:
    return {
        "id": udv["id"],
        "hearing_id": hearing_id,
        "n": number,
        "article_name": udv["actor"]["name"],
        "proposition": udv["proposition"],
        "tier": udv["tier"],
        "evidence": evidence_of(udv),
        "verifier": verifier_of(udv),
        "match_text": udv_text(udv),
    }


def turn_sentences(hearing_id: int, turn: Mapping[str, Any], transcript: str) -> list[Sentence]:
    return [
        Sentence(hearing_id, turn["index"], entry[0], entry[1], text)
        for entry in turn["sentences"]
        if len((text := sentence_text(transcript, entry)).split()) >= PASSAGE_MIN_WORDS
    ]


def turn_owners(speeches: Mapping[str, ActorSpeechRecord]) -> dict[int, dict[int, str]]:
    owners: dict[int, dict[int, str]] = {}
    for key, record in speeches.items():
        for hearing in record.hearings:
            hearing_owners = owners.setdefault(hearing.hearing_id, {})
            for turn in hearing.turns:
                hearing_owners[turn.turn_index] = key
    return owners


def spoken_hearing(record: ActorSpeechRecord, hearing_id: int) -> ActorHearing | None:
    return next((hearing for hearing in record.hearings if hearing.hearing_id == hearing_id), None)


def check_profile_input(profile: ProfileRecord, record: ActorSpeechRecord) -> None:
    turns = {hearing.hearing_id: len(hearing.turns) for hearing in record.hearings}
    unknown = sorted(set(profile.hearing_ids) - turns.keys())
    if unknown:
        raise ConfigError(
            f"profile of {profile.actor}: hearings {unknown[:3]} are not in the actor speeches"
        )
    read = sum(turns[hearing_id] for hearing_id in profile.hearing_ids)
    if read != profile.n_statements or len(profile.hearing_ids) != profile.n_hearings:
        raise ConfigError(
            f"profile of {profile.actor}: {profile.n_statements} statements in "
            f"{profile.n_hearings} hearings, but the actor speeches have {read} turns in "
            f"{len(profile.hearing_ids)} of them; the profiles come from another actor pass"
        )


def profile_provenance(profile: ProfileRecord, run_name: str, source_sha256: str) -> JsonObject:
    return {
        "run": run_name,
        "model": profile.model,
        "prompt_version": profile.prompt_version,
        "generated_at": profile.generated_at,
        "n_statements": profile.n_statements,
        "n_hearings": profile.n_hearings,
        "hearing_ids": list(profile.hearing_ids),
        "input_tokens": profile.input_tokens,
        "output_tokens": profile.output_tokens,
        "source_sha256": source_sha256,
    }


def page_udvs(trail: ActorTrail, read: set[int]) -> list[JsonObject]:
    udvs = [{k: v for k, v in udv.items() if k != "match_text"} for udv in trail.udvs]
    for udv in udvs:
        udv["in_profile"] = udv["hearing_id"] in read
    return udvs


class ProfileSiteBuilder:
    """Collects, hearing by hearing, the passages and UDVs of each profiled actor."""

    def __init__(
        self,
        profiles: Sequence[ProfileRecord],
        speeches: Mapping[str, ActorSpeechRecord],
        *,
        run_name: str,
        source_sha256: str,
        udv_match_min: float = UDV_MATCH_MIN,
        passage_match_min: float = PASSAGE_MATCH_MIN,
    ) -> None:
        by_name = {record.actor: key for key, record in speeches.items()}
        missing = sorted(profile.actor for profile in profiles if profile.actor not in by_name)
        if missing:
            raise ConfigError(f"profiles of actors without speeches: {missing[:3]}")
        self.profiles = sorted(profiles, key=lambda profile: profile.actor)
        self.speeches = speeches
        self.keys = {profile.actor: by_name[profile.actor] for profile in self.profiles}
        for profile in self.profiles:
            check_profile_input(profile, speeches[self.keys[profile.actor]])
        self.slugs = assign_slugs(profile.actor for profile in self.profiles)
        self.profiled = {self.keys[name]: name for name in self.slugs}
        self.run_name = run_name
        self.source_sha256 = source_sha256
        self.udv_match_min = udv_match_min
        self.passage_match_min = passage_match_min
        self.owners = turn_owners(speeches)
        self.trails: dict[str, ActorTrail] = {name: ActorTrail() for name in self.slugs}
        self.hearings: dict[int, JsonObject] = {}
        self.people: dict[int, dict[str, str]] = {}
        self.udv_actors: dict[str, str] = {}
        self.threshold: float | None = None
        self.cuts: JsonObject | None = None

    def add_hearing(self, payload: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
        """Take one exported hearing page and its index entry."""
        hearing_id = payload["hearing"]["id"]
        signals = payload.get("signals")
        if isinstance(signals, Mapping):
            self.threshold = page_verifier_threshold(signals)
            self.cuts = page_verifier_cuts(signals)
        self.hearings[hearing_id] = {
            "id": hearing_id,
            **{key: entry[key] for key in PAGE_HEARING_KEYS},
        }
        linked = self.link_people(hearing_id, payload["people"])
        self.collect_udvs(hearing_id, payload["udvs"], linked)
        self.collect_sentences(hearing_id, payload["turns"], payload["transcript"])

    def link_people(self, hearing_id: int, people: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        owners = self.owners.get(hearing_id, {})
        linked: dict[str, str] = {}
        for person in people:
            key = majority_owner(person["turns"], owners)
            if key is None or key not in self.profiled:
                continue
            name = self.profiled[key]
            linked[person["name"]] = name
            trail = self.trails[name]
            trail.article_names[person["name"]] += 1
            if person["role"]:
                trail.roles[person["role"]] += 1
        if linked:
            self.people[hearing_id] = {person: self.slugs[name] for person, name in linked.items()}
        return linked

    def collect_udvs(
        self, hearing_id: int, udvs: Sequence[Mapping[str, Any]], linked: Mapping[str, str]
    ) -> None:
        for number, udv in enumerate(udvs, start=1):
            actor = linked.get(udv["actor"]["name"])
            if actor is None:
                continue
            self.udv_actors[udv["id"]] = self.slugs[actor]
            self.trails[actor].udvs.append(trail_udv(udv, hearing_id, number))

    def collect_sentences(
        self, hearing_id: int, turns: Sequence[Mapping[str, Any]], transcript: str
    ) -> None:
        turns_by_index = {turn["index"]: turn for turn in turns}
        for key, name in self.profiled.items():
            spoken = spoken_hearing(self.speeches[key], hearing_id)
            if spoken is None:
                continue
            trail = self.trails[name]
            trail.turns_by_hearing[hearing_id] = len(spoken.turns)
            for actor_turn in spoken.turns:
                turn = turns_by_index.get(actor_turn.turn_index)
                if turn is None:
                    raise ConfigError(
                        f"hearing {hearing_id}: turn {actor_turn.turn_index} of {name} is not "
                        "in the exported turns"
                    )
                trail.sentences.extend(turn_sentences(hearing_id, turn, transcript))

    def match_claims(
        self, profile: ProfileRecord, trail: ActorTrail
    ) -> tuple[list[JsonObject], JsonObject]:
        read = set(profile.hearing_ids)
        return match_claims(
            parse_profile(profile.profile),
            [sentence for sentence in trail.sentences if sentence.hearing_id in read],
            [udv for udv in trail.udvs if udv["hearing_id"] in read and udv["evidence"]],
            MatchThresholds(udv=self.udv_match_min, passage=self.passage_match_min),
        )

    def page_hearings(
        self, record: ActorSpeechRecord, trail: ActorTrail, read: set[int]
    ) -> list[JsonObject]:
        udv_count = Counter(udv["hearing_id"] for udv in trail.udvs)
        return [
            {
                **self.hearings[hearing.hearing_id],
                "turns": len(hearing.turns),
                "in_profile": hearing.hearing_id in read,
                "udvs": udv_count[hearing.hearing_id],
            }
            for hearing in record.hearings
            if hearing.hearing_id in self.hearings
        ]

    def actor_page(self, profile: ProfileRecord) -> tuple[JsonObject, JsonObject]:
        """The page of one profiled actor and its entry in ``actors.json``."""
        name = profile.actor
        trail = self.trails[name]
        record = self.speeches[self.keys[name]]
        read = set(profile.hearing_ids)
        sections, counts = self.match_claims(profile, trail)
        hearings = self.page_hearings(record, trail, read)
        role = most_common(trail.roles)
        display_name = display_name_of(trail.article_names, name)
        udvs = page_udvs(trail, read)
        page = {
            "actor": {
                "slug": self.slugs[name],
                "name": name,
                "display_name": display_name,
                "role": role,
                "article_names": sorted(trail.article_names),
                "party_uf": list(record.party_uf),
            },
            "provenance": profile_provenance(profile, self.run_name, self.source_sha256),
            "match": self.match_block(),
            "counts": counts,
            "sections": sections,
            "hearings": hearings,
            "udvs": udvs,
            "verifier_threshold": self.threshold,
            "verifier_cuts": self.cuts,
        }
        summary = {
            "slug": self.slugs[name],
            "name": name,
            "display_name": display_name,
            "role": role,
            "n_hearings": len(hearings),
            "n_hearings_in_profile": sum(1 for hearing in hearings if hearing["in_profile"]),
            "n_turns": sum(hearing["turns"] for hearing in hearings),
            "hearing_ids": [hearing["id"] for hearing in hearings],
            "n_udvs": len(udvs),
            "claims": counts,
        }
        return page, summary

    def match_block(self) -> JsonObject:
        return {
            "method": MATCH_METHOD,
            "udv_min": self.udv_match_min,
            "passage_min": self.passage_match_min,
            "passage_min_words": PASSAGE_MIN_WORDS,
        }

    def write(self, output_dir: Path, run: Mapping[str, Any] | None) -> ProfileSiteExport:
        """Write every actor page and then ``actors.json`` to ``output_dir``."""
        profiles_dir = output_dir / PROFILES_DIR_NAME
        summaries: list[JsonObject] = []
        profile_bytes: dict[str, int] = {}
        for profile in self.profiles:
            page, summary = self.actor_page(profile)
            path = profiles_dir / f"{summary['slug']}.json"
            write_json(page, path)
            profile_bytes[summary["slug"]] = path.stat().st_size
            summaries.append(summary)
        actors = {
            "run": dict(run) if run is not None else None,
            "profiles": {
                "run": self.run_name,
                "models": sorted({profile.model for profile in self.profiles}),
                "prompt_versions": sorted({profile.prompt_version for profile in self.profiles}),
                "source_sha256": self.source_sha256,
                "match": self.match_block(),
            },
            "verifier_cuts": self.cuts,
            "actors": summaries,
            "people": {str(key): people for key, people in sorted(self.people.items())},
            "udvs": dict(sorted(self.udv_actors.items())),
        }
        actors_path = output_dir / ACTORS_FILE_NAME
        write_json(actors, actors_path)
        return ProfileSiteExport(actors, actors_path.stat().st_size, profile_bytes)


def clear_profile_site(output_dir: Path) -> None:
    (output_dir / ACTORS_FILE_NAME).unlink(missing_ok=True)
    profiles_dir = output_dir / PROFILES_DIR_NAME
    if profiles_dir.is_dir():
        for path in profiles_dir.glob("*.json"):
            path.unlink()
