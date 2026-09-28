import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from bookworm.actors.schemas import ActorSpeechRecord
from bookworm.data.io import write_json
from bookworm.errors import ConfigError
from bookworm.profiles.schemas import ProfileRecord

JsonObject = dict[str, Any]

ACTORS_FILE_NAME = "actors.json"
PROFILES_DIR_NAME = "profiles"
UNTITLED_SECTION = "Sem título"
UDV_MATCH_MIN = 0.3
PASSAGE_MATCH_MIN = 0.15
PASSAGE_MIN_WORDS = 5
PASSAGE_MAX_CHARS = 480
SCORE_DECIMALS = 4
PASSAGE_ELLIPSIS = " […]"
MATCH_METHOD = (
    "TF-IDF cosine between each profile item and, separately, the actor's sentences in the "
    "hearings the profile read and the text (proposition and evidence) of the actor's UDVs in "
    "those hearings; vectorizer fitted per actor on those texts, accents removed, sublinear tf, "
    "fixed stop words. A UDV is linked when the item's passage overlaps the UDV evidence in the "
    "transcript (same_sentence) or, failing that, when the item is similar enough to the UDV "
    "text (similar_text)"
)
STOP_WORDS = (
    "a",
    "o",
    "e",
    "as",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "num",
    "numa",
    "por",
    "pelo",
    "pela",
    "pelos",
    "pelas",
    "para",
    "pra",
    "pro",
    "com",
    "sem",
    "sob",
    "sobre",
    "ate",
    "que",
    "se",
    "nao",
    "sim",
    "ja",
    "ha",
    "foi",
    "era",
    "sao",
    "ser",
    "sera",
    "estar",
    "esta",
    "estao",
    "este",
    "esse",
    "essa",
    "esses",
    "essas",
    "isso",
    "isto",
    "aquele",
    "aquela",
    "aquilo",
    "ele",
    "ela",
    "eles",
    "elas",
    "eu",
    "nos",
    "voce",
    "voces",
    "me",
    "te",
    "lhe",
    "lhes",
    "seu",
    "sua",
    "seus",
    "suas",
    "meu",
    "minha",
    "nosso",
    "nossa",
    "ao",
    "aos",
    "como",
    "mas",
    "ou",
    "mais",
    "muito",
    "tambem",
    "so",
    "entao",
    "aqui",
    "la",
    "quando",
    "onde",
    "qual",
    "quais",
    "quem",
    "porque",
    "tem",
    "ter",
    "dito",
    "disse",
    "falou",
    "falaram",
    "foram",
    "forma",
    "entre",
    "apos",
    "ainda",
    "cada",
    "todo",
    "toda",
    "todos",
    "todas",
    "outro",
    "outra",
    "outros",
    "outras",
)
SECTION_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
BULLET_PATTERN = re.compile(r"^\s{0,1}[-*]\s+(.+?)\s*$")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
SAME_SENTENCE_RULE = "same_sentence"
SIMILAR_TEXT_RULE = "similar_text"


@dataclass(frozen=True, slots=True)
class ProfileSection:
    title: str
    claims: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Sentence:
    hearing_id: int
    turn: int
    start: int | None
    end: int | None
    text: str


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


def parse_profile(text: str) -> list[ProfileSection]:
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        heading = SECTION_PATTERN.match(raw.strip())
        if heading is not None:
            sections.append((heading.group(1).strip(), []))
            continue
        if not sections:
            sections.append((UNTITLED_SECTION, []))
        claims = sections[-1][1]
        bullet = BULLET_PATTERN.match(raw)
        if bullet is not None or not claims:
            claims.append(bullet.group(1) if bullet is not None else raw.strip())
        else:
            claims[-1] = f"{claims[-1]} {raw.strip()}"
    return [ProfileSection(title, tuple(claims)) for title, claims in sections if claims]


def actor_slug(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return SLUG_PATTERN.sub("-", folded).strip("-") or "ator"


def assign_slugs(names: Iterable[str]) -> dict[str, str]:
    slugs: dict[str, str] = {}
    taken: set[str] = set()
    for name in sorted(set(names)):
        base = actor_slug(name)
        slug = base
        suffix = 2
        while slug in taken:
            slug = f"{base}-{suffix}"
            suffix += 1
        taken.add(slug)
        slugs[name] = slug
    return slugs


def trim_passage(text: str, max_chars: int = PASSAGE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    room = text[:max_chars]
    boundary = room.rfind(" ")
    kept = room[:boundary] if boundary > max_chars // 2 else room
    return kept.rstrip(" ,;:") + PASSAGE_ELLIPSIS


def majority_owner(turns: Sequence[int], owners: Mapping[int, str]) -> str | None:
    counts = Counter(owners[turn] for turn in turns if turn in owners)
    if not counts:
        return None
    key, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return key


def cosine_rows(vectorizer: TfidfVectorizer, rows: Sequence[str], columns: Sequence[str]) -> Any:
    if not rows or not columns:
        return np.zeros((len(rows), len(columns)))
    left = vectorizer.transform(list(rows))
    right = vectorizer.transform(list(columns))
    return (left @ right.T).toarray()


def fit_vectorizer(corpus: Sequence[str]) -> TfidfVectorizer | None:
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        stop_words=list(STOP_WORDS),
        dtype=np.float64,
    )
    try:
        vectorizer.fit(list(corpus))
    except ValueError:
        return None
    return vectorizer


def verifier_of(udv: Mapping[str, Any]) -> JsonObject | None:
    signals = udv.get("signals")
    if not isinstance(signals, Mapping) or signals.get("verifier") is None:
        return None
    verifier = signals["verifier"]
    return {"probability": verifier["probability"], "supported": verifier["supported"]}


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


def same_sentence(sentence: Sentence, udvs: Sequence[Mapping[str, Any]]) -> int | None:
    if sentence.start is None or sentence.end is None:
        return None
    for position, udv in enumerate(udvs):
        evidence = udv["evidence"]
        if (
            udv["hearing_id"] == sentence.hearing_id
            and evidence["turn"] == sentence.turn
            and evidence["start"] < sentence.end
            and sentence.start < evidence["end"]
        ):
            return position
    return None


def most_common(counts: Counter[str]) -> str | None:
    if not counts:
        return None
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def best_match(scores: Any, minimum: float) -> tuple[int, float] | None:
    if scores.size == 0:
        return None
    position = int(np.argmax(scores))
    value = float(scores[position])
    if value < minimum:
        return None
    return position, round(value, SCORE_DECIMALS)


class ProfileSiteBuilder:
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
        self.owners: dict[int, dict[int, str]] = {}
        for key, record in speeches.items():
            for hearing in record.hearings:
                owners = self.owners.setdefault(hearing.hearing_id, {})
                for turn in hearing.turns:
                    owners[turn.turn_index] = key
        self.trails: dict[str, ActorTrail] = {name: ActorTrail() for name in self.slugs}
        self.hearings: dict[int, JsonObject] = {}
        self.people: dict[int, dict[str, str]] = {}
        self.udv_actors: dict[str, str] = {}
        self.threshold: float | None = None

    def add_hearing(self, payload: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
        hearing_id = payload["hearing"]["id"]
        owners = self.owners.get(hearing_id, {})
        signals = payload.get("signals")
        if isinstance(signals, Mapping):
            self.threshold = signals["verifier"]["threshold"]
        self.hearings[hearing_id] = {
            "id": hearing_id,
            "split": entry["split"],
            "article_date": entry["article_date"],
            "title": entry["title"],
            "assunto": entry["assunto"],
        }
        linked: dict[str, str] = {}
        for person in payload["people"]:
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
        for number, udv in enumerate(payload["udvs"], start=1):
            actor = linked.get(udv["actor"]["name"])
            if actor is None:
                continue
            self.udv_actors[udv["id"]] = self.slugs[actor]
            self.trails[actor].udvs.append(
                {
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
            )
        turns_by_index = {turn["index"]: turn for turn in payload["turns"]}
        for key, name in self.profiled.items():
            record = self.speeches[key]
            spoken = next((h for h in record.hearings if h.hearing_id == hearing_id), None)
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
                trail.sentences.extend(
                    Sentence(hearing_id, turn["index"], sentence["start"], sentence["end"], text)
                    for sentence in turn["sentences"]
                    if len((text := sentence["text"]).split()) >= PASSAGE_MIN_WORDS
                )

    def match_claims(
        self, profile: ProfileRecord, trail: ActorTrail
    ) -> tuple[list[JsonObject], JsonObject]:
        read = set(profile.hearing_ids)
        sentences = [sentence for sentence in trail.sentences if sentence.hearing_id in read]
        udvs = [udv for udv in trail.udvs if udv["hearing_id"] in read and udv["evidence"]]
        sections = parse_profile(profile.profile)
        claims = [claim for section in sections for claim in section.claims]
        vectorizer = fit_vectorizer(
            [sentence.text for sentence in sentences] + [udv["match_text"] for udv in udvs] + claims
        )
        if vectorizer is None:
            passage_scores = np.zeros((len(claims), 0))
            udv_scores = np.zeros((len(claims), 0))
        else:
            passage_scores = cosine_rows(vectorizer, claims, [s.text for s in sentences])
            udv_scores = cosine_rows(vectorizer, claims, [udv["match_text"] for udv in udvs])
        counts = Counter[str]()
        out: list[JsonObject] = []
        row = 0
        for section in sections:
            items: list[JsonObject] = []
            for claim in section.claims:
                passage = best_match(passage_scores[row], self.passage_match_min)
                udv = best_match(udv_scores[row], self.udv_match_min)
                row += 1
                item: JsonObject = {"text": claim, "passage": None, "udv": None}
                if passage is not None:
                    sentence = sentences[passage[0]]
                    item["passage"] = {
                        "hearing_id": sentence.hearing_id,
                        "turn": sentence.turn,
                        "start": sentence.start,
                        "end": sentence.end,
                        "text": trim_passage(sentence.text),
                        "score": passage[1],
                    }
                shared = None if passage is None else same_sentence(sentences[passage[0]], udvs)
                if shared is not None:
                    item["udv"] = {
                        "id": udvs[shared]["id"],
                        "rule": SAME_SENTENCE_RULE,
                        "score": round(float(udv_scores[row - 1][shared]), SCORE_DECIMALS),
                    }
                elif udv is not None:
                    item["udv"] = {
                        "id": udvs[udv[0]]["id"],
                        "rule": SIMILAR_TEXT_RULE,
                        "score": udv[1],
                    }
                counts["claims"] += 1
                linked = item["udv"] is not None
                counts["with_udv"] += linked
                counts["passage_only"] += not linked and passage is not None
                counts["without_evidence"] += not linked and passage is None
                items.append(item)
            out.append({"title": section.title, "claims": items})
        summary = {
            key: counts[key] for key in ("claims", "with_udv", "passage_only", "without_evidence")
        }
        return out, summary

    def actor_page(self, profile: ProfileRecord) -> tuple[JsonObject, JsonObject]:
        name = profile.actor
        trail = self.trails[name]
        record = self.speeches[self.keys[name]]
        read = set(profile.hearing_ids)
        sections, counts = self.match_claims(profile, trail)
        udv_count = Counter(udv["hearing_id"] for udv in trail.udvs)
        hearings = [
            {
                **self.hearings[hearing.hearing_id],
                "turns": len(hearing.turns),
                "in_profile": hearing.hearing_id in read,
                "udvs": udv_count[hearing.hearing_id],
            }
            for hearing in record.hearings
            if hearing.hearing_id in self.hearings
        ]
        role = most_common(trail.roles)
        udvs = [{k: v for k, v in udv.items() if k != "match_text"} for udv in trail.udvs]
        for udv in udvs:
            udv["in_profile"] = udv["hearing_id"] in read
        page = {
            "actor": {
                "slug": self.slugs[name],
                "name": name,
                "role": role,
                "article_names": sorted(trail.article_names),
                "party_uf": list(record.party_uf),
            },
            "provenance": {
                "run": self.run_name,
                "model": profile.model,
                "prompt_version": profile.prompt_version,
                "generated_at": profile.generated_at,
                "n_statements": profile.n_statements,
                "n_hearings": profile.n_hearings,
                "hearing_ids": list(profile.hearing_ids),
                "input_tokens": profile.input_tokens,
                "output_tokens": profile.output_tokens,
                "source_sha256": self.source_sha256,
            },
            "match": self.match_block(),
            "counts": counts,
            "sections": sections,
            "hearings": hearings,
            "udvs": udvs,
            "verifier_threshold": self.threshold,
        }
        summary = {
            "slug": self.slugs[name],
            "name": name,
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
        profiles_dir = output_dir / PROFILES_DIR_NAME
        summaries: list[JsonObject] = []
        profile_bytes: dict[str, int] = {}
        models = sorted({profile.model for profile in self.profiles})
        versions = sorted({profile.prompt_version for profile in self.profiles})
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
                "models": models,
                "prompt_versions": versions,
                "source_sha256": self.source_sha256,
                "match": self.match_block(),
            },
            "actors": summaries,
            "people": {str(key): people for key, people in sorted(self.people.items())},
            "udvs": dict(sorted(self.udv_actors.items())),
        }
        actors_path = output_dir / ACTORS_FILE_NAME
        write_json(actors, actors_path)
        return ProfileSiteExport(actors, actors_path.stat().st_size, profile_bytes)


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


def clear_profile_site(output_dir: Path) -> None:
    (output_dir / ACTORS_FILE_NAME).unlink(missing_ok=True)
    profiles_dir = output_dir / PROFILES_DIR_NAME
    if profiles_dir.is_dir():
        for path in profiles_dir.glob("*.json"):
            path.unlink()
