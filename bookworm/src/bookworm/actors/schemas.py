import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, get_args

from bookworm.models import StrictModel

TurnRole = Literal["chair", "speaker"]

CHAIR_ROLE: TurnRole = "chair"
SPEAKER_ROLE: TurnRole = "speaker"
TURN_ROLES: tuple[TurnRole, ...] = get_args(TurnRole)
SPEECH_SEPARATOR = "\n\n"


class _ActorModel(StrictModel):
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json_line(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


class ActorTurn(_ActorModel):
    turn_index: int
    role: TurnRole
    start_char: int
    end_char: int
    text: str


class ActorHearing(_ActorModel):
    hearing_id: int
    full_speech: str
    turns: list[ActorTurn]


class ActorSpeechRecord(_ActorModel):
    actor: str
    has_party_header: bool
    party_uf: list[str]
    hearings: list[ActorHearing]

    @classmethod
    def from_json_line(cls, line: str) -> "ActorSpeechRecord":
        return cls.model_validate(json.loads(line))

    @property
    def hearing_ids(self) -> list[int]:
        return [hearing.hearing_id for hearing in self.hearings]


class UdvActorLink(_ActorModel):
    udv_id: str
    hearing_id: int
    actor_key: str | None
    actor: str | None
    matched_turns: int
    linked_turns: int

    @classmethod
    def from_json_line(cls, line: str) -> "UdvActorLink":
        return cls.model_validate(json.loads(line))


def write_model_lines(records: Iterable[_ActorModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json_line() + "\n")


def read_actor_speeches(path: Path) -> list[ActorSpeechRecord]:
    with path.open(encoding="utf-8") as handle:
        return [ActorSpeechRecord.from_json_line(line) for line in handle]


def write_actor_speeches(records: Iterable[ActorSpeechRecord], path: Path) -> None:
    write_model_lines(records, path)


def read_udv_actor_links(path: Path) -> list[UdvActorLink]:
    with path.open(encoding="utf-8") as handle:
        return [UdvActorLink.from_json_line(line) for line in handle]


def write_udv_actor_links(links: Iterable[UdvActorLink], path: Path) -> None:
    write_model_lines(links, path)
