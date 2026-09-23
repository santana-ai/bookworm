from pydantic import BaseModel, ConfigDict


class _DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Participant(_DatasetModel):
    nome: str
    cargo: str
    opinioes: tuple[str, ...]


class HearingMetadata(_DatasetModel):
    assunto: str
    envolvidos: tuple[Participant, ...]


class HearingRecord(_DatasetModel):
    id: int
    materia: str
    metadados: HearingMetadata
    transcricao: str
