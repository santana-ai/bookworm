from bookworm.models import StrictModel


class Participant(StrictModel):
    nome: str
    cargo: str
    opinioes: tuple[str, ...]


class HearingMetadata(StrictModel):
    assunto: str
    envolvidos: tuple[Participant, ...]


class HearingRecord(StrictModel):
    id: int
    materia: str
    metadados: HearingMetadata
    transcricao: str
