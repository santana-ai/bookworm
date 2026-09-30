import json
import shutil
from pathlib import Path

from bookworm.actors.schemas import UdvActorLink, write_udv_actor_links
from bookworm.profiles.schemas import ProfileRecord, write_profiles
from bookworm.udv.schemas import Actor, Evidence, Method, Tier, UdvRecord, write_udv_jsonl

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CONFIG_NAME = "profile_validation_mini.toml"
SPLIT_VERSION = "temporal_mini"
MANIFEST = {
    "split_version": SPLIT_VERSION,
    "train": [1, 2, 3],
    "validation": [4],
    "test": [5, 6],
}

ANA_PROFILE = (
    "Ana Souza defende a universalização do saneamento básico nas cidades pequenas. "
    "Ela cobra da companhia estadual investimentos em tratamento de esgoto.\n\n"
    "Ana critica o atraso nas obras de drenagem urbana."
)
BRUNO_PROFILE = (
    "Bruno Lima propõe ampliar o transporte escolar rural com novos ônibus. "
    "Ele cobra das prefeituras o pagamento dos motoristas em dia."
)
CARLA_PROFILE = (
    "Carla Dias defende creches abertas até as dezenove horas. "
    "Ela pede concurso para professoras da educação infantil."
)

UdvSpec = tuple[str, int, Tier, str, str | None]

UDV_SPECS: tuple[UdvSpec, ...] = (
    ("u01", 1, "quote_found", "Defendeu a universalização do saneamento básico.", "ana"),
    ("u02", 2, "semantic_match_high", "Propôs ampliar o transporte escolar rural.", "bruno"),
    ("u03", 3, "quote_found", "Defendeu creches abertas até as dezenove horas.", "carla"),
    ("u04", 5, "semantic_match_high", "Cobrou investimentos em tratamento de esgoto.", "ana"),
    ("u05", 6, "quote_found", "Criticou o atraso nas obras de drenagem urbana.", "bruno"),
    ("u06", 4, "quote_found", "Pediu concurso para professoras da educação infantil.", "carla"),
    ("u07", 1, "semantic_match_weak", "Falou sobre saneamento nas cidades.", "ana"),
    ("u08", 2, "person_not_resolved", "Opinião de pessoa sem turnos.", None),
    ("u09", 3, "quote_found", "Opinião de pessoa sem chave de ator.", None),
    ("u10", 5, "quote_found", "Opinião de ator sem perfil gerado.", "davi"),
    ("u11", 3, "quote_found", "Defendeu drenagem urbana nas cidades pequenas.", "ana"),
    ("u12", 2, "quote_found", "Opinião sem linha de ligação.", "bruno"),
)
UNLINKED_UDV = "u12"


def udv_record(udv_id: str, hearing_id: int, tier: Tier, proposition: str) -> UdvRecord:
    evidence = (
        None
        if tier == "person_not_resolved"
        else Evidence(
            text=f"Trecho da transcrição para {udv_id}.",
            support_type="direct_quote" if tier == "quote_found" else "semantic_similarity",
            score=None if tier == "quote_found" else 0.7,
            quote_prefix=None,
            start_char=0,
            end_char=10,
            speaker_turn=0,
        )
    )
    return UdvRecord(
        id=udv_id,
        hearing_id=hearing_id,
        actor=Actor(name=f"Pessoa {udv_id}", role="Convidada"),
        proposition=proposition,
        evidence=evidence,
        tier=tier,
        provenance=None,
        method=Method(encoder="stub-encoder", revision="stub-revision-1", embedding_threshold=0.6),
    )


def profile_record(key: str, profile: str, hearing_ids: list[int]) -> ProfileRecord:
    return ProfileRecord(
        actor=key.title(),
        profile=profile,
        model="fake-model",
        prompt_version="0" * 12,
        n_statements=3,
        n_hearings=len(hearing_ids),
        hearing_ids=hearing_ids,
        input_tokens=10,
        output_tokens=10,
        generated_at="2026-09-26T00:00:00+00:00",
        duration_seconds=1.0,
    )


def default_profiles() -> list[ProfileRecord]:
    return [
        profile_record("ana", ANA_PROFILE, [1, 2]),
        profile_record("bruno", BRUNO_PROFILE, [2, 3]),
        profile_record("carla", CARLA_PROFILE, [3]),
    ]


def default_udvs() -> list[UdvRecord]:
    return [udv_record(*spec[:4]) for spec in UDV_SPECS]


def default_links() -> list[UdvActorLink]:
    return [
        UdvActorLink(
            udv_id=udv_id,
            hearing_id=hearing_id,
            actor_key=key,
            actor=None if key is None else key.title(),
            matched_turns=0 if key is None else 1,
            linked_turns=0 if key is None else 1,
        )
        for udv_id, hearing_id, _, _, key in UDV_SPECS
        if udv_id != UNLINKED_UDV
    ]


def write_workdir(
    directory: Path,
    *,
    profiles: list[ProfileRecord] | None = None,
    udvs: list[UdvRecord] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / CONFIG_NAME, directory / CONFIG_NAME)
    write_udv_jsonl(default_udvs() if udvs is None else udvs, directory / "udvs.jsonl")
    write_udv_actor_links(default_links(), directory / "udvs_actor_links.jsonl")
    write_profiles(
        default_profiles() if profiles is None else profiles, directory / "profiles.jsonl"
    )
    (directory / "splits.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return directory / CONFIG_NAME
