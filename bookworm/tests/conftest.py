import hashlib
import importlib.util
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from bookworm import (
    Actor,
    HearingRecord,
    Method,
    SplitConfig,
    Tier,
    UdvRecord,
    load_hearings,
    load_split_config,
)
from bookworm.features.encoders import FloatMatrix

LDS_SHA256 = "c4e392ab95ce22f6228eace16f15e9efe1c846724846b873dec420aec3d872c0"
LDS_PATH_VARIABLE = "BOOKWORM_LDS_PATH"
EMBEDDING_CACHE_VARIABLE = "BOOKWORM_EMBEDDING_CACHE"
CHALLENGE_DIR_VARIABLE = "BOOKWORM_CHALLENGE_DIR"
ARTIFACTS_DIR_VARIABLE = "BOOKWORM_ARTIFACTS_DIR"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PACKAGE = "utils"
MINI_THRESHOLD = 0.6

JOAO_OPINION = "Pediu reforço na fiscalização das cooperativas."
JOAO_FIRST_SENTENCE = (
    "Senhor Presidente, a fiscalização das cooperativas precisa ser reforçada antes de "
    "qualquer mudança."
)
SHORT_QUOTE_OPINION = (
    'Disse que o setor precisa de "prazos de transição realmente longos para as cooperativas".'
)
SHORT_QUOTE_SENTENCE = "O setor precisa de prazos de transição mais longos e de regras claras."
THANKS_OPINION = "Agradeceu aos colegas da mesa pela oportunidade de falar."
THANKS_SENTENCE = "Agradeço, novamente, aos colegas da mesa pela oportunidade de falar aqui."
PALMAS_SENTENCE = "Esse é o ponto central da minha fala, sem exagero nenhum.(Palmas.)"
CROSS_TURN_TEXT = f"{PALMAS_SENTENCE} {THANKS_SENTENCE}"

DEFAULT_VECTOR = (0.0, 0.0, 0.0, 1.0)
FIXTURE_VECTORS: dict[str, tuple[float, ...]] = {
    JOAO_OPINION: (1.0, 0.0, 0.0, 0.0),
    JOAO_FIRST_SENTENCE: (3.0, 4.0, 0.0, 0.0),
    SHORT_QUOTE_OPINION: (0.0, 1.0, 0.0, 0.0),
    SHORT_QUOTE_SENTENCE: (0.0, 1.0, 0.0, 0.0),
    THANKS_OPINION: (0.0, 0.0, 1.0, 0.0),
    THANKS_SENTENCE: (0.0, 0.0, 1.0, 0.0),
}

TURNS_HEARING_ID = 3
TURNS_REFORM_TURN_1 = "A reforma do ensino médio é necessária para os estudantes do campo."
TURNS_REFORM_TURN_5 = (
    "Para concluir: a reforma do ensino médio é necessária, mas precisa de professores formados "
    "e bem pagos."
)
TURNS_PHYSICS_SENTENCE = "Faltam professores de física nas escolas rurais.(Palmas.)"
TURNS_TRAINING_OPINION = "Defendeu formação continuada para os professores."
TURNS_TRAINING_SENTENCE = "Os professores precisam de formação continuada."
TURNS_TRANSPORT_OPINION = "Apontou o transporte escolar como o maior gargalo da região."
TURNS_TRANSPORT_SENTENCE = "O transporte escolar é o maior gargalo da região."
TURNS_FAMILIES_OPINION = (
    'Disse que "o transporte escolar é pago pelas famílias" das prefeituras pequenas.'
)
TURNS_OPENING_OPINION = "Abriu a audiência sobre educação no campo."
TURNS_OPENING_SENTENCE = "Declaro aberta a audiência sobre educação no campo."
TURNS_VECTORS: dict[str, tuple[float, ...]] = {
    TURNS_TRAINING_OPINION: (1.0, 0.0, 0.0, 0.0),
    TURNS_TRAINING_SENTENCE: (1.0, 0.0, 0.0, 0.0),
    TURNS_TRANSPORT_OPINION: (0.0, 1.0, 0.0, 0.0),
    TURNS_FAMILIES_OPINION: (0.0, 1.0, 0.0, 0.0),
    TURNS_TRANSPORT_SENTENCE: (0.0, 1.0, 0.0, 0.0),
    TURNS_OPENING_OPINION: (0.0, 0.0, 1.0, 0.0),
    TURNS_OPENING_SENTENCE: (0.0, 0.0, 1.0, 0.0),
}


SPLIT_UDV_TIERS: tuple[tuple[int, Tier], ...] = (
    (1, "quote_found"),
    (1, "semantic_match_high"),
    (8, "no_evidence"),
    (9, "person_not_resolved"),
)


def configured_path(variable: str, default: Path) -> Path:
    raw_path = os.environ.get(variable)
    return Path(raw_path) if raw_path else default


def challenge_path() -> Path:
    return configured_path(CHALLENGE_DIR_VARIABLE, REPOSITORY_ROOT / "challenge")


def artifacts_path() -> Path:
    return configured_path(ARTIFACTS_DIR_VARIABLE, challenge_path() / "artifacts")


def is_reference_package_module(name: str) -> bool:
    return name == REFERENCE_PACKAGE or name.startswith(f"{REFERENCE_PACKAGE}.")


def reference_module_name(module_path: Path) -> str:
    digest = hashlib.sha256(str(module_path.resolve()).encode()).hexdigest()[:12]
    return f"bookworm_reference_{digest}_{module_path.stem}"


def import_challenge_module(challenge_dir: Path, module_name: str) -> ModuleType:
    module_path = challenge_dir.joinpath(*module_name.split(".")).with_suffix(".py")
    if not module_path.is_file():
        pytest.skip(f"reference module not found at {module_path}")
    unique_name = reference_module_name(module_path)
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if is_reference_package_module(name)
    }
    writes_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(challenge_dir))
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[unique_name]
        raise
    finally:
        sys.path.remove(str(challenge_dir))
        sys.dont_write_bytecode = writes_bytecode
        for name in [name for name in sys.modules if is_reference_package_module(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
    assert Path(str(module.__file__)).resolve() == module_path.resolve()
    return module


def split_udv(hearing_id: int, position: int, tier: Tier) -> UdvRecord:
    return UdvRecord(
        id=f"udv-{hearing_id}-0-{position}",
        hearing_id=hearing_id,
        actor=Actor(name="Pessoa Sintética", role="Convidada"),
        proposition="Opinião sintética.",
        evidence=None,
        tier=tier,
        provenance=None,
        method=Method(encoder="stub-encoder", revision="stub-revision-1", embedding_threshold=0.6),
    )


class CacheMissError(AssertionError):
    pass


class CacheOnlyEncoder:
    def __init__(self, name: str, revision: str, device: str) -> None:
        self._name = name
        self._revision = revision
        self._device = device

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def cache_identity(self) -> str:
        return f"{self._name}@{self._revision}@{self._device}"

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        raise CacheMissError(f"no cached embeddings for {len(texts)} texts")

    def runtime_info(self) -> dict[str, Any]:
        return {"device": self._device}


def cache_only_encoder(coverage: Mapping[str, Any]) -> CacheOnlyEncoder:
    encoder = coverage["config"]["encoder"]
    return CacheOnlyEncoder(
        encoder["name"], encoder["revision"], coverage["encoder_runtime"]["device"]
    )


def directory_state(path: Path) -> dict[str, tuple[int, int]]:
    return {
        entry.name: (entry.stat().st_size, entry.stat().st_mtime_ns) for entry in path.iterdir()
    }


class StubEncoder:
    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]] | None = None,
        default: Sequence[float] = DEFAULT_VECTOR,
        *,
        name: str = "stub-encoder",
        revision: str = "stub-revision-1",
    ) -> None:
        self.vectors = dict(FIXTURE_VECTORS if vectors is None else vectors)
        self.default = tuple(default)
        self._name = name
        self._revision = revision
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def cache_identity(self) -> str:
        return f"{self._name}@{self._revision}@cpu"

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        self.calls.append(list(texts))
        rows = [self.vectors.get(text, self.default) for text in texts]
        return np.array(rows, dtype=np.float64).reshape(len(texts), len(self.default))

    def runtime_info(self) -> dict[str, Any]:
        return {"device": "cpu", "max_seq_length": 0, "embedding_dimension": len(self.default)}


@pytest.fixture(scope="session")
def lds_mini_path() -> Path:
    return FIXTURES_DIR / "lds_mini.jsonl"


@pytest.fixture(scope="session")
def udv_mini_config_path() -> Path:
    return FIXTURES_DIR / "udv_mini.toml"


@pytest.fixture(scope="session")
def mini_hearings(lds_mini_path: Path) -> dict[int, HearingRecord]:
    return {hearing.id: hearing for hearing in load_hearings(lds_mini_path)}


@pytest.fixture(scope="session")
def mini_hearing_list(lds_mini_path: Path) -> list[HearingRecord]:
    return load_hearings(lds_mini_path)


@pytest.fixture(scope="session")
def lds_turns_path() -> Path:
    return FIXTURES_DIR / "lds_turns.jsonl"


@pytest.fixture(scope="session")
def turns_hearing(lds_turns_path: Path) -> HearingRecord:
    hearings = load_hearings(lds_turns_path)
    assert [hearing.id for hearing in hearings] == [TURNS_HEARING_ID]
    return hearings[0]


@pytest.fixture
def mini_workdir(
    tmp_path: Path, lds_mini_path: Path, udv_mini_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    shutil.copy(lds_mini_path, tmp_path / lds_mini_path.name)
    shutil.copy(udv_mini_config_path, tmp_path / udv_mini_config_path.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def splits_mini_path() -> Path:
    return FIXTURES_DIR / "splits_mini.jsonl"


@pytest.fixture(scope="session")
def splits_mini_config_path() -> Path:
    return FIXTURES_DIR / "splits_mini.toml"


@pytest.fixture(scope="session")
def split_hearings(splits_mini_path: Path) -> list[HearingRecord]:
    return load_hearings(splits_mini_path)


@pytest.fixture(scope="session")
def split_config(splits_mini_config_path: Path) -> SplitConfig:
    return load_split_config(splits_mini_config_path)


@pytest.fixture(scope="session")
def split_udvs() -> list[UdvRecord]:
    return [
        split_udv(hearing_id, position, tier)
        for position, (hearing_id, tier) in enumerate(SPLIT_UDV_TIERS)
    ]


@pytest.fixture
def splits_workdir(
    tmp_path: Path,
    splits_mini_path: Path,
    splits_mini_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    shutil.copy(splits_mini_path, tmp_path / splits_mini_path.name)
    shutil.copy(splits_mini_config_path, tmp_path / splits_mini_config_path.name)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def lds_path() -> Path:
    raw_path = os.environ.get(LDS_PATH_VARIABLE)
    if not raw_path:
        pytest.skip(f"{LDS_PATH_VARIABLE} is not set")
    path = Path(raw_path)
    if not path.is_file():
        pytest.skip(f"{LDS_PATH_VARIABLE} points to a missing file: {path}")
    return path


@pytest.fixture(scope="session")
def lds_hearings(lds_path: Path) -> list[HearingRecord]:
    return load_hearings(lds_path, LDS_SHA256)


@pytest.fixture(scope="session")
def challenge_dir() -> Path:
    path = challenge_path()
    if not path.is_dir():
        pytest.skip(f"challenge project not found at {path}")
    return path


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    path = artifacts_path()
    if not path.is_dir():
        pytest.skip(f"challenge artifacts not found at {path}")
    return path


@pytest.fixture(scope="session")
def udv_artifacts_dir(artifacts_dir: Path) -> Path:
    path = artifacts_dir / "udv"
    if not path.is_dir():
        pytest.skip(f"UDV artifacts not found at {path}")
    return path


@pytest.fixture(scope="session")
def embedding_cache_dir() -> Path:
    raw_path = os.environ.get(EMBEDDING_CACHE_VARIABLE)
    if not raw_path:
        pytest.skip(f"{EMBEDDING_CACHE_VARIABLE} is not set")
    path = Path(raw_path)
    if not path.is_dir():
        pytest.skip(f"{EMBEDDING_CACHE_VARIABLE} points to a missing directory: {path}")
    return path


@pytest.fixture(scope="session")
def split_artifacts_dir(artifacts_dir: Path) -> Path:
    path = artifacts_dir / "splits"
    if not path.is_dir():
        pytest.skip(f"split artifacts not found at {path}")
    return path


@pytest.fixture(scope="session")
def challenge_split_config_path(challenge_dir: Path) -> Path:
    path = challenge_dir / "configs" / "splits.toml"
    if not path.is_file():
        pytest.skip(f"split config not found at {path}")
    return path


def udv_artifact_path(udv_artifacts_dir: Path, file_name: str) -> Path:
    path = udv_artifacts_dir / file_name
    if not path.is_file():
        pytest.skip(f"UDV artifact not found at {path}")
    return path
