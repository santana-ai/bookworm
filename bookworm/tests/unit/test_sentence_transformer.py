from typing import Any

import numpy as np
import pytest

from bookworm import SentenceEncoder

sentence_transformer = pytest.importorskip("bookworm.features.sentence_transformer")


class FakeModel:
    max_seq_length = 128

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **options: Any) -> Any:
        self.calls.append((texts, options))
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeLoader:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, name: str, revision: str, device: str) -> Any:
        self.calls.append((name, revision, device))
        return self.model


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_select_device_auto_prefers_mps_then_cuda(
    monkeypatch: pytest.MonkeyPatch, mps: bool, cuda: bool, expected: str
) -> None:
    torch = sentence_transformer.torch
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert sentence_transformer.select_device("auto") == expected


def test_encoder_satisfies_the_protocol() -> None:
    encoder: SentenceEncoder = sentence_transformer.SentenceTransformerEncoder(
        "org/model", "rev123", "cpu", loader=FakeLoader()
    )
    assert isinstance(encoder, SentenceEncoder)


def test_select_device_keeps_an_explicit_device() -> None:
    assert sentence_transformer.select_device("cpu") == "cpu"
    assert sentence_transformer.select_device("cuda:1") == "cuda:1"


def test_seed_torch_seeds_the_torch_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    monkeypatch.setattr(sentence_transformer.torch, "manual_seed", seeds.append)
    sentence_transformer.seed_torch(42)
    assert seeds == [42]


def test_model_is_loaded_lazily_once() -> None:
    loader = FakeLoader()
    encoder = sentence_transformer.SentenceTransformerEncoder(
        "org/model", "rev123", "cpu", 16, loader=loader
    )
    assert not encoder.is_loaded
    assert encoder.cache_identity == "org/model@rev123@cpu"
    assert loader.calls == []
    encoder.encode(["uma frase"])
    encoder.encode(["outra frase"])
    assert encoder.is_loaded
    assert loader.calls == [("org/model", "rev123", "cpu")]


def test_encode_passes_the_reference_options() -> None:
    loader = FakeLoader()
    encoder = sentence_transformer.SentenceTransformerEncoder(
        "org/model", "rev123", "cpu", 16, loader=loader
    )
    matrix = encoder.encode(("uma frase", "outra frase"))
    assert matrix.shape == (2, 3)
    assert loader.model.calls == [
        (
            ["uma frase", "outra frase"],
            {"batch_size": 16, "convert_to_numpy": True, "show_progress_bar": False},
        )
    ]


def test_empty_input_uses_the_model_dimension() -> None:
    loader = FakeLoader()
    encoder = sentence_transformer.SentenceTransformerEncoder(
        "org/model", "rev123", "cpu", loader=loader
    )
    empty = encoder.encode([])
    assert empty.shape == (0, 3)
    assert empty.dtype == np.float32
    assert loader.model.calls == []


def test_runtime_info_and_properties() -> None:
    encoder = sentence_transformer.SentenceTransformerEncoder(
        "org/model", "rev123", "cpu", loader=FakeLoader()
    )
    assert (encoder.name, encoder.revision, encoder.device, encoder.batch_size) == (
        "org/model",
        "rev123",
        "cpu",
        64,
    )
    assert encoder.runtime_info() == {
        "device": "cpu",
        "max_seq_length": 128,
        "embedding_dimension": 3,
    }


def test_default_loader_passes_name_revision_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, dict[str, Any]]] = []

    def fake_class(name: str, **options: Any) -> str:
        created.append((name, options))
        return "model"

    monkeypatch.setattr(sentence_transformer, "SentenceTransformer", fake_class)
    assert sentence_transformer.load_sentence_transformer("org/model", "rev123", "cpu") == "model"
    sentence_transformer.load_sentence_transformer(
        "org/model", "rev123", "cpu", local_files_only=True
    )
    assert created == [
        ("org/model", {"revision": "rev123", "device": "cpu", "local_files_only": False}),
        ("org/model", {"revision": "rev123", "device": "cpu", "local_files_only": True}),
    ]
