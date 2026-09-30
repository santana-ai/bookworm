from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import StubEncoder

from bookworm import (
    CachedEncoder,
    EmbeddingCacheMissError,
    RunCacheEncoder,
    SentenceEncoder,
    cache_file_name,
    cache_key,
    sentence_encoder_identity,
)
from bookworm.features.encoders import FloatMatrix

SERAFIM_IDENTITY = (
    "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder@"
    "a01887015444f7599669c509447c5bdbce958916@mps"
)
TEXTS = ["Primeira frase da audiência.", "Segunda frase, com acentuação: ação e opinião."]


class WrongShapeEncoder(StubEncoder):
    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        super().encode(texts)
        return np.zeros((len(texts) + 1, 4))


def test_stub_encoder_satisfies_the_protocol() -> None:
    assert isinstance(StubEncoder(), SentenceEncoder)


def test_cache_key_matches_the_reference_algorithm() -> None:
    assert cache_key(SERAFIM_IDENTITY, TEXTS) == (
        "fb4cc2e41684777bd1fd33fb002f172da8608461122813d1bfc765c6401e1e29"
    )
    cpu_identity = SERAFIM_IDENTITY.removesuffix("@mps") + "@cpu"
    assert cache_key(cpu_identity, TEXTS) == (
        "02b34ef0e8efb83616185db20ffa71f1b810df855f14a53eadfd03845c29dc31"
    )


def test_cache_file_name_uses_label_and_sixteen_hex_characters() -> None:
    assert cache_file_name("sentences_12", SERAFIM_IDENTITY, TEXTS) == (
        "sentences_12_fb4cc2e41684777b.npy"
    )


def test_cache_key_separator_is_not_escaped() -> None:
    assert cache_key(SERAFIM_IDENTITY, ["a", "b"]) != cache_key(SERAFIM_IDENTITY, ["ab"])
    assert cache_key(SERAFIM_IDENTITY, ["a", "b"]) == cache_key(SERAFIM_IDENTITY, ["a\x1eb"])


def test_cache_miss_encodes_and_saves(tmp_path: Path) -> None:
    encoder = StubEncoder()
    cached = CachedEncoder(encoder, tmp_path / "cache")
    embeddings = cached.encode(TEXTS, "opinions_3")
    path = cached.cache_path(TEXTS, "opinions_3")
    assert path == tmp_path / "cache" / cache_file_name("opinions_3", encoder.cache_identity, TEXTS)
    assert path.is_file()
    assert np.array_equal(np.load(path), embeddings)
    assert encoder.calls == [TEXTS]


def test_cache_hit_skips_the_encoder(tmp_path: Path) -> None:
    CachedEncoder(StubEncoder(), tmp_path).encode(TEXTS, "opinions_3")
    encoder = StubEncoder()
    embeddings = CachedEncoder(encoder, tmp_path).encode(TEXTS, "opinions_3")
    assert encoder.calls == []
    assert embeddings.shape == (2, 4)


def test_cache_key_depends_on_the_encoder_identity(tmp_path: Path) -> None:
    CachedEncoder(StubEncoder(), tmp_path).encode(TEXTS, "opinions_3")
    other = StubEncoder(revision="stub-revision-2")
    CachedEncoder(other, tmp_path).encode(TEXTS, "opinions_3")
    assert other.calls == [TEXTS]
    assert len(list(tmp_path.iterdir())) == 2


def test_read_only_cache_never_writes(tmp_path: Path) -> None:
    encoder = StubEncoder()
    cached = CachedEncoder(encoder, tmp_path / "cache", read_only=True)
    cached.encode(TEXTS, "opinions_3")
    assert not (tmp_path / "cache").exists()
    assert encoder.calls == [TEXTS]


def test_no_cache_dir_always_encodes() -> None:
    encoder = StubEncoder()
    cached = CachedEncoder(encoder)
    cached.encode(TEXTS, "opinions_3")
    cached.encode(TEXTS, "opinions_3")
    assert cached.cache_path(TEXTS, "opinions_3") is None
    assert encoder.calls == [TEXTS, TEXTS]


def test_empty_input_is_encoded_without_touching_the_cache(tmp_path: Path) -> None:
    encoder = StubEncoder()
    embeddings = CachedEncoder(encoder, tmp_path / "cache").encode([], "sentences_9")
    assert embeddings.shape == (0, 4)
    assert not (tmp_path / "cache").exists()
    assert encoder.calls == [[]]


def test_wrong_number_of_rows_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 2 embedding rows"):
        CachedEncoder(WrongShapeEncoder(), tmp_path).encode(TEXTS, "opinions_3")
    assert list(tmp_path.iterdir()) == []


def test_corrupted_cache_file_is_rejected(tmp_path: Path) -> None:
    encoder = StubEncoder()
    cached = CachedEncoder(encoder, tmp_path)
    path = cached.cache_path(TEXTS, "opinions_3")
    assert path is not None
    np.save(path, np.zeros((5, 4)))
    with pytest.raises(ValueError, match=path.name):
        cached.encode(TEXTS, "opinions_3")


def test_runtime_info_of_the_stub() -> None:
    info: dict[str, Any] = StubEncoder().runtime_info()
    assert info == {"device": "cpu", "max_seq_length": 0, "embedding_dimension": 4}


def test_cache_only_reads_a_hit_without_the_encoder(tmp_path: Path) -> None:
    expected = CachedEncoder(StubEncoder(), tmp_path).encode(TEXTS, "opinions_3")
    encoder = StubEncoder()
    cached = CachedEncoder(encoder, tmp_path, cache_only=True)
    assert cached.read_only
    assert np.array_equal(cached.encode(TEXTS, "opinions_3"), expected)
    assert encoder.calls == []


def test_cache_only_fails_on_a_miss_without_encoding(tmp_path: Path) -> None:
    encoder = StubEncoder()
    cached = CachedEncoder(encoder, tmp_path / "cache", cache_only=True)
    path = cached.cache_path(TEXTS, "opinions_3")
    with pytest.raises(EmbeddingCacheMissError, match="opinions_3: no cached embeddings") as error:
        cached.encode(TEXTS, "opinions_3")
    assert f"2 texts of stub-encoder@stub-revision-1@cpu ({path})" in str(error.value)
    assert encoder.calls == []
    assert not (tmp_path / "cache").exists()


def test_cache_only_without_a_cache_dir_fails() -> None:
    cached = CachedEncoder(StubEncoder(), cache_only=True)
    with pytest.raises(EmbeddingCacheMissError, match=r"\(no cache directory\)"):
        cached.encode(TEXTS, "opinions_3")


def test_cache_only_empty_input_never_calls_the_encoder(tmp_path: Path) -> None:
    encoder = StubEncoder()
    embeddings = CachedEncoder(encoder, tmp_path, cache_only=True).encode([], "sentences_9")
    assert embeddings.shape == (0, 0)
    assert encoder.calls == []


def test_run_cache_encoder_names_the_cache_of_a_run() -> None:
    encoder = RunCacheEncoder(
        "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder",
        "a01887015444f7599669c509447c5bdbce958916",
        "mps",
    )
    assert isinstance(encoder, SentenceEncoder)
    assert encoder.cache_identity == SERAFIM_IDENTITY
    assert encoder.cache_identity == sentence_encoder_identity(
        encoder.name, encoder.revision, encoder.device
    )
    assert encoder.runtime_info() == {"device": "mps"}
    with pytest.raises(EmbeddingCacheMissError, match="cannot encode 2 texts"):
        encoder.encode(TEXTS)
