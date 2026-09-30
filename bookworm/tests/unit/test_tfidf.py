import numpy as np
import pytest
import sklearn

from bookworm import SentenceEncoder, TfidfEncoder

CORPUS = [
    "O setor precisa de prazos de transição mais longos.",
    "A fiscalização das cooperativas precisa ser reforçada.",
    "Os mestres da cultura popular precisam de reconhecimento.",
]


def test_fit_and_encode_shapes() -> None:
    encoder = TfidfEncoder.fit(CORPUS)
    matrix = encoder.encode(["prazos de transição", "cultura popular"])
    assert isinstance(encoder, SentenceEncoder)
    assert matrix.shape == (2, encoder.dimension)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_unknown_words_produce_a_zero_row() -> None:
    matrix = TfidfEncoder.fit(CORPUS).encode(["xyzzy plugh"])
    assert not matrix.any()


def test_empty_input_keeps_the_vocabulary_dimension() -> None:
    encoder = TfidfEncoder.fit(CORPUS)
    assert encoder.encode([]).shape == (0, encoder.dimension)


def test_encoding_is_deterministic() -> None:
    first = TfidfEncoder.fit(CORPUS)
    second = TfidfEncoder.fit(list(CORPUS))
    assert first.cache_identity == second.cache_identity
    assert np.array_equal(first.encode(CORPUS), second.encode(CORPUS))


def test_identity_tracks_corpus_and_parameters() -> None:
    base = TfidfEncoder.fit(CORPUS)
    assert TfidfEncoder.fit(CORPUS[:2]).cache_identity != base.cache_identity
    assert TfidfEncoder.fit(CORPUS, max_features=5).cache_identity != base.cache_identity
    assert base.cache_identity == f"tfidf@scikit-learn-{sklearn.__version__}@{base.corpus_sha256}"


def test_max_features_limits_the_dimension() -> None:
    assert TfidfEncoder.fit(CORPUS, max_features=5).dimension == 5


def test_names_and_runtime_info() -> None:
    encoder = TfidfEncoder.fit(CORPUS)
    assert encoder.name == "tfidf"
    assert encoder.revision == f"scikit-learn-{sklearn.__version__}"
    assert encoder.runtime_info() == {
        "device": "cpu",
        "max_seq_length": None,
        "embedding_dimension": encoder.dimension,
    }


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty vocabulary"):
        TfidfEncoder.fit([])
