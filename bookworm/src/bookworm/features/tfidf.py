import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from bookworm.features.encoders import CACHE_TEXT_SEPARATOR, FloatMatrix

TFIDF_ENCODER_NAME = "tfidf"


class TfidfEncoder:
    def __init__(self, vectorizer: TfidfVectorizer, corpus_sha256: str) -> None:
        self._vectorizer = vectorizer
        self._corpus_sha256 = corpus_sha256
        self._dimension = len(vectorizer.vocabulary_)

    @classmethod
    def fit(cls, corpus: Iterable[str], *, max_features: int | None = None) -> "TfidfEncoder":
        texts = list(corpus)
        vectorizer = TfidfVectorizer(max_features=max_features, dtype=np.float32)
        vectorizer.fit(texts)
        digest = hashlib.sha256(json.dumps({"max_features": max_features}).encode())
        for text in texts:
            digest.update(text.encode())
            digest.update(CACHE_TEXT_SEPARATOR)
        return cls(vectorizer, digest.hexdigest())

    @property
    def name(self) -> str:
        return TFIDF_ENCODER_NAME

    @property
    def revision(self) -> str:
        return f"scikit-learn-{sklearn.__version__}"

    @property
    def corpus_sha256(self) -> str:
        return self._corpus_sha256

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def cache_identity(self) -> str:
        return f"{self.name}@{self.revision}@{self._corpus_sha256}"

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        matrix: FloatMatrix = self._vectorizer.transform(list(texts)).toarray()
        return matrix

    def runtime_info(self) -> dict[str, Any]:
        return {"device": "cpu", "max_seq_length": None, "embedding_dimension": self._dimension}
