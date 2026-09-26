import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from bookworm.errors import EmbeddingCacheMissError

FloatMatrix = NDArray[np.floating[Any]]

CACHE_TEXT_SEPARATOR = b"\x1e"
CACHE_KEY_CHARS = 16


@runtime_checkable
class SentenceEncoder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def cache_identity(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> FloatMatrix: ...

    def runtime_info(self) -> dict[str, Any]: ...


def sentence_encoder_identity(name: str, revision: str, device: str) -> str:
    return f"{name}@{revision}@{device}"


def cache_key(identity: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256(identity.encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(CACHE_TEXT_SEPARATOR)
    return digest.hexdigest()


def cache_file_name(label: str, identity: str, texts: Sequence[str]) -> str:
    return f"{label}_{cache_key(identity, texts)[:CACHE_KEY_CHARS]}.npy"


def check_row_count(embeddings: FloatMatrix, texts: Sequence[str], source: str) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"{source}: expected {len(texts)} embedding rows, got shape {embeddings.shape}"
        )


class RunCacheEncoder:
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
    def device(self) -> str:
        return self._device

    @property
    def cache_identity(self) -> str:
        return sentence_encoder_identity(self._name, self._revision, self._device)

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        raise EmbeddingCacheMissError(
            f"{self.cache_identity} cannot encode {len(texts)} texts: it only names the "
            "embedding cache files of a finished run"
        )

    def runtime_info(self) -> dict[str, Any]:
        return {"device": self._device}


class CachedEncoder:
    def __init__(
        self,
        encoder: SentenceEncoder,
        cache_dir: Path | None = None,
        *,
        read_only: bool = False,
        cache_only: bool = False,
    ) -> None:
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.read_only = read_only or cache_only
        self.cache_only = cache_only

    def cache_path(self, texts: Sequence[str], label: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / cache_file_name(label, self.encoder.cache_identity, texts)

    def encode(self, texts: Sequence[str], label: str) -> FloatMatrix:
        if not texts:
            if self.cache_only:
                return np.empty((0, 0), dtype=np.float32)
            return self.encoder.encode([])
        path = self.cache_path(texts, label)
        if path is not None and path.exists():
            cached: FloatMatrix = np.load(path)
            check_row_count(cached, texts, path.name)
            return cached
        if self.cache_only:
            location = "no cache directory" if path is None else str(path)
            raise EmbeddingCacheMissError(
                f"{label}: no cached embeddings for {len(texts)} texts of "
                f"{self.encoder.cache_identity} ({location}); this command only reads the "
                "embedding cache written by build-udvs for the run"
            )
        embeddings = self.encoder.encode(texts)
        check_row_count(embeddings, texts, label)
        if path is not None and not self.read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, embeddings)
        return embeddings
