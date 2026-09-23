from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from bookworm.features.encoders import FloatMatrix

AUTO_DEVICE = "auto"

ModelLoader = Callable[[str, str, str], SentenceTransformer]


def select_device(requested: str) -> str:
    if requested != AUTO_DEVICE:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)


def load_sentence_transformer(
    name: str, revision: str, device: str, *, local_files_only: bool = False
) -> SentenceTransformer:
    return SentenceTransformer(
        name, revision=revision, device=device, local_files_only=local_files_only
    )


class SentenceTransformerEncoder:
    def __init__(
        self,
        name: str,
        revision: str,
        device: str = AUTO_DEVICE,
        batch_size: int = 64,
        *,
        loader: ModelLoader = load_sentence_transformer,
    ) -> None:
        self._name = name
        self._revision = revision
        self._device = select_device(device)
        self._batch_size = batch_size
        self._loader = loader
        self._model: SentenceTransformer | None = None

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
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cache_identity(self) -> str:
        return f"{self._name}@{self._revision}@{self._device}"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = self._loader(self._name, self._revision, self._device)
        return self._model

    def encode(self, texts: Sequence[str]) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.model.get_embedding_dimension() or 0), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix: FloatMatrix = np.asarray(embeddings)
        return matrix

    def runtime_info(self) -> dict[str, Any]:
        return {
            "device": self._device,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_embedding_dimension(),
        }
