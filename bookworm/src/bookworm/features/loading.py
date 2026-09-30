"""Construction of the configured sentence encoder, importing the optional extra lazily."""

from collections.abc import Callable, Iterable

from bookworm.config import EncoderSettings, SentenceTransformerSettings, TfidfSettings
from bookworm.errors import ConfigError
from bookworm.features.encoders import SentenceEncoder
from bookworm.features.tfidf import TfidfEncoder


def load_sentence_transformer_encoder(
    settings: SentenceTransformerSettings, seed: int
) -> SentenceEncoder:
    """Seed torch and build the configured encoder; needs the ``embeddings`` extra."""
    try:
        from bookworm.features import sentence_transformer
    except ModuleNotFoundError as error:
        raise ConfigError(
            f"encoder kind {settings.kind!r} needs the optional 'embeddings' extra: {error}"
        ) from error
    sentence_transformer.seed_torch(seed)
    return sentence_transformer.SentenceTransformerEncoder(
        settings.name, settings.revision, settings.device, settings.batch_size
    )


def load_encoder(
    settings: EncoderSettings, seed: int, tfidf_corpus: Callable[[], Iterable[str]]
) -> SentenceEncoder:
    """Fit TF-IDF on ``tfidf_corpus()`` or load the sentence-transformers model of ``settings``.

    The corpus is only built for a TF-IDF encoder.
    """
    if isinstance(settings, TfidfSettings):
        return TfidfEncoder.fit(tfidf_corpus(), max_features=settings.max_features)
    return load_sentence_transformer_encoder(settings, seed)
