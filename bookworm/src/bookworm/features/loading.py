"""Construction of the sentence-transformers encoder, importing the optional extra lazily."""

from bookworm.config import SentenceTransformerSettings
from bookworm.errors import ConfigError
from bookworm.features.encoders import SentenceEncoder


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
