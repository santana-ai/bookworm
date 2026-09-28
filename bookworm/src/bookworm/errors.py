"""Exception hierarchy; the CLI turns every ``BookwormError`` into exit code 2."""

from pathlib import Path


class BookwormError(Exception):
    pass


class DatasetIntegrityError(BookwormError):
    def __init__(self, path: Path, actual_sha256: str, expected_sha256: str) -> None:
        self.path = path
        self.actual_sha256 = actual_sha256
        self.expected_sha256 = expected_sha256
        super().__init__(f"{path}: sha256 {actual_sha256} != expected {expected_sha256}")


class ConfigError(BookwormError):
    pass


class SplitError(BookwormError):
    pass


class EmbeddingCacheMissError(BookwormError):
    pass
