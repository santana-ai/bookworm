"""The ``code`` section of the reports: the source files a run depends on and their sha256.

Each file is recorded under its path relative to the repository root, such as
``experiments/src/experiments/udv/calibrate_threshold.py`` or
``bookworm/src/bookworm/udv/quotes.py``. Reports written before the repository was reorganized
record the old ``utils/<name>.py`` paths; ``docs/path_map.md`` maps them to the current files.
A package, or a directory, stands for every ``.py`` file directly inside it.
"""

from pathlib import Path
from types import ModuleType
from typing import cast

from bookworm import sha256_of_file

PROJECT_DIR = Path(__file__).resolve().parents[3]
REPOSITORY_DIR = PROJECT_DIR.parent

Source = ModuleType | Path


def module_path(module: ModuleType) -> Path:
    return Path(cast(str, module.__file__))


def source_path(source: Source) -> Path:
    return module_path(source) if isinstance(source, ModuleType) else source


def expand_source(source: Source) -> list[Path]:
    path = source_path(source)
    if isinstance(source, ModuleType) and hasattr(source, "__path__"):
        path = path.parent
    return sorted(path.glob("*.py")) if path.is_dir() else [path]


def source_label(source: Source) -> str:
    path = source_path(source).resolve()
    if path.is_relative_to(REPOSITORY_DIR):
        return path.relative_to(REPOSITORY_DIR).as_posix()
    return path.as_posix()


def source_hashes(*sources: Source) -> dict[str, str]:
    return {source_label(source): sha256_of_file(source_path(source)) for source in sources}


def code_section(*sources: Source) -> dict[str, str]:
    """The ``code`` section of a report: the sha256 of each source file, in the order given."""
    paths = [path for source in sources for path in expand_source(source)]
    return {source_label(path): sha256_of_file(path) for path in paths}
