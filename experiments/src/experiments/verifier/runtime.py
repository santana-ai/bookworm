"""Clock, progress and append-only file helpers shared by the long-running verifier commands."""

import time
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def incomplete_tail(data: bytes) -> int:
    """Length of a trailing line cut short by an interrupted append."""
    if not data or data.endswith(b"\n"):
        return 0
    return len(data) - (data.rfind(b"\n") + 1)


def throughput(done: int, total: int, started: float) -> tuple[float, float]:
    """Items per second since `started` and the estimated minutes left."""
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0
    minutes_left = (total - done) / rate / 60 if rate else float("nan")
    return rate, minutes_left


def package_files(module_file: str, entry_point: str) -> tuple[Path, ...]:
    """The entry-point module next to a package and every module of that package."""
    package_dir = Path(module_file).resolve().parent
    return (package_dir.parent / entry_point, *sorted(package_dir.glob("*.py")))
