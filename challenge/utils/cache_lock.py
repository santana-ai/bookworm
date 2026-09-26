import fcntl
from pathlib import Path
from typing import IO

HELD: dict[Path, IO[bytes]] = {}


class CacheLockedError(RuntimeError):
    pass


def lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def acquire_writer_lock(path: Path) -> Path:
    lock = lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    target = lock.resolve()
    if target in HELD:
        return target
    handle = open(target, "ab")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise CacheLockedError(
            f"{path} is open for writing in another process (it holds the lock {target}); "
            "a cache file takes one writer at a time, so nothing was read or written"
        ) from error
    HELD[target] = handle
    return target
