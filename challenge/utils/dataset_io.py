import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

Record = dict[str, Any]


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_path(module: ModuleType) -> Path:
    return Path(cast(str, module.__file__))


def load_jsonl(path: Path) -> list[Record]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_gated_jsonl(path: Path, expected_sha256: str) -> list[Record]:
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise SystemExit(f"{path}: sha256 {actual} != expected {expected_sha256}")
    return load_jsonl(path)


def write_jsonl(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: Record | list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
