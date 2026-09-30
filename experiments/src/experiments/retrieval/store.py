"""On-disk vector and score store of the retrieval harness, keyed by text digests."""

import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

Record = dict[str, Any]

SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
DIGEST_BYTES = 32


def slug(text: str) -> str:
    return SLUG_PATTERN.sub("_", text).strip("_") or "empty"


def text_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def pair_digest(query: str, passage: str) -> bytes:
    return hashlib.sha256(query.encode() + b"\x1f" + passage.encode()).digest()


def save_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as f:
        np.save(f, array)
    os.replace(temporary, path)


class VectorStore:
    def __init__(self, directory: Path, metadata: Record, shard_size: int) -> None:
        self.directory = directory
        self.metadata = metadata
        self.shard_size = shard_size
        self.index: dict[bytes, tuple[int, int]] = {}
        self.shards: list[np.ndarray] = []
        self.shard_lengths: list[np.ndarray] = []
        self.pending_keys: list[bytes] = []
        self.pending_vectors: list[np.ndarray] = []
        self.pending_lengths: list[int] = []
        self.pending_sources: list[str] = []
        self.pending_index: dict[bytes, int] = {}
        self.provenance: Record = {}
        self.written_shards = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        self.check_metadata()
        self.load_shards()

    def check_metadata(self) -> None:
        path = self.directory / "store.json"
        if path.exists():
            with open(path) as f:
                stored = json.load(f)
            if stored["key"] != self.metadata["key"]:
                raise SystemExit(f"{path} belongs to another store: {stored['key']}")
            return
        with open(path, "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load_shards(self) -> None:
        for keys_path in sorted(self.directory.glob("shard_*_keys.npy")):
            vectors_path = keys_path.with_name(keys_path.name.replace("_keys", "_vectors"))
            keys = np.load(keys_path)
            lengths_path = keys_path.with_name(keys_path.name.replace("_keys", "_lengths"))
            vectors = np.load(vectors_path, mmap_mode="r")
            lengths = np.load(lengths_path)
            if not len(keys) == len(vectors) == len(lengths):
                raise SystemExit(f"{keys_path}: keys, vectors and lengths differ in size")
            shard = len(self.shards)
            self.shards.append(vectors)
            self.shard_lengths.append(lengths)
            for row, key in enumerate(keys):
                self.index.setdefault(bytes(key), (shard, row))

    def __len__(self) -> int:
        return len(self.index) + len(self.pending_index)

    def contains(self, key: bytes) -> bool:
        return key in self.index or key in self.pending_index

    def get(self, key: bytes) -> np.ndarray:
        if key in self.pending_index:
            return self.pending_vectors[self.pending_index[key]]
        shard, row = self.index[key]
        return np.asarray(self.shards[shard][row])

    def length(self, key: bytes) -> int:
        if key in self.pending_index:
            return self.pending_lengths[self.pending_index[key]]
        shard, row = self.index[key]
        return int(self.shard_lengths[shard][row])

    def gather(self, keys: list[bytes]) -> np.ndarray:
        return np.stack([self.get(key) for key in keys]) if keys else np.empty((0, 0))

    def add(self, keys: list[bytes], vectors: np.ndarray, lengths: list[int], source: str) -> None:
        for key, vector, length in zip(keys, vectors, lengths, strict=True):
            if self.contains(key):
                continue
            self.pending_index[key] = len(self.pending_vectors)
            self.pending_keys.append(key)
            self.pending_vectors.append(np.asarray(vector, dtype=np.float32))
            self.pending_lengths.append(int(length))
            self.pending_sources.append(source)
        if len(self.pending_keys) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending_keys:
            return
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        name = f"shard_{stamp}_{os.getpid()}"
        vectors = np.stack(self.pending_vectors).astype(np.float32)
        keys = np.frombuffer(b"".join(self.pending_keys), dtype=np.uint8).reshape(-1, DIGEST_BYTES)
        lengths = np.array(self.pending_lengths, dtype=np.int32)
        save_atomic(self.directory / f"{name}_vectors.npy", vectors)
        save_atomic(self.directory / f"{name}_lengths.npy", lengths)
        sidecar = {
            "rows": len(self.pending_keys),
            "rows_by_source": dict(sorted(Counter(self.pending_sources).items())),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **self.provenance,
        }
        with open(self.directory / f"{name}_provenance.json", "w") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)
        save_atomic(self.directory / f"{name}_keys.npy", keys)
        shard = len(self.shards)
        self.shards.append(np.load(self.directory / f"{name}_vectors.npy", mmap_mode="r"))
        self.shard_lengths.append(lengths)
        for row, key in enumerate(self.pending_keys):
            self.index.setdefault(key, (shard, row))
        self.pending_keys, self.pending_vectors, self.pending_index = [], [], {}
        self.pending_lengths, self.pending_sources = [], []
        self.written_shards += 1
