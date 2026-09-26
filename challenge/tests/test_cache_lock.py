import dataclasses
import fcntl
from pathlib import Path

import pytest

from utils import nli_verifier_experiments as e3
from utils.cache_lock import CacheLockedError, acquire_writer_lock, lock_path
from utils.decision_models import AnswerCache, DecisionModelError
from utils.translation import Segmenter, TranslationStore

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nli_verifier.toml"
SEGMENTER = Segmenter(join_abbreviations=frozenset(), join_short_parts=False)
SIGNATURE = {"model": "fake/translator", "revision": "0" * 40}


def hold_lock_elsewhere(path: Path):
    lock = lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock, "ab")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_a_writer_can_reopen_its_own_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl"
    first = acquire_writer_lock(path)
    assert acquire_writer_lock(path) == first
    TranslationStore.open(path, SIGNATURE, SEGMENTER, writable=True)
    TranslationStore.open(path, SIGNATURE, SEGMENTER, writable=True)


def test_a_second_writer_is_refused_before_the_tail_is_cut(tmp_path: Path) -> None:
    path = tmp_path / "translations.jsonl"
    path.write_bytes(b'{"key": "cut')
    handle = hold_lock_elsewhere(path)
    try:
        with pytest.raises(CacheLockedError, match="one writer at a time"):
            acquire_writer_lock(path)
        with pytest.raises(SystemExit, match="open for writing in another process"):
            TranslationStore.open(path, SIGNATURE, SEGMENTER, writable=True)
        assert path.read_bytes() == b'{"key": "cut'
        reader = TranslationStore.open(path, SIGNATURE, SEGMENTER)
        assert reader.incomplete_tail_bytes == len(b'{"key": "cut')
    finally:
        handle.close()


def test_laya_and_nli_caches_take_the_writer_lock(tmp_path: Path) -> None:
    answers = tmp_path / "laya.jsonl"
    answers.write_bytes(b'{"key": "cut')
    handle = hold_lock_elsewhere(answers)
    try:
        with pytest.raises(DecisionModelError, match="one writer at a time"):
            AnswerCache.open(answers, "laya@x")
        assert answers.read_bytes() == b'{"key": "cut'
    finally:
        handle.close()
    config = dataclasses.replace(e3.load_config(CONFIG), cache_dir=tmp_path)
    spec = config.scorers["xnli_mdeberta_en_m2m100"]
    logits = tmp_path / f"nli_{spec.key}_{spec.revision[:12]}_cpu.jsonl"
    handle = hold_lock_elsewhere(logits)
    try:
        with pytest.raises(SystemExit, match="one writer at a time"):
            e3.open_logit_cache(config, spec, "cpu")
    finally:
        handle.close()
