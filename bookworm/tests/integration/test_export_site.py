import json
import statistics
from collections import Counter
from pathlib import Path

import pytest
from conftest import directory_state, udv_artifact_path

from bookworm import (
    CachedEncoder,
    HearingRecord,
    RunCacheEncoder,
    display_title,
    export_site,
    load_udv_jsonl,
)
from bookworm.data.io import JsonObject
from bookworm.udv.site import TITLE_ELLIPSIS, TITLE_MAX_CHARS
from bookworm.udv.verify import coverage_hearings

pytestmark = pytest.mark.dataset

RUN_NAME = "udv_v1"
TIER_TOTALS = {
    "quote_found": 277,
    "semantic_match_high": 1785,
    "semantic_match_weak": 43,
    "no_evidence": 8,
    "person_not_resolved": 90,
}
SUPPORT_TYPE_TOTALS = {
    "direct_quote": 277,
    "semantic_with_short_quote": 111,
    "semantic_similarity": 1717,
}


@pytest.fixture(scope="module")
def coverage(udv_artifacts_dir: Path) -> JsonObject:
    path = udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}_coverage.json")
    payload: JsonObject = json.loads(path.read_text(encoding="utf-8"))
    return payload


@pytest.fixture(scope="module")
def site_dir(
    tmp_path_factory: pytest.TempPathFactory,
    coverage: JsonObject,
    lds_hearings: list[HearingRecord],
    udv_artifacts_dir: Path,
    embedding_cache_dir: Path,
    split_artifacts_dir: Path,
) -> Path:
    records = load_udv_jsonl(udv_artifact_path(udv_artifacts_dir, f"{RUN_NAME}.jsonl"))
    manifest = json.loads((split_artifacts_dir / "temporal_v1.json").read_text(encoding="utf-8"))
    encoder = RunCacheEncoder(
        coverage["config"]["encoder"]["name"],
        coverage["config"]["encoder"]["revision"],
        coverage["encoder_runtime"]["device"],
    )
    before = directory_state(embedding_cache_dir)
    output = tmp_path_factory.mktemp("site")
    export_site(
        coverage_hearings(coverage, lds_hearings),
        records,
        CachedEncoder(encoder, embedding_cache_dir, cache_only=True),
        output,
        run_name=RUN_NAME,
        pipeline=coverage["pipeline"],
        split_manifest=manifest,
    )
    assert directory_state(embedding_cache_dir) == before
    return output


@pytest.fixture(scope="module")
def index(site_dir: Path) -> JsonObject:
    payload: JsonObject = json.loads((site_dir / "index.json").read_text(encoding="utf-8"))
    return payload


def test_every_hearing_has_a_file_listed_in_the_index(site_dir: Path, index: JsonObject) -> None:
    files = sorted(int(path.stem) for path in (site_dir / "hearings").glob("*.json"))
    ids = [entry["id"] for entry in index["hearings"]]
    assert len(files) == 206
    assert ids == files == sorted(ids)
    assert index["run"] == {
        "name": RUN_NAME,
        "encoder": "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder",
        "revision": "a01887015444f7599669c509447c5bdbce958916",
        "threshold": 0.45,
    }


def test_index_totals_match_the_run(index: JsonObject, coverage: JsonObject) -> None:
    entries = index["hearings"]
    tiers: Counter[str] = Counter()
    support_types: Counter[str] = Counter()
    for entry in entries:
        tiers.update(entry["tiers"])
        support_types.update(entry["support_types"])
    assert dict(tiers) == TIER_TOTALS == coverage["opinions"]["by_tier"]
    assert dict(support_types) == SUPPORT_TYPE_TOTALS == coverage["evidence_support_types"]
    udvs = [entry["n_udvs"] for entry in entries]
    people = [entry["n_people"] for entry in entries]
    assert (min(udvs), statistics.median(udvs), max(udvs), sum(udvs)) == (3, 10, 31, 2203)
    assert (min(people), statistics.median(people), max(people)) == (2, 5, 12)
    assert all(len(entry["actors"]) == entry["n_people"] for entry in entries)
    assert Counter(entry["split"] for entry in entries) == {
        "train": 144,
        "validation": 32,
        "test": 30,
    }


def test_titles_are_headlines_within_the_limit(
    index: JsonObject, lds_hearings: list[HearingRecord]
) -> None:
    titles = [entry["title"] for entry in index["hearings"]]
    assert max(len(title) for title in titles) <= TITLE_MAX_CHARS
    assert sum(title.endswith(TITLE_ELLIPSIS) for title in titles) == 1
    assert len(set(titles)) == 206
    headlines = [
        display_title(hearing.materia, hearing.metadados.assunto, max_chars=10_000)
        for hearing in lds_hearings
    ]
    assert all(
        headline == hearing.materia.splitlines()[0]
        for headline, hearing in zip(headlines, lds_hearings, strict=True)
    )
    assert (min(map(len, headlines)), max(map(len, headlines))) == (58, 124)


def test_file_sizes(site_dir: Path) -> None:
    sizes = {int(path.stem): path.stat().st_size for path in (site_dir / "hearings").glob("*.json")}
    ordered = sorted(sizes.values())
    assert sum(ordered) == 46_836_254
    assert (ordered[0], statistics.median(ordered), ordered[-1]) == (79_531, 199_367, 2_388_858)
    assert min(sizes, key=sizes.__getitem__) == 67
    assert max(sizes, key=sizes.__getitem__) == 6
    assert (site_dir / "index.json").stat().st_size == 184_112


def test_hearing_70_entry(index: JsonObject) -> None:
    entry = next(entry for entry in index["hearings"] if entry["id"] == 70)
    assert entry["split"] == "train"
    assert entry["article_date"] == "2022-08-03"
    assert (entry["n_udvs"], entry["n_people"], entry["transcript_words"]) == (12, 6, 10243)
    assert sum(entry["tiers"].values()) == 12


def test_largest_file_is_the_longest_transcript(index: JsonObject) -> None:
    words = {entry["id"]: entry["transcript_words"] for entry in index["hearings"]}
    assert max(words, key=words.__getitem__) == 6
    assert words[6] == 147_728
