import json
from pathlib import Path

from utils import udv_v2_downstream as downstream
from utils.dataset_io import Record


def udv(uid: str, hearing: int, tier: str, turn: int | None) -> Record:
    evidence = None if turn is None else {"speaker_turn": turn, "text": "t"}
    return {"id": uid, "hearing_id": hearing, "tier": tier, "evidence": evidence}


def test_evidence_turn_links_follow_the_owner_of_the_evidence_turn() -> None:
    udvs = [
        udv("a", 1, "quote_found", 3),
        udv("b", 1, "semantic_match_high", 4),
        udv("c", 2, "no_evidence", None),
    ]
    owners = {(1, 3): "Ana", (2, 4): "Bruno"}
    assert downstream.evidence_turn_links(udvs, owners) == {"a": "Ana"}


def test_link_comparison_counts_disagreements_and_profiled_splits() -> None:
    udvs = {
        "a": udv("a", 1, "quote_found", 3),
        "b": udv("b", 1, "semantic_match_high", 4),
        "c": udv("c", 2, "semantic_match_weak", 5),
    }
    library = {"a": "Ana", "b": "Bruno", "c": "Carla"}
    evidence = {"a": "Ana", "b": "Beatriz"}
    comparison = downstream.link_comparison(
        udvs, library, evidence, {"Ana", "Carla"}, {1: "test", 2: "train"}
    )
    assert (comparison["both_linked"], comparison["disagreements"]) == (2, 1)
    assert comparison["library_only"] == {"c": "Carla"}
    assert comparison["evidence_turn_only"] == {}
    library_counts = comparison["profiled_by_split"]["library"]
    assert library_counts["test"] == {
        "udvs": 1,
        "actors": 1,
        "hearings": 1,
        "by_tier": {"quote_found": 1},
    }
    assert library_counts["train"]["by_tier"] == {"semantic_match_weak": 1}
    assert comparison["profiled_by_split"]["evidence_turn"]["train"]["udvs"] == 0


def test_split_tier_counts_list_every_tier() -> None:
    counts = downstream.split_tier_counts(
        [udv("a", 1, "quote_found", 1), udv("b", 2, "no_evidence", None)],
        {1: "train", 2: "test"},
    )
    assert counts["train"]["by_tier"]["quote_found"] == 1
    assert counts["test"]["by_tier"]["no_evidence"] == 1
    assert counts["validation"] == {
        "udvs": 0,
        "hearings": 0,
        "by_tier": dict.fromkeys(downstream.TIERS, 0),
    }


def test_question_overlap_separates_tier_and_option_changes() -> None:
    def question(uid: str, tier: str, options: list[str]) -> Record:
        return {"udv_id": uid, "tier": tier, "options": options}

    v1 = {
        "test": [question("a", "quote_found", ["x"]), question("b", "semantic_match_high", ["y"])]
    }
    v2 = {
        "test": [question("a", "quote_found", ["x"]), question("b", "semantic_match_weak", ["z"])]
    }
    overlap = downstream.question_overlap(v1, v2)["test"]
    assert (overlap["same_udv"], overlap["identical"]) == (2, 1)
    assert (overlap["tier_changed"], overlap["options_changed"]) == (1, 1)


def test_artifact_section_reads_dotted_fields(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"counts": {"scored": 3}}))
    artifact = downstream.Artifact(
        "x", {"udv_v1": None, "udv_v2": str(path)}, ("counts.scored", "missing.key")
    )
    section = downstream.artifact_section(artifact)
    assert section["udv_v1"] is None
    assert section["udv_v2"] == {"counts.scored": 3, "missing.key": None}
