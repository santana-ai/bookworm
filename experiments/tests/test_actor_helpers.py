import json
from pathlib import Path

import pytest

from experiments.actors.chat import check_output_budget, strip_think, system_user_messages
from experiments.actors.io import Job, fingerprint, load_rows, read_jsonl_rows, run_resumable
from experiments.actors.simulation import linked_udvs, turn_owners


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines))


def test_read_jsonl_rows_skips_blank_and_invalid_lines(tmp_path, caplog):
    path = tmp_path / "rows.jsonl"
    write_lines(path, ['{"id": "a"}', "", "{broken", '{"id": "b"}'])
    rows = list(read_jsonl_rows(path))
    assert rows == [(1, {"id": "a"}), (4, {"id": "b"})]
    assert "ignoring invalid line 3" in caplog.text


def test_load_rows_is_empty_for_a_missing_file(tmp_path):
    assert load_rows(tmp_path / "missing.jsonl", "id") == {}


def test_run_resumable_recomputes_only_missing_or_stale_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_lines(
        path,
        [
            json.dumps({"id": "a", "value": 1, "fingerprint": "fa"}),
            json.dumps({"id": "b", "value": 2, "fingerprint": "old"}),
        ],
    )
    computed: list[str] = []

    def job(name: str, key: str) -> Job:
        def compute() -> dict[str, object]:
            computed.append(name)
            return {"id": name, "value": 10}

        return Job(id=name, label=name, fingerprint=key, compute=compute)

    rows = run_resumable(path, "id", [job("a", "fa"), job("b", "fb"), job("c", "fc")], 3)
    assert computed == ["b", "c"]
    assert [row["fingerprint"] for row in rows] == ["fa", "fb", "fc"]
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows


def test_fingerprint_is_a_short_order_sensitive_hash():
    assert len(fingerprint("a", 1)) == 16
    assert fingerprint("a", 1) != fingerprint(1, "a")


def test_strip_think_removes_a_closed_block_and_rejects_an_open_one():
    assert strip_think("<think>plan</think>\n Perfil.") == "Perfil."
    with pytest.raises(RuntimeError, match="unterminated"):
        strip_think("<think>plan")


def test_check_output_budget_rejects_a_full_budget():
    check_output_budget(9, 10)
    with pytest.raises(RuntimeError, match="max_output_tokens=10"):
        check_output_budget(10, 10)


def test_system_user_messages():
    assert system_user_messages("s", "u") == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_linked_udvs_follow_the_owner_of_the_evidence_turn():
    records = [
        {"actor": "Ana", "hearings": [{"hearing_id": 1, "turns": [{"turn_index": 0}]}]},
        {"actor": "Beto", "hearings": [{"hearing_id": 1, "turns": [{"turn_index": 1}]}]},
    ]
    udvs = [
        {"id": "u0", "hearing_id": 1, "evidence": {"speaker_turn": 0}},
        {"id": "u1", "hearing_id": 1, "evidence": {"speaker_turn": 1}},
        {"id": "u2", "hearing_id": 1, "evidence": None},
        {"id": "u3", "hearing_id": 2, "evidence": {"speaker_turn": 0}},
    ]
    linked = linked_udvs(udvs, turn_owners(records), {"Ana"}, {1})
    assert [(udv["id"], actor) for udv, actor in linked] == [("u0", "Ana")]
