import json
from datetime import date
from pathlib import Path

from utils import actor_simulation
from utils import generate_actor_profiles as profiles


def profiles_config(tmp_path: Path, speeches: Path) -> profiles.ProfilesConfig:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("Sistema.")
    (prompts / "user.md.j2").write_text(
        "{{ actor_label }}{% for h in hearings %} {{ h.date }}{% endfor %}"
    )
    return profiles.ProfilesConfig(
        speeches_path=speeches,
        lds_path=tmp_path / "lds.jsonl",
        lds_sha256="0" * 64,
        profiles_path=tmp_path / "out.jsonl",
        prompts_dir=prompts,
        system_profile_file="system.md",
        user_profile_file="user.md.j2",
        model="",
        device_map="auto",
        temperature=0.2,
        top_p=0.9,
        max_output_tokens=10,
        seed=42,
    )


def test_profile_dry_run_hashes_the_rendered_prompts(tmp_path):
    speeches = tmp_path / "speeches.jsonl"
    records = [
        {"actor": "Ana", "hearings": [{"hearing_id": 1, "turns": []}]},
        {"actor": "Beto", "hearings": [{"hearing_id": 2, "turns": []}]},
    ]
    speeches.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = profiles_config(tmp_path, speeches)
    prompts = profiles.load_prompts(config)
    metadata = {
        1: {"date": date(2024, 1, 2), "date_br": "02/01/2024", "assunto": "a"},
        2: {"date": date(2024, 1, 3), "date_br": "03/01/2024", "assunto": "b"},
    }
    report = profiles.dry_run_report(config, prompts, records, metadata)
    assert report["dry_run"] is True and report["model"] is None
    assert report["actors"] == 2
    assert report["user_prompt_chars"] == {"total": 29, "min": 14, "max": 15}
    assert report["inputs"]["speeches"]["sha256"] == profiles.sha256_of_file(speeches)
    expected = profiles.canonical_sha256(
        [["Ana", "Sistema.", "Ana 02/01/2024"], ["Beto", "Sistema.", "Beto 03/01/2024"]]
    )
    assert report["rendered_prompts_sha256"] == expected
    changed = {**metadata, 2: {**metadata[2], "date_br": "04/01/2024"}}
    other = profiles.dry_run_report(config, prompts, records, changed)
    assert other["rendered_prompts_sha256"] != expected


def test_canonical_sha256_ignores_key_order():
    first = actor_simulation.canonical_sha256({"a": 1, "b": [1, 2]})
    assert first == actor_simulation.canonical_sha256({"b": [1, 2], "a": 1})
    assert len(first) == 64
