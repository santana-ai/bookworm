import json
import random
from pathlib import Path

from utils.udv_pipeline import (
    CONFIDENCE_THRESHOLD,
    best_semantic_match,
    find_opinion_quote_evidence,
    resolve_person_speech,
    split_into_turns,
    split_sentences,
)

SAMPLE_SIZE = 20
RANDOM_SEED = 42


def build_pools(lds_records):
    high_pool = []
    weak_pool = []

    for hearing in lds_records[:20]:
        turns = split_into_turns(hearing["transcricao"])
        for participant in hearing["metadados"]["envolvidos"]:
            matched_turns, speech = resolve_person_speech(participant, turns)
            if not matched_turns:
                continue
            sentences = split_sentences(speech)
            for opinion_text in participant["opinioes"]:
                if find_opinion_quote_evidence(opinion_text, speech) is not None:
                    continue
                best_sentence, score = best_semantic_match(opinion_text, sentences)
                if best_sentence is None:
                    continue
                entry = {
                    "hearing_id": hearing["id"],
                    "person": participant["nome"],
                    "opinion": opinion_text,
                    "evidence": best_sentence,
                    "score": score,
                }
                (high_pool if score >= CONFIDENCE_THRESHOLD else weak_pool).append(entry)

    return high_pool, weak_pool


def build_review_items(lds_records):
    random.seed(RANDOM_SEED)
    high_pool, weak_pool = build_pools(lds_records)
    sample_high = random.sample(high_pool, SAMPLE_SIZE)
    sample_weak = random.sample(weak_pool, SAMPLE_SIZE)

    items = []
    for entry in sample_high:
        items.append({"tier": "semantic_match_high", **entry, "judgment": None, "note": None})
    for entry in sample_weak:
        items.append({"tier": "semantic_match_weak", **entry, "judgment": None, "note": None})
    return items


if __name__ == "__main__":
    dataset_path = Path("dataset/PublicHearingBR_LDS.jsonl")
    output_path = Path(__file__).resolve().parent.parent / "udv_manual_review_template.json"
    with open(dataset_path) as f:
        lds_records = [json.loads(line) for line in f]
    review_items = build_review_items(lds_records)
    with open(output_path, "w") as f:
        json.dump(review_items, f, ensure_ascii=False, indent=2)
    print(f"{len(review_items)} itens escritos em {output_path}, prontos para revisão manual")
