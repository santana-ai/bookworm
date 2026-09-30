import json

import pytest
from pydantic import ValidationError

from bookworm import HearingMetadata, HearingRecord, Participant


def hearing_payload() -> dict[str, object]:
    return {
        "id": 7,
        "materia": "Texto da matéria",
        "metadados": {
            "assunto": "Assunto",
            "envolvidos": [{"nome": "Fulana", "cargo": "Convidada", "opinioes": ["Opinou."]}],
        },
        "transcricao": "A SRA. FULANA - Opinião.",
    }


def test_hearing_record_accepts_the_lds_shape() -> None:
    hearing = HearingRecord.model_validate_json(json.dumps(hearing_payload()))
    assert hearing.metadados == HearingMetadata(
        assunto="Assunto",
        envolvidos=(Participant(nome="Fulana", cargo="Convidada", opinioes=("Opinou.",)),),
    )


def test_hearing_record_rejects_unknown_keys() -> None:
    payload = hearing_payload()
    payload["date"] = "2024-01-01"
    with pytest.raises(ValidationError):
        HearingRecord.model_validate_json(json.dumps(payload))


def test_hearing_record_rejects_missing_keys() -> None:
    payload = hearing_payload()
    del payload["materia"]
    with pytest.raises(ValidationError):
        HearingRecord.model_validate_json(json.dumps(payload))


def test_hearing_record_does_not_coerce_types() -> None:
    payload = hearing_payload()
    payload["id"] = "7"
    with pytest.raises(ValidationError):
        HearingRecord.model_validate_json(json.dumps(payload))


def test_hearing_record_is_frozen() -> None:
    hearing = HearingRecord.model_validate_json(json.dumps(hearing_payload()))
    with pytest.raises(ValidationError):
        hearing.__setattr__("id", 8)
