import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from experiments.verifier.decision_models import (
    JEV_ENDPOINT,
    AnswerCache,
    DecisionModelError,
    FakeDecisionModel,
    HttpResponse,
    JevDecisionModel,
    JevRequestError,
    JevSpec,
    LayaDecisionModel,
    LayaSpec,
    MissingApiKeyError,
    RateLimiter,
    ReplayDecisionModel,
    ReplayMissError,
    TransportError,
    choice_question,
    fit_state,
    jev_request_body,
    jev_request_key,
    noul_question,
    parse_answer,
    score_question,
    serialize_state,
    urllib_transport,
)

SECRET = "sk-test-0123456789abcdef"
ENV = "TYPESAFE_API_KEY"
NLI = choice_question(
    "p1_nli",
    "What is the relationship between `premise` and `hypothesis`?",
    [
        ("entailment", "the premise implies the hypothesis is true"),
        ("neutral", "the premise neither implies nor contradicts the hypothesis"),
        ("contradiction", "the premise implies the hypothesis is false"),
    ],
)
NLI_REVERSED = choice_question(
    "p2_nli_reversed",
    NLI.instructions,
    list(reversed(list(zip(NLI.option_names, NLI.option_texts, strict=True)))),
)
INFERABLE = noul_question("p3_inferable", "Can the `hypothesis` be inferred from the `premise`?")
COVERAGE = score_question(
    "p7_coverage",
    "How much of the `hypothesis` is stated in the `premise`?",
    ["none of it", "a small part of it", "about half of it", "most of it", "all of it"],
)
QUESTIONS = [NLI, NLI_REVERSED, INFERABLE, COVERAGE]
STATE = {"premise": "O deputado defendeu o projeto.", "hypothesis": "O deputado apoia o projeto."}


def jev_answer(question, index: int) -> dict:
    if question.type == "noul":
        return {"type": "noul", "noul": 0.25 + 0.1 * index}
    names = list(question.option_names)
    values = [1.0 / len(names)] * len(names)
    answer = {
        "type": question.type,
        "probabilities": dict(zip(names, values, strict=True)),
        "confidence": 0.5,
    }
    if question.type == "choice":
        answer["choice"] = names[0]
    else:
        answer["score"] = 2.0
        answer["legend"] = {str(i): text for i, text in enumerate(question.option_texts)}
    return answer


def jev_response(questions, model: str = "jev-1.13.0", tokens: int = 300) -> bytes:
    body = {
        "model": model,
        "answers": {q.key: jev_answer(q, i) for i, q in enumerate(questions)},
        "usage": {"input_tokens": tokens, "output_tokens": 20},
    }
    return json.dumps(body).encode()


@dataclass
class FakeTransport:
    responses: list[HttpResponse | Exception]
    calls: list[tuple[str, bytes, dict[str, str], float]] = field(default_factory=list)

    def __call__(self, url, body, headers, timeout) -> HttpResponse:
        self.calls.append((url, body, dict(headers), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@dataclass
class FakeClock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def spec(tmp_path: Path, **changes) -> JevSpec:
    values = {
        "model_version": "jev-1.13.0",
        "endpoint": "https://api.typesafe.ai/v1/systemone",
        "api_key_env": ENV,
        "timeout_seconds": 30.0,
        "max_retries": 3,
        "backoff_initial_seconds": 2.0,
        "backoff_max_seconds": 60.0,
        "requests_per_minute": 0.0,
        "max_concurrency": 1,
        "price_usd_per_million_input_tokens": 0.042,
        "cache_dir": tmp_path,
    }
    return JevSpec(**{**values, **changes})


def ok(questions, **kwargs) -> HttpResponse:
    return HttpResponse(200, {}, jev_response(questions, **kwargs))


def test_question_payloads_keep_option_order():
    forward, reverse = NLI.payload(), NLI_REVERSED.payload()
    assert list(forward["criteria"]) == ["entailment", "neutral", "contradiction"]
    assert list(reverse["criteria"]) == ["contradiction", "neutral", "entailment"]
    assert NLI.payload_json() != NLI_REVERSED.payload_json()
    assert INFERABLE.payload() == {"type": "noul", "instructions": INFERABLE.instructions}
    assert COVERAGE.payload()["criteria"][0] == "none of it"
    with pytest.raises(DecisionModelError):
        score_question("bad", "x?", ["only one"])


def test_parse_answer_maps_every_type():
    noul = parse_answer(INFERABLE, {"type": "noul", "noul": 0.8}, 10)
    assert noul.probabilities == pytest.approx({"false": 0.2, "true": 0.8})
    assert noul.choice == "true"
    choice = parse_answer(
        NLI_REVERSED,
        {
            "type": "choice",
            "probabilities": {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},
        },
        None,
    )
    assert list(choice.probabilities) == ["contradiction", "neutral", "entailment"]
    assert choice.choice == "entailment"
    score = parse_answer(
        COVERAGE, {"type": "score", "probabilities": {str(i): 0.2 for i in range(5)}}, None
    )
    assert score.expected_level() == pytest.approx(2.0)
    with pytest.raises(DecisionModelError):
        parse_answer(NLI, {"type": "choice", "probabilities": {"yes": 1.0}}, None)
    with pytest.raises(DecisionModelError):
        parse_answer(NLI, {"type": "noul", "noul": 0.5}, None)


def test_fake_model_is_deterministic_and_normalized():
    model = FakeDecisionModel()
    first = model.predict_batch([STATE, {**STATE, "premise": "outro"}], QUESTIONS)
    second = model.predict_batch([STATE], QUESTIONS)
    assert first[0]["p1_nli"].probabilities == second[0]["p1_nli"].probabilities
    assert first[0]["p1_nli"].probabilities != first[1]["p1_nli"].probabilities
    for answers in first:
        for answer in answers.values():
            assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert model.calls == [
        (2, tuple(q.key for q in QUESTIONS)),
        (1, tuple(q.key for q in QUESTIONS)),
    ]


def words(text: str) -> int:
    return len(text.split())


def test_fit_state_cuts_only_the_premise_at_whitespace():
    state = {"premise": "um dois três quatro cinco seis sete oito", "hypothesis": "a b c"}
    whole = fit_state(state, "premise", 100, words)
    assert not whole.truncated and whole.state == state
    room = words(serialize_state({"premise": "um dois três", "hypothesis": "a b c"}))
    fitted = fit_state(state, "premise", room, words)
    assert fitted.truncated and not fitted.overflow
    assert fitted.state["hypothesis"] == "a b c"
    assert fitted.state["premise"] == "um dois três"
    assert fitted.fitted_tokens <= room
    assert fitted.chars_kept == len("um dois três")
    tight = fit_state(state, "premise", 1, words)
    assert tight.overflow and tight.state["premise"] == ""


def test_answer_cache_round_trip_and_isolation(tmp_path):
    path = tmp_path / "laya.jsonl"
    cache = AnswerCache.open(path, "sig-a")
    key = cache.key(NLI, STATE)
    assert key != cache.key(NLI_REVERSED, STATE)
    cache.append([{"key": key, "logits": [1.0, 0.0, -1.0]}])
    AnswerCache.open(path, "sig-b").append([{"key": "other", "logits": [0.0]}])
    with open(path, "a") as f:
        f.write('{"signature": "sig-a", "key": "cut')
    reopened = AnswerCache.open(path, "sig-a")
    assert list(reopened.entries) == [key]
    assert reopened.other_signature_lines == 1
    assert reopened.incomplete_tail_bytes > 0
    assert path.read_bytes().endswith(b"\n")


def test_missing_key_error_names_the_variable_and_no_secret(tmp_path):
    with pytest.raises(MissingApiKeyError) as error:
        JevDecisionModel(spec(tmp_path), transport=FakeTransport([]), environ={})
    message = str(error.value)
    assert ENV in message
    assert SECRET not in message
    with pytest.raises(MissingApiKeyError):
        JevDecisionModel(spec(tmp_path), transport=FakeTransport([]), environ={ENV: ""})


def test_jev_request_construction_and_replay_record(tmp_path):
    transport = FakeTransport([ok(QUESTIONS)])
    clock = FakeClock()
    model = JevDecisionModel(
        spec(tmp_path),
        transport=transport,
        environ={ENV: SECRET},
        clock=clock.clock,
        sleep=clock.sleep,
    )
    answers = model.predict_batch([STATE], QUESTIONS)
    url, body, headers, timeout = transport.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert headers["Content-Type"] == "application/json"
    sent = json.loads(body)
    assert sent == {
        "state": STATE,
        "model": "jev-1.13.0",
        "questions": {q.key: q.payload() for q in QUESTIONS},
    }
    assert list(sent["questions"]["p2_nli_reversed"]["criteria"])[0] == "contradiction"
    assert body == jev_request_body("jev-1.13.0", STATE, QUESTIONS)
    assert answers[0]["p3_inferable"].probabilities["true"] == pytest.approx(0.45)
    assert answers[0]["p1_nli"].input_tokens == 300
    cache_text = model.spec.cache_path.read_text()
    assert SECRET not in cache_text
    assert "Authorization" not in cache_text
    line = json.loads(cache_text.splitlines()[0])
    assert line["key"] == jev_request_key("jev-1.13.0", body)
    assert line["request"] == sent and line["status"] == 200
    assert SECRET not in json.dumps(model.describe()) and SECRET not in repr(model)
    again = model.predict_batch([STATE], QUESTIONS)
    assert len(transport.calls) == 1
    assert again[0]["p1_nli"].probabilities == answers[0]["p1_nli"].probabilities


def test_jev_retries_honor_retry_after_and_backoff(tmp_path):
    transport = FakeTransport(
        [
            HttpResponse(429, {"retry-after": "7"}, b'{"error": "rate"}'),
            TransportError("ConnectionResetError: reset"),
            HttpResponse(529, {}, b"overloaded"),
            ok(QUESTIONS),
        ]
    )
    clock = FakeClock()
    model = JevDecisionModel(
        spec(tmp_path),
        transport=transport,
        environ={ENV: SECRET},
        clock=clock.clock,
        sleep=clock.sleep,
    )
    model.predict_batch([STATE], QUESTIONS)
    assert clock.sleeps == [7.0, 4.0, 8.0]
    lines = [json.loads(line) for line in model.spec.cache_path.read_text().splitlines()]
    assert [line["status"] for line in lines] == [429, None, 529, 200]
    assert model.describe()["counters"]["retries"] == 3


def test_jev_does_not_retry_client_errors_and_redacts(tmp_path):
    body = json.dumps({"detail": f"bad key {SECRET}"}).encode()
    transport = FakeTransport([HttpResponse(401, {}, body)])
    model = JevDecisionModel(spec(tmp_path), transport=transport, environ={ENV: SECRET})
    with pytest.raises(JevRequestError) as error:
        model.predict_batch([STATE], QUESTIONS)
    assert "HTTP 401" in str(error.value)
    assert SECRET not in str(error.value)
    assert len(transport.calls) == 1


def test_jev_refuses_an_unpinned_answer(tmp_path):
    transport = FakeTransport([ok(QUESTIONS, model="jev-1.14.0")])
    model = JevDecisionModel(spec(tmp_path), transport=transport, environ={ENV: SECRET})
    with pytest.raises(JevRequestError, match="pinned"):
        model.predict_batch([STATE], QUESTIONS)


def test_jev_gives_up_after_max_retries(tmp_path):
    transport = FakeTransport([HttpResponse(529, {}, b"") for _ in range(4)])
    clock = FakeClock()
    model = JevDecisionModel(
        spec(tmp_path),
        transport=transport,
        environ={ENV: SECRET},
        clock=clock.clock,
        sleep=clock.sleep,
    )
    with pytest.raises(JevRequestError, match="gave up after 4 attempts"):
        model.predict_batch([STATE], QUESTIONS)
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_replay_answers_from_the_record_and_fails_on_a_miss(tmp_path):
    live = JevDecisionModel(
        spec(tmp_path), transport=FakeTransport([ok(QUESTIONS)]), environ={ENV: SECRET}
    )
    recorded = live.predict_batch([STATE], QUESTIONS)
    replay = ReplayDecisionModel(spec(tmp_path))
    replayed = replay.predict_batch([STATE], QUESTIONS)
    assert replayed[0]["p7_coverage"].probabilities == recorded[0]["p7_coverage"].probabilities
    with pytest.raises(ReplayMissError) as error:
        replay.predict_batch([{**STATE, "premise": "não gravado"}], QUESTIONS)
    assert "1 of 1" in str(error.value) and ENV in str(error.value)
    with pytest.raises(ReplayMissError):
        replay.predict_batch([STATE], QUESTIONS[:2])


def test_jev_resumes_after_a_cut_replay_line(tmp_path):
    first = JevDecisionModel(
        spec(tmp_path), transport=FakeTransport([ok(QUESTIONS)]), environ={ENV: SECRET}
    )
    first.predict_batch([STATE], QUESTIONS)
    with open(first.spec.cache_path, "a") as f:
        f.write('{"key": "cut", "status": 20')
    other = {**STATE, "premise": "A comissão aprovou o parecer."}
    second = JevDecisionModel(
        spec(tmp_path), transport=FakeTransport([ok(QUESTIONS)]), environ={ENV: SECRET}
    )
    assert second.cache.incomplete_tail_bytes > 0
    second.predict_batch([other], QUESTIONS)
    replay = ReplayDecisionModel(spec(tmp_path))
    assert replay.cache.incomplete_tail_bytes == 0 and replay.cache.lines == 2
    assert len(replay.predict_batch([STATE, other], QUESTIONS)) == 2


@dataclass(frozen=True)
class Device:
    type: str

    def __str__(self) -> str:
        return self.type


@dataclass
class FallbackAgent:
    device: Device
    temperature: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature_by_options: dict[str, float] = field(default_factory=dict)

    def predict_batch(self, states, questions):
        self.device = Device("cpu")
        return [{"answers": {}, "usage": {"input_tokens": 0}} for _ in states]


def test_laya_refuses_a_device_change_during_a_forward_pass(tmp_path):
    model = LayaDecisionModel.__new__(LayaDecisionModel)
    model.spec = LayaSpec(
        repo_id="convaiinnovations/laya",
        revision="0" * 40,
        subfolder="multilingual",
        device="mps",
        batch_size=16,
        max_length=1024,
        head_max_length=256,
        truncate_key="premise",
        cache_dir=tmp_path,
    )
    model.name = "convaiinnovations/laya/multilingual"
    model.agent = FallbackAgent(Device("mps"))
    model.captured = []
    model.cache = AnswerCache.open(tmp_path / "laya.jsonl", "sig")
    model.counters = {"computed": 0, "forward_seconds": 0.0, "batches": 0}
    fitted = [fit_state(STATE, "premise", 10_000, words)]
    with pytest.raises(DecisionModelError, match="moved the model from mps to cpu"):
        model.compute(NLI, fitted, [model.cache.key(NLI, STATE)])
    assert not model.cache.path.exists() and model.cache.entries == {}


def test_the_real_transport_is_blocked_in_tests():
    with pytest.raises(RuntimeError, match="must not open network connections"):
        urllib_transport(JEV_ENDPOINT, b"{}", {"Content-Type": "application/json"}, 1.0)


def test_rate_limiter_spaces_request_starts():
    clock = FakeClock()
    limiter = RateLimiter(60.0, clock.clock, clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.sleeps == [1.0, 1.0]
