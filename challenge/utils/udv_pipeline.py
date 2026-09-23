import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Record = dict[str, Any]

CONFIDENCE_THRESHOLD = 0.25
UDV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "udv.toml"

TURN_HEADER_PATTERN = re.compile(
    r"(?:O\s+SR\.|A\s+SRA\.)\s*([A-ZÀ-Ü][A-ZÀ-Ü\.\s]{0,60}?)\s*(?:\(([^)]*)\))?\s*-\s?"
)
DOUBLE_QUOTE_PATTERN = re.compile(r'“([^”]{10,})”|"([^"]{10,})"')
SINGLE_QUOTE_PATTERN = re.compile(r"(?<!\w)'([^']{10,})'(?!\w)")
DOUBLE_QUOTE_PATTERNS = (DOUBLE_QUOTE_PATTERN,)
QUOTE_PATTERNS = (DOUBLE_QUOTE_PATTERN, SINGLE_QUOTE_PATTERN)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[.!?]\))(?<!\(\.\.\.\))\s+")
PARTY_INFO_MARKERS = ("/", " - ")
STAGE_DIRECTION_PATTERN = re.compile(r"^(?:\([^()]*\)\s*)+$")
WORD_TOKEN_PATTERN = re.compile(r"\w+")
QUOTE_PREFIX_LENGTHS = (10, 6, 4, 3)
TRUSTED_PREFIX_WORDS = 6
MIN_SENTENCE_WORDS = 4


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def normalize_name(name: str) -> str:
    return strip_accents(name).upper().strip()


def split_into_turns(transcript: str) -> list[Record]:
    matches = list(TURN_HEADER_PATTERN.finditer(transcript))
    turns = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        raw_speech = transcript[start:end]
        speech = raw_speech.strip()
        speech_start = start + (len(raw_speech) - len(raw_speech.lstrip()))
        turns.append(
            {
                "turn_index": index,
                "raw_name": match.group(1).strip(),
                "party_info": (match.group(2) or "").strip(),
                "speech": speech,
                "start_char": speech_start,
                "end_char": speech_start + len(speech),
            }
        )
    return turns


def is_party_info(text: str) -> bool:
    return any(marker in text for marker in PARTY_INFO_MARKERS)


def resolve_turn_name(turn: Record) -> str:
    party_info = turn["party_info"]
    if ". " in party_info:
        head, tail = party_info.rsplit(". ", 1)
        if is_party_info(tail):
            return head.strip()
    return turn["raw_name"]


def turn_name_candidates(turn: Record) -> list[str]:
    candidates = [turn["raw_name"]]
    party_info = turn["party_info"]
    resolved_name = resolve_turn_name(turn)
    if resolved_name != turn["raw_name"]:
        candidates.append(resolved_name)
    elif party_info and not is_party_info(party_info):
        candidates.append(party_info)
    return candidates


def names_match(name_a: str, name_b: str) -> bool:
    tokens_a = set(normalize_name(name_a).split())
    tokens_b = set(normalize_name(name_b).split())
    if len(tokens_a) <= 1 or len(tokens_b) <= 1:
        return tokens_a == tokens_b
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def matching_turns(name: str, turns: list[Record]) -> list[Record]:
    return [t for t in turns if any(names_match(name, c) for c in turn_name_candidates(t))]


def single_token_matching_turns(name: str, turns: list[Record]) -> list[Record]:
    token = normalize_name(name)
    speakers = {
        normalize_name(candidate)
        for turn in turns
        for candidate in turn_name_candidates(turn)
        if token in normalize_name(candidate).split()
    }
    if len(speakers) != 1:
        return []
    speaker = speakers.pop()
    return [t for t in turns if any(normalize_name(c) == speaker for c in turn_name_candidates(t))]


def resolve_person_speech(person: Record, turns: list[Record]) -> tuple[list[Record], str]:
    matched_turns = matching_turns(person["nome"], turns)
    if not matched_turns and len(normalize_name(person["nome"]).split()) == 1:
        matched_turns = single_token_matching_turns(person["nome"], turns)
    speech = normalize_whitespace(" ".join(t["speech"] for t in matched_turns))
    return matched_turns, speech


def turn_text(turn: Record) -> str:
    return normalize_whitespace(turn["speech"])


def quoted_text(match: re.Match[str]) -> str:
    return next(group for group in match.groups() if group is not None)


def extract_quotes(
    opinion_text: str, patterns: tuple[re.Pattern[str], ...] = QUOTE_PATTERNS
) -> list[str]:
    matches = sorted(
        (match for pattern in patterns for match in pattern.finditer(opinion_text)),
        key=lambda match: (match.start(), -match.end()),
    )
    return [normalize_whitespace(quoted_text(match)) for match in matches]


def quote_prefix_pattern(prefix: str) -> re.Pattern[str]:
    first, rest = prefix[0], prefix[1:]
    lower, upper = first.lower(), first.upper()
    if lower == upper:
        head = rf"(?<!\w){re.escape(first)}"
    else:
        head = rf"(?:(?<!\w){re.escape(lower)}|{re.escape(upper)})"
    return re.compile(rf"{head}(?i:{re.escape(rest)})(?!\w)")


def quote_prefixes(quote: str) -> list[tuple[str, int]]:
    words = quote.split()
    prefixes = []
    for prefix_length in QUOTE_PREFIX_LENGTHS:
        prefix = " ".join(words[:prefix_length])
        if len(prefix) > 5:
            prefixes.append((prefix, min(prefix_length, len(words))))
    return prefixes


def find_quote_match(quote: str, person_speech: str) -> Record | None:
    for prefix, words in quote_prefixes(quote):
        if quote_prefix_pattern(prefix).search(person_speech):
            return {"prefix": prefix, "words": words}
    return None


def find_opinion_quote_match(opinion_text: str, person_speech: str) -> Record | None:
    for quote in extract_quotes(opinion_text):
        match = find_quote_match(quote, person_speech)
        if match is not None:
            return match
    return None


def is_trusted_quote(match: Record | None) -> bool:
    return match is not None and match["words"] >= TRUSTED_PREFIX_WORDS


def find_opinion_quote_evidence(opinion_text: str, person_speech: str) -> str | None:
    match = find_opinion_quote_match(opinion_text, person_speech)
    return match["prefix"] if is_trusted_quote(match) else None


def is_sentence(part: str) -> bool:
    return len(part.split()) >= MIN_SENTENCE_WORDS and not STAGE_DIRECTION_PATTERN.match(part)


def split_sentences(speech: str) -> list[str]:
    parts = SENTENCE_BOUNDARY_PATTERN.split(speech)
    return [normalize_whitespace(part) for part in parts if is_sentence(part)]


def split_turn_sentences(turns: list[Record]) -> list[Record]:
    return [
        {"text": sentence, "turn_index": turn["turn_index"]}
        for turn in turns
        for sentence in split_sentences(turn_text(turn))
    ]


def sentence_part_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))
    return spans


def sentences_agree(sentence: str, other_sentence: str) -> bool:
    return sentence == other_sentence or sentence in other_sentence or other_sentence in sentence


def enclosing_sentence(prefix: str, speech: str) -> str:
    speech = normalize_whitespace(speech)
    hit = quote_prefix_pattern(prefix).search(speech)
    if hit is None:
        raise ValueError(f"prefix not found in speech: {prefix!r}")
    return enclosing_turn_sentence(speech, hit.start(), hit.end())


def enclosing_turn_sentence(text: str, start: int, end: int) -> str:
    covered = [text[a:b] for a, b in sentence_part_spans(text) if a < end and start < b]
    return normalize_whitespace(" ".join(covered))


def find_prefix_occurrences(prefix: str, turns: list[Record]) -> list[Record]:
    pattern = quote_prefix_pattern(prefix)
    occurrences = []
    for turn in turns:
        text = turn_text(turn)
        for hit in pattern.finditer(text):
            occurrences.append(
                {
                    "turn_index": turn["turn_index"],
                    "start": hit.start(),
                    "end": hit.end(),
                    "sentence": enclosing_turn_sentence(text, hit.start(), hit.end()),
                }
            )
    return occurrences


def find_turn_quote_match(quote: str, turns: list[Record]) -> Record | None:
    for prefix, words in quote_prefixes(quote):
        occurrences = find_prefix_occurrences(prefix, turns)
        if occurrences:
            return {"prefix": prefix, "words": words, "occurrences": occurrences}
    return None


def word_tokens(text: str) -> set[str]:
    return set(WORD_TOKEN_PATTERN.findall(strip_accents(text).lower()))


def token_jaccard(text: str, other_text: str) -> float:
    tokens, other_tokens = word_tokens(text), word_tokens(other_text)
    union = tokens | other_tokens
    return len(tokens & other_tokens) / len(union) if union else 0.0


def choose_occurrence(occurrences: list[Record], opinion_text: str) -> Record:
    return max(
        occurrences, key=lambda occurrence: token_jaccard(occurrence["sentence"], opinion_text)
    )


def find_opinion_turn_quote_match(
    opinion_text: str,
    turns: list[Record],
    patterns: tuple[re.Pattern[str], ...] = QUOTE_PATTERNS,
) -> Record | None:
    best: Record | None = None
    for quote_index, quote in enumerate(extract_quotes(opinion_text, patterns)):
        match = find_turn_quote_match(quote, turns)
        if match is not None and (best is None or match["words"] > best["words"]):
            best = {**match, "quote_index": quote_index}
    if best is None:
        return None
    occurrences = best["occurrences"]
    occurrence = (
        choose_occurrence(occurrences, opinion_text) if is_trusted_quote(best) else occurrences[0]
    )
    return {
        "prefix": best["prefix"],
        "words": best["words"],
        "quote_index": best["quote_index"],
        "occurrence_count": len(occurrences),
        "occurrence_index": occurrences.index(occurrence),
        "turn_index": occurrence["turn_index"],
        "start": occurrence["start"],
        "end": occurrence["end"],
        "sentence": occurrence["sentence"],
    }


def locate_sentence_span(sentence: str, transcript: str, turns: list[Record]) -> Record | None:
    pattern = re.compile(r"\s+".join(re.escape(token) for token in sentence.split()))
    for turn in turns:
        match = pattern.search(transcript, turn["start_char"], turn["end_char"])
        if match is not None:
            return {
                "start_char": match.start(),
                "end_char": match.end(),
                "speaker_turn": turn["turn_index"],
            }
    return None


def locate_turn_sentence_span(
    sentence: str, transcript: str, turns: list[Record], turn_index: int
) -> Record | None:
    source_turns = [turn for turn in turns if turn["turn_index"] == turn_index]
    return locate_sentence_span(sentence, transcript, source_turns)


def best_semantic_match(opinion_text: str, sentences: list[str]) -> tuple[str | None, float]:
    if not sentences:
        return None, 0.0
    corpus = [opinion_text] + sentences
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return None, 0.0
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])


_embedding_models: dict[tuple[str, str], Any] = {}


def load_encoder_spec(config_path: Path = UDV_CONFIG_PATH) -> tuple[str, str]:
    with open(config_path, "rb") as f:
        encoder = tomllib.load(f)["encoder"]
    return encoder["name"], encoder["revision"]


def get_embedding_model(config_path: Path = UDV_CONFIG_PATH) -> Any:
    spec = load_encoder_spec(config_path)
    if spec not in _embedding_models:
        from sentence_transformers import SentenceTransformer

        name, revision = spec
        _embedding_models[spec] = SentenceTransformer(name, revision=revision)
    return _embedding_models[spec]


def best_embedding_match(
    opinion_text: str, sentences: list[str], model: Any
) -> tuple[str | None, float]:
    if not sentences:
        return None, 0.0
    embeddings = model.encode([opinion_text] + sentences)
    similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])


def best_match_from_embeddings(
    opinion_embedding: Any, sentence_embeddings: Any, sentences: list[str]
) -> tuple[str | None, float]:
    if not sentences:
        return None, 0.0
    similarities = cosine_similarity(
        opinion_embedding.reshape(1, -1), sentence_embeddings
    ).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])
