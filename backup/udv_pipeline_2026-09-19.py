import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIDENCE_THRESHOLD = 0.25
EMBEDDING_MODEL_NAME = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder"

TURN_HEADER_PATTERN = re.compile(
    r"(?:O\s+SR\.|A\s+SRA\.)\s*([A-ZÀ-Ü][A-ZÀ-Ü\.\s]{0,60}?)\s*(?:\(([^)]*)\))?\s*-\s?"
)
QUOTE_PATTERN = re.compile(r'“([^”]{10,})”|"([^"]{10,})"')
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARTY_INFO_MARKERS = ("/", " - ")
STAGE_DIRECTION_PATTERN = re.compile(r"^(?:\([^()]*\)\s*)+$")


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text):
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def normalize_name(name):
    return strip_accents(name).upper().strip()


def split_into_turns(transcript):
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


def is_party_info(text):
    return any(marker in text for marker in PARTY_INFO_MARKERS)


def resolve_turn_name(turn):
    party_info = turn["party_info"]
    if ". " in party_info:
        head, tail = party_info.rsplit(". ", 1)
        if is_party_info(tail):
            return head.strip()
    return turn["raw_name"]


def turn_name_candidates(turn):
    candidates = [turn["raw_name"]]
    party_info = turn["party_info"]
    resolved_name = resolve_turn_name(turn)
    if resolved_name != turn["raw_name"]:
        candidates.append(resolved_name)
    elif party_info and not is_party_info(party_info):
        candidates.append(party_info)
    return candidates


def names_match(name_a, name_b):
    tokens_a = set(normalize_name(name_a).split())
    tokens_b = set(normalize_name(name_b).split())
    if len(tokens_a) <= 1 or len(tokens_b) <= 1:
        return tokens_a == tokens_b
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def matching_turns(name, turns):
    return [t for t in turns if any(names_match(name, c) for c in turn_name_candidates(t))]


def single_token_matching_turns(name, turns):
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


def resolve_person_speech(person, turns):
    matched_turns = matching_turns(person["nome"], turns)
    if not matched_turns and len(normalize_name(person["nome"]).split()) == 1:
        matched_turns = single_token_matching_turns(person["nome"], turns)
    speech = normalize_whitespace(" ".join(t["speech"] for t in matched_turns))
    return matched_turns, speech


def extract_quotes(opinion_text):
    return [
        normalize_whitespace(match.group(1) or match.group(2))
        for match in QUOTE_PATTERN.finditer(opinion_text)
    ]


def quote_prefix_pattern(prefix):
    left_boundary = "" if prefix[:1].isupper() else r"(?<!\w)"
    return re.compile(rf"{left_boundary}{re.escape(prefix)}(?!\w)")


def find_quote_evidence(quote, person_speech):
    words = quote.split()
    for prefix_length in (10, 6, 4, 3):
        prefix = " ".join(words[:prefix_length])
        if len(prefix) > 5 and quote_prefix_pattern(prefix).search(person_speech):
            return prefix
    return None


def find_opinion_quote_evidence(opinion_text, person_speech):
    for quote in extract_quotes(opinion_text):
        found = find_quote_evidence(quote, person_speech)
        if found is not None:
            return found
    return None


def split_sentences(speech):
    parts = SENTENCE_BOUNDARY_PATTERN.split(speech)
    return [
        normalize_whitespace(part)
        for part in parts
        if len(part.split()) >= 4 and not STAGE_DIRECTION_PATTERN.match(part)
    ]


def enclosing_sentence(prefix, speech):
    speech = normalize_whitespace(speech)
    hit = quote_prefix_pattern(prefix).search(speech)
    if hit is None:
        raise ValueError(f"prefix not found in speech: {prefix!r}")
    covered = []
    position = 0
    for part in SENTENCE_BOUNDARY_PATTERN.split(speech):
        end = position + len(part)
        if position < hit.end() and hit.start() < end:
            covered.append(part)
        position = end + 1
    return normalize_whitespace(" ".join(covered))


def locate_sentence_span(sentence, transcript, turns):
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


def best_semantic_match(opinion_text, sentences):
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


_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def best_embedding_match(opinion_text, sentences, model):
    if not sentences:
        return None, 0.0
    embeddings = model.encode([opinion_text] + sentences)
    similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])


def best_match_from_embeddings(opinion_embedding, sentence_embeddings, sentences):
    if not sentences:
        return None, 0.0
    similarities = cosine_similarity(
        opinion_embedding.reshape(1, -1), sentence_embeddings
    ).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])
