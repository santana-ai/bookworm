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
        turns.append(
            {
                "raw_name": match.group(1).strip(),
                "party_info": (match.group(2) or "").strip(),
                "speech": transcript[start:end].strip(),
            }
        )
    return turns


def resolve_turn_name(turn):
    if "." in turn["party_info"]:
        return turn["party_info"].split(".")[0].strip()
    return turn["raw_name"]


def names_match(name_a, name_b):
    tokens_a = set(normalize_name(name_a).split())
    tokens_b = set(normalize_name(name_b).split())
    if len(tokens_a) <= 1 or len(tokens_b) <= 1:
        return tokens_a == tokens_b
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def resolve_person_speech(pessoa, turns):
    matched_turns = [t for t in turns if names_match(pessoa["nome"], t["resolved_name"])]
    speech = normalize_whitespace(" ".join(t["speech"] for t in matched_turns))
    return matched_turns, speech


def extract_quote(opinion_text):
    match = QUOTE_PATTERN.search(opinion_text)
    if not match:
        return None
    return normalize_whitespace(match.group(1) or match.group(2))


def find_quote_evidence(quote, person_speech):
    words = quote.split()
    for prefix_length in (10, 6, 4, 3):
        prefix = " ".join(words[:prefix_length])
        if len(prefix) > 5 and prefix in person_speech:
            return prefix
    return None


def split_sentences(speech):
    parts = re.split(r"(?<=[.!?])\s+", speech)
    return [normalize_whitespace(part) for part in parts if len(part.split()) >= 4]


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
    similarities = cosine_similarity(opinion_embedding.reshape(1, -1), sentence_embeddings).flatten()
    best_index = similarities.argmax()
    return sentences[best_index], float(similarities[best_index])
