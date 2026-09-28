"""The retrievers of the harness: lexical, dense, cross-encoder rerankers and decision rerankers."""

import gc
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.common.hub_offline import pinned_weights_file
from experiments.common.transcript import WORD_TOKEN_PATTERN, strip_accents
from experiments.common.udv_run import UdvConfig, cache_key
from experiments.retrieval.data import (
    HearingData,
    Query,
    SpeakerContext,
    Unit,
    context_key,
)
from experiments.retrieval.store import VectorStore, pair_digest, slug, text_digest
from experiments.verifier.decision_models import (
    DecisionModel,
    LayaDecisionModel,
    LayaSpec,
    sha256_text,
)
from experiments.verifier.decision_scoring import BatteryQuestion

Record = dict[str, Any]

RERANK_OFFSET = 1.0e6


@dataclass(frozen=True)
class Ranking:
    order: np.ndarray
    sort_key: np.ndarray
    display: np.ndarray


@dataclass(frozen=True)
class DenseSpec:
    name: str
    model: str
    revision: str
    query_prefix: str
    passage_prefix: str
    max_seq_length: int
    batch_size: int
    token_budget: int
    production_cache: bool
    source: Record


@dataclass(frozen=True)
class RerankSpec:
    name: str
    model: str
    revision: str
    base: str
    top_k: int
    max_length: int
    batch_size: int
    source: Record


@dataclass(frozen=True)
class DecisionRerankSpec:
    name: str
    scorer: str
    base: str
    top_k: int
    question: BatteryQuestion
    laya: LayaSpec
    source: Record


@dataclass
class Runtime:
    device: str
    embeddings_dir: Path
    rerank_dir: Path
    shard_size: int
    udv_config: UdvConfig
    masked_texts_by_hearing: dict[int, list[str]] = field(default_factory=dict)


class Retriever(Protocol):
    retriever_id: str

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]: ...

    def describe(self) -> Record: ...

    def close(self) -> None: ...


def order_by(sort_key: np.ndarray) -> np.ndarray:
    return np.argsort(-sort_key, kind="stable")


def score_ranking(scores: np.ndarray) -> Ranking:
    scores = np.asarray(scores, dtype=np.float64)
    return Ranking(order_by(scores), scores, scores)


def contexts_of(hearing: HearingData, queries: list[Query]) -> list[SpeakerContext]:
    keys = list(dict.fromkeys(query.context_key for query in queries))
    return [hearing.contexts[key] for key in keys]


def hearing_context(hearing: HearingData) -> SpeakerContext:
    return hearing.contexts[context_key(hearing.hearing_id, None)]


def unit_rows(context: SpeakerContext, corpus: SpeakerContext, kind: str) -> list[int]:
    position = {unit.unit_id: row for row, unit in enumerate(corpus.units[kind])}
    return [position[unit.unit_id] for unit in context.units[kind]]


def bm25_tokens(text: str) -> list[str]:
    return WORD_TOKEN_PATTERN.findall(strip_accents(text).lower())


class TfidfRetriever:
    def __init__(self, retriever_id: str, params: Record, fit_scope: str) -> None:
        self.retriever_id = retriever_id
        self.params = params
        self.fit_scope = fit_scope
        self.counts: Counter[str] = Counter()

    def vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=self.params["analyzer"],
            token_pattern=self.params["token_pattern"],
            ngram_range=tuple(self.params["ngram_range"]),
            lowercase=self.params["lowercase"],
            strip_accents=self.params["strip_accents"],
            sublinear_tf=self.params["sublinear_tf"],
            min_df=self.params["min_df"],
            norm="l2",
        )

    def fit(self, texts: list[str]) -> TfidfVectorizer | None:
        vectorizer = self.vectorizer()
        try:
            vectorizer.fit(texts)
        except ValueError:
            self.counts["empty_vocabulary_fits"] += 1
            return None
        self.counts["fits"] += 1
        return vectorizer

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        corpus = hearing_context(hearing) if self.fit_scope == "hearing" else None
        shared = self.fit([u.text for u in corpus.units[kind]]) if corpus else None
        shared_matrix = (
            shared.transform([u.text for u in corpus.units[kind]]) if corpus and shared else None
        )
        rankings: dict[str, Ranking] = {}
        for context in contexts_of(hearing, queries):
            members = [query for query in queries if query.context_key == context.key]
            units = context.units[kind]
            if corpus is not None:
                vectorizer = shared
                matrix = (
                    shared_matrix[unit_rows(context, corpus, kind)]
                    if shared_matrix is not None
                    else None
                )
            else:
                vectorizer = self.fit([unit.text for unit in units])
                matrix = vectorizer.transform([u.text for u in units]) if vectorizer else None
            for query in members:
                if vectorizer is None or matrix is None:
                    scores = np.zeros(len(units))
                else:
                    query_vector = vectorizer.transform([query.text])
                    scores = sparse.csr_matrix(query_vector @ matrix.T).toarray().ravel()
                rankings[query.query_id] = score_ranking(scores)
        return rankings

    def describe(self) -> Record:
        return {
            "kind": "tfidf",
            "fit_scope": self.fit_scope,
            "fit_corpus": (
                "the units of the same kind built over every turn of the hearing"
                if self.fit_scope == "hearing"
                else "the candidate units of the speaker"
            ),
            "query_in_fit": False,
            "params": self.params,
            "counts": dict(self.counts),
        }

    def close(self) -> None:
        return None


class Bm25Retriever:
    def __init__(self, retriever_id: str, params: Record, fit_scope: str) -> None:
        self.retriever_id = retriever_id
        self.params = params
        self.fit_scope = fit_scope
        self.counts: Counter[str] = Counter()

    def model(self, texts: list[str]) -> BM25Okapi:
        self.counts["fits"] += 1
        return BM25Okapi(
            [bm25_tokens(text) for text in texts],
            k1=self.params["k1"],
            b=self.params["b"],
            epsilon=self.params["epsilon"],
        )

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        corpus = hearing_context(hearing) if self.fit_scope == "hearing" else None
        shared = self.model([u.text for u in corpus.units[kind]]) if corpus else None
        rankings: dict[str, Ranking] = {}
        for context in contexts_of(hearing, queries):
            members = [query for query in queries if query.context_key == context.key]
            if corpus is not None and shared is not None:
                rows = unit_rows(context, corpus, kind)
                for query in members:
                    scores = shared.get_batch_scores(bm25_tokens(query.text), rows)
                    rankings[query.query_id] = score_ranking(np.array(scores))
                continue
            local = self.model([unit.text for unit in context.units[kind]])
            for query in members:
                scores = local.get_scores(bm25_tokens(query.text))
                rankings[query.query_id] = score_ranking(np.array(scores))
        return rankings

    def describe(self) -> Record:
        return {
            "kind": "bm25",
            "implementation": "rank_bm25.BM25Okapi",
            "tokenizer": "bookworm strip_accents, lowercase, WORD_TOKEN_PATTERN (\\w+)",
            "fit_scope": self.fit_scope,
            "fit_corpus": (
                "the units of the same kind built over every turn of the hearing"
                if self.fit_scope == "hearing"
                else "the candidate units of the speaker"
            ),
            "params": self.params,
            "counts": dict(self.counts),
        }

    def close(self) -> None:
        return None


def prefix_directory(prefix: str) -> str:
    digest = hashlib.sha256(prefix.encode()).hexdigest()[:8]
    return f"prefix-{slug(prefix) if prefix else 'none'}-{digest}"


def native_max_seq_length(model: str, revision: str) -> int | None:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model, "sentence_bert_config.json", revision=revision)
    if not isinstance(path, str):
        return None
    with open(path) as f:
        return int(json.load(f)["max_seq_length"])


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def token_batches(lengths: list[int], batch_size: int, token_budget: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        widest = lengths[current[0]] if current else lengths[index]
        if current and (len(current) >= batch_size or (len(current) + 1) * widest > token_budget):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


class DenseEncoder:
    def __init__(self, spec: DenseSpec, runtime: Runtime) -> None:
        self.spec = spec
        self.runtime = runtime
        self.model: Any = None
        self.tokenizer_instance: Any = None
        self.native_max_seq_length = native_max_seq_length(spec.model, spec.revision)
        self.counts: Counter[str] = Counter()
        self.encode_seconds = 0.0
        self.load_seconds = 0.0
        self.imported_hearings: set[int] = set()
        base = runtime.embeddings_dir / slug(spec.model) / spec.revision[:12]
        self.stores = {
            role: VectorStore(
                base / prefix_directory(prefix) / f"msl{spec.max_seq_length}",
                {
                    "key": (
                        f"{spec.model}@{spec.revision}|prefix={prefix!r}|"
                        f"max_seq_length={spec.max_seq_length}"
                    ),
                    "model": spec.model,
                    "revision": spec.revision,
                    "prefix": prefix,
                    "max_seq_length": spec.max_seq_length,
                    "stored_vectors": "raw model output, float32, normalized only at scoring",
                },
                runtime.shard_size,
            )
            for role, prefix in (("query", spec.query_prefix), ("passage", spec.passage_prefix))
        }
        for store in self.stores.values():
            store.provenance = {
                "device": runtime.device,
                "encoded_rows_batch_size": spec.batch_size,
                "encoded_rows_token_budget": spec.token_budget,
                "precision": "float32",
            }

    def load(self) -> Any:
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            self.model = SentenceTransformer(
                self.spec.model,
                revision=self.spec.revision,
                device=self.runtime.device,
                local_files_only=True,
            )
            self.model.max_seq_length = self.spec.max_seq_length
            self.counts["model_loads"] += 1
            self.load_seconds += time.perf_counter() - started
        return self.model

    def production_ready(self) -> bool:
        config = self.runtime.udv_config
        return (
            self.spec.production_cache
            and self.spec.model == config.model_name
            and self.spec.revision == config.model_revision
            and self.spec.query_prefix == ""
            and self.spec.passage_prefix == ""
            and self.spec.max_seq_length == self.native_max_seq_length
        )

    def import_list(self, texts: list[str], label: str, role: str) -> None:
        if not texts:
            return
        config = self.runtime.udv_config
        digest = cache_key(texts, config, self.runtime.device)[:16]
        path = config.cache_dir / f"{label}_{digest}.npy"
        store = self.stores[role]
        keys = [text_digest(text) for text in texts]
        if all(store.contains(key) for key in keys):
            return
        if not path.exists():
            self.counts[f"production_{role}_lists_missing"] += 1
            return
        try:
            vectors = np.load(path)
        except (OSError, ValueError):
            self.counts[f"production_{role}_lists_unreadable"] += 1
            return
        if len(vectors) != len(texts):
            self.counts[f"production_{role}_lists_size_mismatch"] += 1
            return
        fresh = [position for position, key in enumerate(keys) if not store.contains(key)]
        lengths = self.token_lengths([texts[position] for position in fresh], role)
        store.add(
            [keys[p] for p in fresh], vectors[fresh], lengths, f"production_cache:{path.name}"
        )
        self.counts[f"production_{role}_lists_imported"] += 1
        self.counts[f"production_{role}_texts_imported"] += len(fresh)

    def import_production(self, hearing: HearingData) -> None:
        if not self.production_ready() or hearing.hearing_id in self.imported_hearings:
            return
        self.imported_hearings.add(hearing.hearing_id)
        sentences = [text for person in hearing.people for text in person["sentences"]]
        self.import_list(sentences, f"sentences_{hearing.hearing_id}", "passage")
        masked = self.runtime.masked_texts_by_hearing.get(hearing.hearing_id, [])
        self.import_list(masked, f"masked_{hearing.hearing_id}", "query")

    def tokenizer(self) -> Any:
        if self.tokenizer_instance is None:
            from transformers import AutoTokenizer

            self.tokenizer_instance = AutoTokenizer.from_pretrained(
                self.spec.model, revision=self.spec.revision, local_files_only=True
            )
        return self.tokenizer_instance

    def token_lengths(self, texts: list[str], role: str) -> list[int]:
        if not texts:
            return []
        prefix = self.spec.query_prefix if role == "query" else self.spec.passage_prefix
        encoded = self.tokenizer()([prefix + text for text in texts], add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: list[str], role: str) -> tuple[np.ndarray, list[int]]:
        model = self.load()
        prefix = self.spec.query_prefix if role == "query" else self.spec.passage_prefix
        started = time.perf_counter()
        lengths = self.token_lengths(texts, role)
        capped = [min(length, self.spec.max_seq_length) for length in lengths]
        batches = token_batches(capped, self.spec.batch_size, self.spec.token_budget)
        dimension = model.get_embedding_dimension()
        vectors = np.zeros((len(texts), dimension), dtype=np.float32)
        for batch in batches:
            vectors[batch] = model.encode(
                [prefix + texts[index] for index in batch],
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        self.encode_seconds += time.perf_counter() - started
        self.counts[f"{role}_texts_encoded"] += len(texts)
        self.counts[f"{role}_tokens_encoded"] += sum(capped)
        return vectors, lengths

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        store = self.stores[role]
        keys = [text_digest(text) for text in texts]
        self.counts[f"{role}_texts_requested"] += len(texts)
        missing = list(
            dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if not store.contains(k))
        )
        if missing:
            vectors, lengths = self.encode(missing, role)
            store.add([text_digest(text) for text in missing], vectors, lengths, "encoded")
        return normalize_rows(store.gather(keys).astype(np.float32))

    def truncation(self, texts: list[str], role: str) -> Record:
        store = self.stores[role]
        lengths = np.array([store.length(text_digest(text)) for text in texts])
        return {
            "texts": int(len(lengths)),
            "over_max_seq_length": int((lengths > self.spec.max_seq_length).sum()),
            "max_tokens": int(lengths.max()) if len(lengths) else None,
            "mean_tokens": round(float(lengths.mean()), 1) if len(lengths) else None,
        }

    def flush(self) -> None:
        for store in self.stores.values():
            store.flush()

    def close(self) -> None:
        self.flush()
        if self.model is not None:
            self.model = None
            gc.collect()
            release_device_memory(self.runtime.device)


def release_device_memory(device: str) -> None:
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


class DenseRetriever:
    def __init__(self, spec: DenseSpec, runtime: Runtime) -> None:
        self.retriever_id = spec.name
        self.spec = spec
        self.encoder = DenseEncoder(spec, runtime)
        self.truncation_texts: dict[str, dict[str, set[str]]] = {}

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        self.encoder.import_production(hearing)
        contexts = contexts_of(hearing, queries)
        passages = list(dict.fromkeys(u.text for c in contexts for u in c.units[kind]))
        passage_vectors = self.encoder.embed(passages, "passage")
        row_of = {text: row for row, text in enumerate(passages)}
        query_vectors = self.encoder.embed([query.text for query in queries], "query")
        seen = self.truncation_texts.setdefault(kind, {"query": set(), "passage": set()})
        seen["passage"].update(passages)
        seen["query"].update(query.text for query in queries)
        rankings: dict[str, Ranking] = {}
        for position, query in enumerate(queries):
            context = hearing.contexts[query.context_key]
            rows = [row_of[unit.text] for unit in context.units[kind]]
            scores = passage_vectors[rows] @ query_vectors[position]
            rankings[query.query_id] = score_ranking(scores)
        return rankings

    def describe(self) -> Record:
        return {
            "kind": "dense",
            "model": self.spec.model,
            "revision": self.spec.revision,
            "query_prefix": self.spec.query_prefix,
            "passage_prefix": self.spec.passage_prefix,
            "max_seq_length": self.spec.max_seq_length,
            "native_max_seq_length": self.encoder.native_max_seq_length,
            "weights_file": pinned_weights_file(self.spec.model, self.spec.revision),
            "local_files_only": True,
            "similarity": "cosine (dot product of L2-normalized vectors)",
            "batch_size": self.spec.batch_size,
            "token_budget": self.spec.token_budget,
            "production_cache_reuse": self.encoder.production_ready(),
            "counts": dict(self.encoder.counts),
            "encode_seconds": round(self.encoder.encode_seconds, 1),
            "model_load_seconds": round(self.encoder.load_seconds, 1),
            "truncation": {
                kind: {
                    role: self.encoder.truncation(sorted(texts), role)
                    for role, texts in roles.items()
                }
                for kind, roles in self.truncation_texts.items()
            },
            "card_notes": self.spec.source.get("card_notes"),
        }

    def close(self) -> None:
        self.encoder.close()


def rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(1, len(order) + 1)
    return positions


class RrfRetriever:
    def __init__(self, retriever_id: str, components: list[Retriever], k: int) -> None:
        self.retriever_id = retriever_id
        self.components = components
        self.k = k

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        parts = [component.rank_hearing(hearing, kind, queries) for component in self.components]
        rankings: dict[str, Ranking] = {}
        for query in queries:
            fused = sum(
                1.0 / (self.k + rank_positions(part[query.query_id].order)) for part in parts
            )
            rankings[query.query_id] = score_ranking(np.asarray(fused, dtype=np.float64))
        return rankings

    def describe(self) -> Record:
        return {
            "kind": "rrf",
            "k": self.k,
            "components": [component.retriever_id for component in self.components],
            "component_details": [component.describe() for component in self.components],
            "rank_definition": "1-based position in each component's order (score desc, index asc)",
        }

    def close(self) -> None:
        for component in self.components:
            component.close()


def rerank_top(
    base: dict[str, Ranking],
    hearing: HearingData,
    kind: str,
    queries: list[Query],
    top_k: int,
    pair_scores: Callable[[list[tuple[str, str]]], np.ndarray],
) -> dict[str, Ranking]:
    selected: dict[str, np.ndarray] = {}
    pairs: list[tuple[str, str]] = []
    for query in queries:
        units: list[Unit] = hearing.contexts[query.context_key].units[kind]
        top = base[query.query_id].order[:top_k]
        selected[query.query_id] = top
        pairs.extend((query.text, units[index].text) for index in top)
    scores = pair_scores(pairs)
    rankings: dict[str, Ranking] = {}
    cursor = 0
    for query in queries:
        order = base[query.query_id].order
        top = selected[query.query_id]
        top_scores = scores[cursor : cursor + len(top)]
        cursor += len(top)
        reranked = top[np.lexsort((np.arange(len(top)), -top_scores))]
        final = np.concatenate([reranked, order[len(top) :]])
        sort_key = -rank_positions(order).astype(np.float64)
        display = np.full(len(order), np.nan)
        sort_key[top] = RERANK_OFFSET + top_scores
        display[top] = top_scores
        rankings[query.query_id] = Ranking(final, sort_key, display)
    return rankings


class RerankRetriever:
    def __init__(self, spec: RerankSpec, base: Retriever, runtime: Runtime) -> None:
        self.retriever_id = spec.name
        self.spec = spec
        self.base = base
        self.runtime = runtime
        self.model: Any = None
        self.counts: Counter[str] = Counter()
        self.score_seconds = 0.0
        self.store = VectorStore(
            runtime.rerank_dir / slug(spec.model) / spec.revision[:12] / f"maxlen{spec.max_length}",
            {
                "key": f"{spec.model}@{spec.revision}|max_length={spec.max_length}|identity",
                "model": spec.model,
                "revision": spec.revision,
                "max_length": spec.max_length,
                "stored_scores": "raw logit of the single output, activation Identity",
            },
            runtime.shard_size,
        )
        self.store.provenance = {
            "device": runtime.device,
            "scored_rows_batch_size": spec.batch_size,
            "precision": "float32",
        }

    def load(self) -> Any:
        if self.model is None:
            import torch
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.spec.model,
                revision=self.spec.revision,
                device=self.runtime.device,
                max_length=self.spec.max_length,
                activation_fn=torch.nn.Identity(),
                local_files_only=True,
            )
            self.counts["model_loads"] += 1
        return self.model

    def pair_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        keys = [pair_digest(query, passage) for query, passage in pairs]
        missing = list(
            dict.fromkeys(
                pair for pair, key in zip(pairs, keys, strict=True) if not self.store.contains(key)
            )
        )
        self.counts["pairs_requested"] += len(pairs)
        if missing:
            started = time.perf_counter()
            scores = self.load().predict(
                missing,
                batch_size=self.spec.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            self.score_seconds += time.perf_counter() - started
            self.counts["pairs_scored"] += len(missing)
            self.store.add(
                [pair_digest(q, p) for q, p in missing],
                np.asarray(scores, dtype=np.float32).reshape(-1, 1),
                [-1] * len(missing),
                "scored",
            )
        return self.store.gather(keys).reshape(-1).astype(np.float64)

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        base = self.base.rank_hearing(hearing, kind, queries)
        return rerank_top(base, hearing, kind, queries, self.spec.top_k, self.pair_scores)

    def describe(self) -> Record:
        return {
            "kind": "rerank",
            "model": self.spec.model,
            "revision": self.spec.revision,
            "base": self.base.retriever_id,
            "base_details": self.base.describe(),
            "weights_file": pinned_weights_file(self.spec.model, self.spec.revision),
            "local_files_only": True,
            "top_k": self.spec.top_k,
            "max_length": self.spec.max_length,
            "batch_size": self.spec.batch_size,
            "activation": "Identity (raw logit; the order equals the sigmoid order)",
            "ordering": (
                "the top_k units of the base order sorted by cross-encoder score (base order on "
                "ties), followed by the remaining units in base order"
            ),
            "counts": dict(self.counts),
            "score_seconds": round(self.score_seconds, 1),
            "card_notes": self.spec.source.get("card_notes"),
        }

    def close(self) -> None:
        self.store.flush()
        self.base.close()
        if self.model is not None:
            self.model = None
            gc.collect()
            release_device_memory(self.runtime.device)


class DecisionRerankRetriever:
    def __init__(
        self,
        spec: DecisionRerankSpec,
        base: Retriever,
        runtime: Runtime,
        load_model: Callable[[LayaSpec], DecisionModel] = LayaDecisionModel,
    ) -> None:
        self.retriever_id = spec.name
        self.spec = spec
        self.base = base
        self.runtime = runtime
        self.load_model = load_model
        self.model: DecisionModel | None = None
        self.model_details: Record | None = None
        self.counts: Counter[str] = Counter()
        self.score_seconds = 0.0
        laya, question = spec.laya, spec.question.question
        payload_digest = sha256_text(question.payload_json())[:8]
        self.store = VectorStore(
            runtime.rerank_dir
            / slug(laya.repo_id)
            / laya.revision[:12]
            / slug(laya.label)
            / f"{question.key}-{payload_digest}"
            / laya.device,
            {
                "key": (
                    f"{laya.repo_id}@{laya.revision}|subfolder={laya.label}|"
                    f"max_len={laya.max_length}|head_max_len={laya.head_max_length}|"
                    f"truncate={laya.truncate_key}|device={laya.device}|"
                    f"question={question.payload_json()}|state=premise:unit,hypothesis:query|"
                    "signal=support"
                ),
                "model": laya.repo_id,
                "revision": laya.revision,
                "subfolder": laya.subfolder,
                "question": question.key,
                "question_payload": question.payload(),
                "device": laya.device,
                "stored_scores": (
                    "support signal of the battery question (P(true) for a noul, P(support "
                    "option) for a choice, expected level / (levels - 1) for a score), float32"
                ),
                "stored_lengths": "input tokens of the Laya forward pass",
            },
            runtime.shard_size,
        )
        self.store.provenance = {
            "device": laya.device,
            "scored_rows_batch_size": laya.batch_size,
            "precision": "float32",
        }

    def load(self) -> DecisionModel:
        if self.model is None:
            self.model = self.load_model(self.spec.laya)
            self.counts["model_loads"] += 1
        return self.model

    def pair_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        keys = [pair_digest(query, passage) for query, passage in pairs]
        missing = list(
            dict.fromkeys(
                pair for pair, key in zip(pairs, keys, strict=True) if not self.store.contains(key)
            )
        )
        self.counts["pairs_requested"] += len(pairs)
        if missing:
            battery_question = self.spec.question
            states = [{"premise": passage, "hypothesis": query} for query, passage in missing]
            started = time.perf_counter()
            answers = self.load().predict_batch(states, [battery_question.question])
            self.score_seconds += time.perf_counter() - started
            key = battery_question.question.key
            scores = np.array(
                [battery_question.support(answer[key]) for answer in answers], dtype=np.float32
            )
            lengths = [
                -1 if answer[key].input_tokens is None else int(cast(int, answer[key].input_tokens))
                for answer in answers
            ]
            self.counts["pairs_scored"] += len(missing)
            self.counts["pairs_truncated"] += sum(answer[key].truncated for answer in answers)
            self.store.add(
                [pair_digest(q, p) for q, p in missing], scores.reshape(-1, 1), lengths, "scored"
            )
        return self.store.gather(keys).reshape(-1).astype(np.float64)

    def rank_hearing(
        self, hearing: HearingData, kind: str, queries: list[Query]
    ) -> dict[str, Ranking]:
        base = self.base.rank_hearing(hearing, kind, queries)
        return rerank_top(base, hearing, kind, queries, self.spec.top_k, self.pair_scores)

    def describe(self) -> Record:
        laya, question = self.spec.laya, self.spec.question.question
        return {
            "kind": "decision_rerank",
            "scorer": self.spec.scorer,
            "model": laya.repo_id,
            "revision": laya.revision,
            "subfolder": laya.subfolder,
            "max_len": laya.max_length,
            "head_max_len": laya.head_max_length,
            "truncated_key": laya.truncate_key,
            "answer_cache_dir": str(laya.cache_dir),
            "question": question.key,
            "question_payload": question.payload(),
            "state": "{premise: candidate unit text, hypothesis: query text}",
            "signal": (
                "support signal of the question: P(true) for a noul, P(support option) for a "
                "choice, expected level / (levels - 1) for a score"
            ),
            "base": self.base.retriever_id,
            "base_details": self.base.describe(),
            "top_k": self.spec.top_k,
            "batch_size": laya.batch_size,
            "ordering": (
                "the top_k units of the base order sorted by the support signal (base order on "
                "ties), followed by the remaining units in base order"
            ),
            "counts": dict(self.counts),
            "score_seconds": round(self.score_seconds, 1),
            "decision_model": self.model_details,
            "notes": self.spec.source.get("notes"),
        }

    def close(self) -> None:
        self.store.flush()
        self.base.close()
        if self.model is not None:
            self.model_details = self.model.describe()
            self.model = None
            gc.collect()
            release_device_memory(self.runtime.device)
