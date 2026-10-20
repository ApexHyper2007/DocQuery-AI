"""Keyword, dense, and reciprocal-rank-fusion retrieval over indexed docs."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index" / "chroma"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "docquery_docs"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[.+#-][A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Keep API identifiers and punctuation-bearing names as searchable tokens."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class BM25Index:
    """Minimal Okapi BM25 index, built in memory from the processed JSONL file."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one document chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 > 0 and b between 0 and 1")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk["content"]) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) or 1.0
        self.term_frequencies = [Counter(doc) for doc in self.tokens]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        self.idf = {
            term: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result: list[float] = []
        for counts, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            normalization = self.k1 * (1 - self.b + self.b * length / self.average_length)
            for term in terms:
                frequency = counts.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (frequency + normalization)
            result.append(score)
        return result

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
        return [
            {**self.chunks[index], "bm25_score": scores[index]}
            for index in ranked[:k]
            if scores[index] > 0
        ]


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]], *, rrf_k: int = 60, limit: int = 8
) -> list[dict[str, Any]]:
    """Fuse ranked lists by stable chunk ID without mixing incomparable raw scores."""
    if rrf_k <= 0 or limit <= 0:
        raise ValueError("rrf_k and limit must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            entry = fused.setdefault(chunk_id, {**item, "rrf_score": 0.0, "retrieval_methods": []})
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            method = "bm25" if "bm25_score" in item else "dense"
            if method not in entry["retrieval_methods"]:
                entry["retrieval_methods"].append(method)
            # Keep dense similarity/distance metadata when the same chunk also
            # occurs in BM25 results, while preserving the base chunk content.
            for key in ("dense_distance", "dense_similarity", "bm25_score"):
                if key in item:
                    entry[key] = item[key]
    return sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))[:limit]


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"No chunks found at {path}; run src/process_docs.py")
    for chunk in chunks:
        if not chunk.get("chunk_id") or not chunk.get("content"):
            raise ValueError("Every processed chunk requires a chunk_id and non-empty content")
    return chunks


class HybridRetriever:
    """Dense Chroma search plus local BM25 with RRF score fusion."""

    def __init__(
        self,
        *,
        chunks_path: Path = DEFAULT_CHUNKS,
        persist_dir: Path = DEFAULT_INDEX,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        dense_encoder: Encoder | None = None,
        collection: Any | None = None,
    ) -> None:
        self.chunks = _read_chunks(chunks_path)
        self.bm25 = BM25Index(self.chunks, k1=bm25_k1, b=bm25_b)
        if collection is None or dense_encoder is None:
            import chromadb
            from sentence_transformers import SentenceTransformer

            client = chromadb.PersistentClient(path=str(persist_dir))
            if collection is None:
                collection = client.get_collection(collection_name)
            cache_dir = PROJECT_ROOT / ".cache" / "sentence-transformers"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dense_encoder = dense_encoder or SentenceTransformer(model_name, device=device, cache_folder=str(cache_dir))
        self.collection = collection
        self.encoder = dense_encoder
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1 or candidate_k < 1:
            raise ValueError("k and candidate_k must be positive")
        bm25_results = self.bm25.search(query, candidate_k)
        collection_count = self.collection.count()
        if collection_count < 1:
            raise ValueError("The Chroma collection is empty; run src/index_docs.py first")
        query_vector = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        result = self.collection.query(
            query_embeddings=query_vector.tolist(), n_results=min(candidate_k, collection_count),
            include=["distances", "metadatas", "documents"],
        )
        dense_results: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            dense_results.append({
                **self.by_id.get(chunk_id, {}),
                **(metadata or {}),
                "chunk_id": chunk_id,
                "content": content,
                "dense_distance": float(distance),
                "dense_similarity": 1.0 - float(distance),
            })
        fused = reciprocal_rank_fusion([bm25_results, dense_results], rrf_k=rrf_k, limit=k)
        return fused
