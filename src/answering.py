"""Ground retrieved documentation snippets with Ollama and source labels."""

from __future__ import annotations

import re
from typing import Any, Callable

from src.ollama_client import generate_grounded_answer


def answer_question(
    question: str,
    retriever: Any,
    *,
    top_k: int = 6,
    candidate_k: int = 20,
    rrf_k: int = 60,
    min_similarity: float = 0.20,
    min_bm25_score: float = 1.5,
    generator: Callable[..., str] = generate_grounded_answer,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty")
    if top_k < 1 or candidate_k < 1 or top_k > candidate_k * 2:
        raise ValueError("top_k and candidate_k must be positive, with top_k no greater than 2 * candidate_k")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    if min_bm25_score < 0:
        raise ValueError("min_bm25_score must be non-negative")
    results = retriever.search(question, k=top_k, candidate_k=candidate_k, rrf_k=rrf_k)
    if not results:
        return {
            "answer": "I could not find relevant documentation for that question in the indexed sources.",
            "insufficient_context": True,
            "sources": [],
        }
    best_similarity = max((item.get("dense_similarity", -1.0) for item in results), default=-1.0)
    best_bm25 = max((item.get("bm25_score", 0.0) for item in results), default=0.0)
    if best_similarity < min_similarity and best_bm25 < min_bm25_score:
        return {
            "answer": "The indexed documentation does not provide enough relevant context to answer this question reliably.",
            "insufficient_context": True,
            "sources": _labeled_sources(results[:3]),
        }
    cited_sources = _labeled_sources(results)
    answer = generator(question, results)
    if not isinstance(answer, str) or not answer.strip():
        return {
            "answer": "The answer model returned no usable text. Please try again.",
            "insufficient_context": True,
            "sources": cited_sources,
        }
    citations = set(re.findall(r"\[(S\d+)\]", answer))
    allowed_citations = {f"S{number}" for number in range(1, len(results) + 1)}
    if not citations or not citations.issubset(allowed_citations):
        return {
            "answer": "I couldn't validate the answer's source citations. Please retry or inspect the retrieved excerpts below.",
            "insufficient_context": True,
            "sources": cited_sources,
        }
    return {"answer": answer, "insufficient_context": False, "sources": cited_sources}


def _labeled_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for number, item in enumerate(items, start=1):
        public = _public_source(item)
        public["citation"] = f"S{number}"
        sources.append(public)
    return sources


def _public_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "chunk_id", "source", "page_title", "section_title", "heading_path",
            "source_url", "configured_url", "content", "dense_similarity", "bm25_score", "rrf_score",
        )
        if item.get(key) is not None
    }
