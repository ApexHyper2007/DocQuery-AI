"""Ground retrieved documentation snippets with Ollama and source labels."""

from __future__ import annotations

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
            "sources": [_public_source(item) for item in results[:3]],
        }
    cited_sources = []
    for number, item in enumerate(results, start=1):
        public = _public_source(item)
        public["citation"] = f"S{number}"
        cited_sources.append(public)
    answer = generator(question, results)
    return {"answer": answer, "insufficient_context": False, "sources": cited_sources}


def _public_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "chunk_id", "source", "page_title", "section_title", "heading_path",
            "source_url", "configured_url", "content", "dense_similarity", "bm25_score", "rrf_score",
        )
        if item.get(key) is not None
    }
