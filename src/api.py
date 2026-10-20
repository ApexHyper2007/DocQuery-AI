"""FastAPI service for DocQuery hybrid retrieval and grounded answers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.answering import answer_question
from src.ollama_client import OllamaError
from src.retrieval import HybridRetriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)


class QueryResponse(BaseModel):
    answer: str
    insufficient_context: bool
    sources: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        collection_name=os.getenv("DOCQUERY_COLLECTION", "docquery_docs"),
        model_name=os.getenv("DOCQUERY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device=os.getenv("DOCQUERY_DEVICE", "cpu"),
        bm25_k1=float(os.getenv("DOCQUERY_BM25_K1", "1.5")),
        bm25_b=float(os.getenv("DOCQUERY_BM25_B", "0.75")),
    )


def create_app(retriever: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="DocQuery AI", version="0.2.0",
        description="Documentation-grounded PyTorch, torchvision, and CUDA assistant",
    )
    app.state.retriever = retriever

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "docquery-ai"}

    @app.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> dict[str, Any]:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must contain non-whitespace text")
        if request.top_k > request.candidate_k * 2:
            raise HTTPException(status_code=422, detail="top_k cannot exceed 2 * candidate_k")
        active_retriever = app.state.retriever or get_retriever()
        try:
            return answer_question(
                question, active_retriever,
                top_k=request.top_k, candidate_k=request.candidate_k,
                rrf_k=int(os.getenv("DOCQUERY_RRF_K", "60")),
                min_similarity=float(os.getenv("DOCQUERY_MIN_SIMILARITY", "0.20")),
                min_bm25_score=float(os.getenv("DOCQUERY_MIN_BM25_SCORE", "1.5")),
            )
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503, detail=f"Retrieval is not ready: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return app


app = create_app()
