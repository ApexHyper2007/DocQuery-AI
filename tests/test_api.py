from fastapi.testclient import TestClient

from src.api import create_app


class FakeRetriever:
    def search(self, *args, **kwargs):
        return [{
            "chunk_id": "doc-1", "source": "pytorch", "title_slug": "tensor-add",
            "content": "Adds tensors.", "source_url": "https://docs.pytorch.org/api",
            "dense_similarity": 0.8, "bm25_score": 2.0, "rrf_score": 0.03,
        }]


def test_health_route():
    client = TestClient(create_app(FakeRetriever()))
    assert client.get("/health").json()["status"] == "ok"


def test_query_route_returns_answer_and_source_citation(monkeypatch):
    monkeypatch.setattr(
        "src.api.answer_question",
        lambda question, retriever, **kwargs: {
            "answer": "It adds tensors [S1].", "insufficient_context": False,
            "sources": [{"citation": "S1", "source_url": "https://docs.pytorch.org/api"}],
        },
    )
    client = TestClient(create_app(FakeRetriever()))
    response = client.post("/api/query", json={"question": "How does tensor add work?"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["citation"] == "S1"


def test_query_route_rejects_whitespace_only_questions():
    client = TestClient(create_app(FakeRetriever()))
    response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_route_rejects_top_k_that_exceeds_candidates():
    client = TestClient(create_app(FakeRetriever()))
    response = client.post("/api/query", json={"question": "tensor add", "top_k": 10, "candidate_k": 2})
    assert response.status_code == 422
