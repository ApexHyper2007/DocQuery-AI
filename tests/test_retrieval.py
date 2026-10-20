from src.answering import answer_question
from src.retrieval import BM25Index, reciprocal_rank_fusion, tokenize


def sample_chunks():
    return [
        {"chunk_id": "tensor-add-0", "source": "pytorch", "title_slug": "tensor-add", "content": "torch.Tensor.add adds another tensor and supports alpha."},
        {"chunk_id": "cuda-0", "source": "cuda", "title_slug": "programming-model", "content": "CUDA organizes work into grids, blocks, and threads."},
    ]


def test_tokenizer_keeps_common_api_identifiers():
    assert "torch.tensor.add" in tokenize("Use torch.Tensor.add(input, alpha=1).")


def test_bm25_returns_matching_document_first():
    results = BM25Index(sample_chunks()).search("torch.Tensor.add alpha", k=2)
    assert results[0]["chunk_id"] == "tensor-add-0"
    assert results[0]["bm25_score"] > 0


def test_reciprocal_rank_fusion_rewards_agreement():
    bm25 = [{"chunk_id": "shared", "bm25_score": 3}, {"chunk_id": "keyword", "bm25_score": 2}]
    dense = [{"chunk_id": "shared", "dense_similarity": 0.7}, {"chunk_id": "semantic", "dense_similarity": 0.6}]
    result = reciprocal_rank_fusion([bm25, dense], limit=3)
    assert result[0]["chunk_id"] == "shared"
    assert set(result[0]["retrieval_methods"]) == {"bm25", "dense"}
    assert result[0]["dense_similarity"] == 0.7


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, *args, **kwargs):
        return self.results


def test_answering_refuses_weak_context_without_calling_llm():
    def should_not_run(*args, **kwargs):
        raise AssertionError("LLM must not be called for weak context")

    result = answer_question(
        "unrelated question", FakeRetriever([{"chunk_id": "x", "content": "irrelevant", "dense_similarity": 0.05, "bm25_score": 0.1}]),
        generator=should_not_run,
    )
    assert result["insufficient_context"] is True


def test_answering_attaches_citation_ids_and_preserves_source_url():
    result = answer_question(
        "How does this work?",
        FakeRetriever([{
            "chunk_id": "tensor-add-0", "content": "It adds tensors.", "source_url": "https://docs.pytorch.org/api",
            "dense_similarity": 0.8, "bm25_score": 2.0,
        }]),
        generator=lambda question, sources: "It adds tensors [S1].",
    )
    assert result["sources"][0]["citation"] == "S1"
    assert result["sources"][0]["source_url"].endswith("/api")
