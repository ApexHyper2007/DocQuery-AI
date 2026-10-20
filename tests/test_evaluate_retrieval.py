from src.evaluate_retrieval import evaluate


class FakeRetriever:
    def search(self, question, *, k, candidate_k):
        if question == "q1":
            return [
                {"source": "pytorch", "title_slug": "tensor-add", "chunk_id": "a"},
                {"source": "pytorch", "title_slug": "tensor-add", "chunk_id": "b"},
                {"source": "cuda", "title_slug": "platform", "chunk_id": "c"},
            ][:k]
        return [{"source": "cuda", "title_slug": "platform", "chunk_id": "c"}]


def test_evaluation_metrics_use_unique_source_pages(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"question":"q1","relevant_sources":["pytorch:tensor-add"]},'
        '{"question":"q2","relevant_sources":["pytorch:tensor-add"]}]',
        encoding="utf-8",
    )

    result = evaluate(FakeRetriever(), cases, k=2)

    assert result["cases"] == 2
    assert result["hit_rate_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["precision_at_k"] == 0.25
    assert result["mrr_at_k"] == 0.5
    assert result["details"][0]["ranked_sources"] == ["pytorch:tensor-add", "cuda:platform"]
