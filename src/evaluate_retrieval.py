"""Evaluate retrieval against manually labeled source-level questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.retrieval import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "tests" / "eval_questions.json"


def reciprocal_rank(results: list[dict[str, Any]], relevant_sources: set[str]) -> float:
    ranked_sources = list(dict.fromkeys(
        f"{item.get('source')}:{item.get('title_slug')}" for item in results
    ))
    for rank, source_key in enumerate(ranked_sources, start=1):
        if source_key in relevant_sources:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Any, cases_path: Path, *, k: int = 5) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"No evaluation cases found in {cases_path}")
    hits = 0
    recalls = []
    precisions = []
    reciprocal_ranks = []
    details = []
    fetch_k = max(20, k * 5)
    for case in cases:
        relevant = set(case["relevant_sources"])
        results = retriever.search(case["question"], k=fetch_k, candidate_k=fetch_k)
        ranked_sources = list(dict.fromkeys(
            f"{item.get('source')}:{item.get('title_slug')}" for item in results
        ))
        ranked_at_k = ranked_sources[:k]
        relevant_hits = len(set(ranked_at_k) & relevant)
        found = relevant_hits > 0
        hits += int(found)
        rr = reciprocal_rank(results, relevant)
        reciprocal_ranks.append(rr)
        recalls.append(relevant_hits / len(relevant) if relevant else 0.0)
        precisions.append(relevant_hits / k)
        details.append({
            "question": case["question"], "hit_at_k": found,
            "recall_at_k": recalls[-1], "precision_at_k": precisions[-1],
            "reciprocal_rank": rr, "ranked_sources": ranked_at_k,
        })
    total = len(cases)
    return {
        "cases": total,
        "k": k,
        "hit_rate_at_k": hits / total,
        "recall_at_k": sum(recalls) / total,
        "precision_at_k": sum(precisions) / total,
        "mrr_at_k": sum(reciprocal_ranks) / total,
        "details": details,
        "note": "Metrics use unique source pages, not relevant chunk IDs; inspect passages and answers before tuning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    print(json.dumps(evaluate(HybridRetriever(), args.cases, k=args.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
