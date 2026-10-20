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
    for rank, item in enumerate(results, start=1):
        source_key = f"{item.get('source')}:{item.get('title_slug')}"
        if source_key in relevant_sources:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Any, cases_path: Path, *, k: int = 5) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError(f"No evaluation cases found in {cases_path}")
    hits = 0
    reciprocal_ranks = []
    details = []
    for case in cases:
        relevant = set(case["relevant_sources"])
        results = retriever.search(case["question"], k=k, candidate_k=max(20, k))
        found = any(f"{item.get('source')}:{item.get('title_slug')}" in relevant for item in results)
        hits += int(found)
        rr = reciprocal_rank(results, relevant)
        reciprocal_ranks.append(rr)
        details.append({"question": case["question"], "hit_at_k": found, "reciprocal_rank": rr})
    total = len(cases)
    return {
        "cases": total,
        "k": k,
        "hit_rate_at_k": hits / total,
        "mrr_at_k": sum(reciprocal_ranks) / total,
        "details": details,
        "note": "Source-level labels are a coarse starter benchmark; manually inspect chunk relevance before tuning.",
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
