"""Retrieval metrics over document-level rankings.

All functions take ``ranked`` (doc ids in rank order, de-duplicated) and ``relevant``
(the set of doc ids a golden question needs).
"""
from __future__ import annotations

import math


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant document (0 if none retrieved)."""
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents found in the top k. Matters for multi-hop questions."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: rewards putting *all* relevant docs near the top."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def keyword_coverage(texts: list[str], keywords: list[str]) -> float:
    """Fraction of expected answer keywords present anywhere in the retrieved context."""
    if not keywords:
        return 0.0
    blob = " ".join(texts).lower().replace(",", "")
    return sum(1 for kw in keywords if kw.lower().replace(",", "") in blob) / len(keywords)
