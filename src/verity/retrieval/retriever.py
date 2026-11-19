"""Hybrid retrieval pipeline.

query ──► [LLM multi-query expansion] ──► for each query: BM25 top-N  ┐
                                                         dense top-N  ┴─► Reciprocal Rank Fusion
      ──► [LLM listwise rerank] ──► [MMR diversity] ──► top-k chunks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .. import prompts
from ..index.store import KnowledgeIndex
from ..ingest.chunking import Chunk
from ..llm import LLM, structured
from ..tracing import span

Mode = Literal["bm25", "dense", "hybrid"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    idx: int


@dataclass
class RetrievalResult:
    query: str
    queries: list[str]
    hits: list[Hit] = field(default_factory=list)

    def doc_ids(self) -> list[str]:
        """Unique doc ids in rank order (first appearance)."""
        seen: dict[str, None] = {}
        for h in self.hits:
            seen.setdefault(h.chunk.doc_id, None)
        return list(seen)


class ExpandedQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)


class RankedIds(BaseModel):
    ranked_ids: list[int]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d)). Scale-free, so BM25 and
    cosine scores never have to be calibrated against each other."""
    weights = weights or [1.0] * len(rankings)
    out: dict[int, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for rank, idx in enumerate(ranking, start=1):
            out[idx] = out.get(idx, 0.0) + w / (k + rank)
    return out


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, relevance: list[float], k: int, lam: float = 0.75) -> list[int]:
    """Maximal Marginal Relevance over candidate positions; trades relevance for diversity so
    the context window isn't spent on five near-duplicate chunks."""
    rel = np.asarray(relevance, dtype=np.float32)
    if rel.max() > rel.min():
        rel = (rel - rel.min()) / (rel.max() - rel.min())
    chosen: list[int] = []
    remaining = list(range(len(rel)))
    sim = cand_vecs @ cand_vecs.T
    while remaining and len(chosen) < k:
        if not chosen:
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i, j] for j in chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen


class HybridRetriever:
    def __init__(self, index: KnowledgeIndex, llm: LLM | None = None, *, mode: Mode = "hybrid", top_k: int = 8,
                 candidate_k: int = 30, expand: bool = True, rerank: bool = True, use_mmr: bool = True,
                 max_queries: int = 4):
        self.index, self.llm, self.mode = index, llm, mode
        self.top_k, self.candidate_k, self.max_queries = top_k, candidate_k, max_queries
        self.expand = expand and llm is not None
        self.rerank = rerank and llm is not None
        self.use_mmr = use_mmr

    # ------------------------------------------------------------------ stages
    def expand_query(self, question: str) -> list[str]:
        if not self.expand:
            return [question]
        with span("retrieval.expand"):
            try:
                out = structured(self.llm, [
                    {"role": "system", "content": prompts.EXPAND.format(n=self.max_queries)},
                    {"role": "user", "content": f"QUESTION: {question}"}], ExpandedQueries)
                qs = [question] + [q for q in out.queries if q.strip() and q.strip() != question]
                return qs[: self.max_queries + 1]
            except Exception:
                return [question]

    def _ranked(self, query: str) -> list[list[int]]:
        n = self.candidate_k
        lists = []
        if self.mode in ("bm25", "hybrid"):
            lists.append([i for i, _ in self.index.bm25_search(query, n)])
        if self.mode in ("dense", "hybrid"):
            lists.append([i for i, _ in self.index.dense_search(query, n)])
        return lists

    def fuse(self, queries: list[str]) -> list[tuple[int, float]]:
        with span("retrieval.fuse", queries=len(queries), mode=self.mode):
            rankings: list[list[int]] = []
            weights: list[float] = []
            for qi, q in enumerate(queries):
                for r in self._ranked(q):
                    rankings.append(r)
                    weights.append(1.0 if qi == 0 else 0.8)  # original question counts a bit more
            fused = rrf(rankings, weights=weights)
            return sorted(fused.items(), key=lambda x: -x[1])[: self.candidate_k]

    def llm_rerank(self, question: str, cands: list[tuple[int, float]], keep: int) -> list[tuple[int, float]]:
        if not self.rerank or not cands:
            return cands
        pool = cands[: min(len(cands), 20)]
        with span("retrieval.rerank", candidates=len(pool)):
            passages = "\n\n".join(f"[{n}] {self.index.chunks[i].text[:600]}" for n, (i, _) in enumerate(pool, 1))
            try:
                out = structured(self.llm, [
                    {"role": "system", "content": prompts.RERANK},
                    {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n{passages}"}], RankedIds)
            except Exception:
                return cands
            order = [pool[n - 1] for n in dict.fromkeys(out.ranked_ids) if 1 <= n <= len(pool)]
            if not order:
                return cands
            # reranked items first (score by new rank), then the remainder in fused order
            seen = {i for i, _ in order}
            ranked = [(i, 1.0 / r) for r, (i, _) in enumerate(order, 1)]
            rest = [(i, s * 1e-3) for i, s in cands if i not in seen]
            return (ranked + rest)[: max(keep, len(ranked))]

    # ------------------------------------------------------------------ public
    def retrieve(self, question: str, *, queries: list[str] | None = None, k: int | None = None) -> RetrievalResult:
        k = k or self.top_k
        with span("retrieval", question=question[:200]) as s:
            queries = queries or self.expand_query(question)
            cands = self.fuse(queries)
            cands = self.llm_rerank(question, cands, keep=k * 2)
            if self.use_mmr and len(cands) > k:
                idxs = [i for i, _ in cands]
                qv = self.index.embedder.embed([question])[0]
                order = mmr(qv, self.index.vectors[idxs], [sc for _, sc in cands], k)
                cands = [cands[o] for o in order]
            cands = cands[:k]
            hits = [Hit(self.index.chunks[i], sc, i) for i, sc in cands]
            s.set(queries=queries, hits=[h.chunk.id for h in hits])
            return RetrievalResult(question, queries, hits)
