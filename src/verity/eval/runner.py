"""Evaluation harness.

* ``retrieval``  – MRR, nDCG@k, Recall@k and keyword coverage for any set of retriever
                   configurations (an ablation table). Needs no LLM when expansion/rerank
                   are off, so it runs in CI.
* ``answers``    – runs the full agent on each golden question and scores the answer with an
                   LLM-as-judge (accuracy / completeness / relevance, 1–5), plus groundedness
                   rate, latency, tokens and cost from the traces.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from .. import prompts
from ..agent.agent import KnowledgeAgent
from ..index.store import KnowledgeIndex
from ..llm import LLM, structured
from ..retrieval.retriever import HybridRetriever
from ..tracing import STORE
from .metrics import keyword_coverage, ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass
class GoldenItem:
    question: str
    reference_answer: str
    sources: list[str]
    keywords: list[str]
    category: str


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem(**json.loads(line)) for line in Path(path).open() if line.strip()]


@dataclass
class RetrievalReport:
    name: str
    k: int
    n: int
    mrr: float
    ndcg: float
    recall: float
    keyword_coverage: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    seconds: float = 0.0

    def row(self) -> str:
        return (f"| {self.name} | {self.mrr:.3f} | {self.ndcg:.3f} | {self.recall:.3f} | "
                f"{self.keyword_coverage:.3f} | {1000 * self.seconds / max(self.n, 1):.1f} |")


def eval_retrieval(name: str, retriever: HybridRetriever, golden: list[GoldenItem], k: int = 5) -> RetrievalReport:
    per_cat: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    rows = []
    t0 = time.perf_counter()
    for g in golden:
        res = retriever.retrieve(g.question, k=max(k, retriever.top_k))
        ranked = res.doc_ids()
        rel = set(g.sources)
        texts = [h.chunk.text for h in res.hits[:k]]
        r = (reciprocal_rank(ranked, rel), ndcg_at_k(ranked, rel, k), recall_at_k(ranked, rel, k),
             keyword_coverage(texts, g.keywords))
        rows.append(r)
        per_cat[g.category].append(r)
    secs = time.perf_counter() - t0

    def mean(xs, i):
        return statistics.fmean(x[i] for x in xs)

    return RetrievalReport(
        name, k, len(golden), mean(rows, 0), mean(rows, 1), mean(rows, 2), mean(rows, 3),
        {c: {"n": len(v), "mrr": mean(v, 0), "ndcg": mean(v, 1), "recall": mean(v, 2), "kw": mean(v, 3)}
         for c, v in sorted(per_cat.items())}, secs)


class JudgeScore(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    feedback: str = ""


def judge(llm: LLM, g: GoldenItem, answer: str) -> JudgeScore:
    return structured(llm, [
        {"role": "system", "content": prompts.JUDGE},
        {"role": "user", "content": f"QUESTION: {g.question}\nREFERENCE ANSWER: {g.reference_answer}\n"
                                    f"GENERATED ANSWER: {answer}"}], JudgeScore)


def eval_answers(agent: KnowledgeAgent, judge_llm: LLM, golden: list[GoldenItem], k: int = 5) -> dict:
    results = []
    for g in golden:
        t0 = time.perf_counter()
        ans = agent.ask(g.question)
        latency = time.perf_counter() - t0
        tr = STORE.get(ans.trace_id) if ans.trace_id else None
        score = judge(judge_llm, g, ans.answer)
        results.append({
            "question": g.question, "category": g.category, "answer": ans.answer,
            "accuracy": score.accuracy, "completeness": score.completeness, "relevance": score.relevance,
            "feedback": score.feedback, "grounded": ans.grounded, "iterations": ans.iterations,
            "mrr": reciprocal_rank(ans.retrieved_doc_ids, set(g.sources)),
            "recall": recall_at_k(ans.retrieved_doc_ids, set(g.sources), 2 * k),
            "latency_s": latency,
            "tokens": (tr.usage.prompt_tokens + tr.usage.completion_tokens) if tr else 0,
            "cost_usd": tr.usage.cost_usd if tr else 0.0,
        })
    keys = ["accuracy", "completeness", "relevance", "mrr", "recall", "latency_s", "tokens", "cost_usd", "iterations"]
    summary = {k_: statistics.fmean(r[k_] for r in results) for k_ in keys}
    summary["grounded_rate"] = statistics.fmean(1.0 if r["grounded"] else 0.0 for r in results)
    summary["n"] = len(results)
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    summary["by_category"] = {c: {k_: statistics.fmean(r[k_] for r in rs) for k_ in ("accuracy", "completeness")}
                              | {"n": len(rs)} for c, rs in sorted(by_cat.items())}
    return {"summary": summary, "results": results}


# ---------------------------------------------------------------- ablation


ABLATIONS: list[dict] = [
    {"name": "BM25 · fixed chunks", "strategy": "fixed", "mode": "bm25"},
    {"name": "Dense (LSA) · fixed chunks", "strategy": "fixed", "mode": "dense"},
    {"name": "Hybrid RRF · fixed chunks", "strategy": "fixed", "mode": "hybrid"},
    {"name": "BM25 · markdown + context headers", "strategy": "markdown", "mode": "bm25"},
    {"name": "Dense (LSA) · markdown + context headers", "strategy": "markdown", "mode": "dense"},
    {"name": "Hybrid RRF · markdown + context headers", "strategy": "markdown", "mode": "hybrid"},
    {"name": "Hybrid RRF · markdown · + MMR", "strategy": "markdown", "mode": "hybrid", "mmr": True},
]


def run_ablation(docs, golden: list[GoldenItem], k: int = 5, llm: LLM | None = None,
                 embedder_factory=None, size: int = 700, overlap: int = 120) -> list[RetrievalReport]:
    from ..index.embeddings import LSAEmbedder

    embedder_factory = embedder_factory or (lambda: LSAEmbedder(256))
    indexes: dict[str, KnowledgeIndex] = {}
    reports = []
    configs = list(ABLATIONS)
    if llm is not None:
        configs += [
            {"name": "Hybrid · markdown · + LLM query expansion", "strategy": "markdown", "mode": "hybrid", "expand": True},
            {"name": "Hybrid · markdown · + expansion + LLM rerank", "strategy": "markdown", "mode": "hybrid",
             "expand": True, "rerank": True},
        ]
    for cfg in configs:
        if cfg["strategy"] not in indexes:
            indexes[cfg["strategy"]] = KnowledgeIndex.build(docs, strategy=cfg["strategy"], size=size, overlap=overlap,
                                                            embedder=embedder_factory())
        r = HybridRetriever(indexes[cfg["strategy"]], llm, mode=cfg["mode"], top_k=8, candidate_k=30,
                            expand=cfg.get("expand", False), rerank=cfg.get("rerank", False),
                            use_mmr=cfg.get("mmr", False))
        reports.append(eval_retrieval(cfg["name"], r, golden, k))
    return reports


def ablation_markdown(reports: list[RetrievalReport], k: int) -> str:
    head = (f"| Configuration | MRR | nDCG@{k} | Recall@{k} | Keyword cov. | ms/query |\n"
            "|---|---|---|---|---|---|")
    return "\n".join([head] + [r.row() for r in reports])


def category_markdown(report: RetrievalReport) -> str:
    head = "| Category | n | MRR | nDCG | Recall | Keyword cov. |\n|---|---|---|---|---|---|"
    rows = [f"| {c} | {v['n']} | {v['mrr']:.3f} | {v['ndcg']:.3f} | {v['recall']:.3f} | {v['kw']:.3f} |"
            for c, v in report.by_category.items()]
    return "\n".join([head] + rows)
