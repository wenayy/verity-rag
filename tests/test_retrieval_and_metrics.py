import json
import math

import numpy as np
import pytest

from verity.eval.metrics import keyword_coverage, ndcg_at_k, recall_at_k, reciprocal_rank
from verity.llm import FakeLLM
from verity.retrieval.retriever import HybridRetriever, mmr, rrf


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == pytest.approx(fused[2])
    assert fused[1] > fused[3] > 0 and 4 in fused
    weighted = rrf([[1], [2]], weights=[1.0, 0.5])
    assert weighted[1] > weighted[2]


def test_mmr_avoids_duplicates():
    v = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    order = mmr(np.array([1, 0], dtype=np.float32), v, [1.0, 0.99, 0.5], k=2, lam=0.5)
    assert order == [0, 2]  # the near-duplicate (1) is skipped in favour of the diverse item


def test_metrics_known_values():
    ranked = ["a", "b", "c", "d"]
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0
    assert recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert ndcg_at_k(ranked, {"a"}, 5) == 1.0
    assert ndcg_at_k(ranked, {"b"}, 5) == pytest.approx(1 / math.log2(3))
    assert keyword_coverage(["USD 78,000 annual"], ["78,000", "Priya"]) == 0.5


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_retriever_modes_find_the_right_doc(index, mode):
    r = HybridRetriever(index, None, mode=mode, top_k=5, expand=False, rerank=False)
    res = r.retrieve("What is the annual contract value for Nordhavn Logistics?")
    assert "contracts/nordhavn-logistics" in res.doc_ids()[:3]


def test_llm_rerank_reorders_candidates(index):
    target = next(i for i, c in enumerate(index.chunks) if c.doc_id == "policies/expense-policy")

    def handler(messages, json_mode):
        user = messages[-1]["content"]
        # pick the passage number whose text is from the expense policy and put it first
        import re
        nums = re.findall(r"\[(\d+)\] (.*)", user)
        first = [int(n) for n, t in nums if "Travel and Expense" in t] or [1]
        return json.dumps({"ranked_ids": first})

    r = HybridRetriever(index, FakeLLM(handler), mode="hybrid", top_k=3, expand=False, rerank=True, use_mmr=False)
    res = r.retrieve("meal allowance Lisbon")
    assert res.hits[0].chunk.doc_id == "policies/expense-policy"
    assert target is not None


def test_query_expansion_is_used(index):
    def handler(messages, json_mode):
        return json.dumps({"queries": ["Kaveri Foods contract", "Priya Nair office"]})

    r = HybridRetriever(index, FakeLLM(handler), mode="hybrid", top_k=8, expand=True, rerank=False)
    res = r.retrieve("Who manages Kaveri and where do they sit?")
    assert res.queries[0] == "Who manages Kaveri and where do they sit?" and len(res.queries) == 3
    assert {"contracts/kaveri-foods", "employees/priya-nair"} <= set(res.doc_ids())
