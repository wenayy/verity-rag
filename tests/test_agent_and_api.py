import json
import re

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from verity.agent.agent import KnowledgeAgent
from verity.api.server import create_app
from verity.eval.runner import GoldenItem, eval_answers
from verity.llm import FakeLLM, default_fake_handler, structured
from verity.retrieval.retriever import HybridRetriever
from verity.tracing import STORE


def make_agent(index, handler=None, **kw):
    llm = FakeLLM(handler)
    return KnowledgeAgent(HybridRetriever(index, llm, top_k=6), llm, **kw), llm


def test_happy_path_cites_sources_and_traces(index):
    agent, _ = make_agent(index)
    ans = agent.ask("What is the annual contract value for Kaveri Foods?")
    assert ans.citations and ans.citations[0].n == 1
    assert "contracts/kaveri-foods" in ans.retrieved_doc_ids
    assert ans.grounded is True and ans.iterations == 1
    t = STORE.get(ans.trace_id)
    names = {c.name for c in t.root.children}
    assert {"agent.route", "retrieval", "agent.grade", "agent.generate", "agent.verify"} <= names
    assert t.usage.llm_calls >= 5


def test_corrective_loop_refines_query_when_evidence_is_insufficient(index):
    grades = []

    def handler(messages, json_mode):
        if "TASK:GRADE" in messages[0]["content"]:
            grades.append(1)
            if len(grades) == 1:
                return json.dumps({"relevant_ids": [1], "sufficient": False, "missing": "office of Priya Nair"})
            return json.dumps({"relevant_ids": [1, 2], "sufficient": True, "missing": ""})
        return default_fake_handler(messages, json_mode)

    agent, _ = make_agent(index, handler, max_iterations=3)
    events = list(agent.stream("Who manages Kaveri Foods and where are they based?"))
    steps = [e.data["step"] for e in events if e.type == "step"]
    assert steps.count("retrieve") == 2 and "refine" in steps
    final = events[-1].data["answer"]
    assert final.iterations == 2
    refine = next(e for e in events if e.type == "step" and e.data["step"] == "refine")
    assert refine.data["query"] == "office of Priya Nair"


def test_ungrounded_answer_is_regenerated(index):
    verdicts = iter([{"grounded": False, "unsupported_claims": ["costs a million"]},
                     {"grounded": True, "unsupported_claims": []}])
    strict_seen = []

    def handler(messages, json_mode):
        sys = messages[0]["content"]
        if "TASK:VERIFY" in sys:
            return json.dumps(next(verdicts))
        if "Remove or correct them" in sys:
            strict_seen.append(sys)
            return "Corrected answer [1]"
        return default_fake_handler(messages, json_mode)

    agent, _ = make_agent(index, handler)
    events = list(agent.stream("What does RouteIQ Growth cost?"))
    assert any(e.type == "revision" for e in events)
    final = events[-1].data["answer"]
    assert final.answer == "Corrected answer [1]" and final.grounded is True
    assert "costs a million" in strict_seen[0]


def test_router_skips_retrieval_for_small_talk(index):
    def handler(messages, json_mode):
        if "TASK:ROUTE" in messages[0]["content"]:
            return json.dumps({"needs_retrieval": False, "reason": "greeting"})
        return "Hi! Ask me about Helix Freight."

    agent, _ = make_agent(index, handler)
    events = list(agent.stream("hello there"))
    assert not any(e.type == "step" and e.data["step"] == "retrieve" for e in events)
    assert events[-1].data["answer"].answer.startswith("Hi!")


def test_follow_up_is_contextualized(index):
    def handler(messages, json_mode):
        if "TASK:CONTEXTUALIZE" in messages[0]["content"]:
            assert "Priya Nair" in messages[-1]["content"]
            return json.dumps({"standalone_question": "Which office is Priya Nair in?"})
        return default_fake_handler(messages, json_mode)

    agent, _ = make_agent(index, handler)
    hist = [{"role": "user", "content": "Who manages Kaveri Foods?"}, {"role": "assistant", "content": "Priya Nair [1]"}]
    ans = agent.ask("Which office is she in?", hist)
    assert ans.standalone_question == "Which office is Priya Nair in?"
    assert "employees/priya-nair" in ans.retrieved_doc_ids


def test_structured_output_retries_after_invalid_json():
    class Out(BaseModel):
        n: int

    replies = iter(["not json at all", '```json\n{"n": 3}\n```'])
    llm = FakeLLM(lambda m, j: next(replies))
    assert structured(llm, [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], Out).n == 3
    assert "not valid" in llm.calls[1][-1]["content"]


def test_eval_answers_smoke(index):
    agent, llm = make_agent(index)
    golden = [GoldenItem("What is the annual contract value for Kaveri Foods?", "USD 78,000.",
                         ["contracts/kaveri-foods"], ["78,000"], "numerical")]
    res = eval_answers(agent, llm, golden)
    assert res["summary"]["n"] == 1 and res["summary"]["accuracy"] == 4 and res["summary"]["mrr"] > 0


@pytest.fixture()
def client(index):
    agent, _ = make_agent(index)
    with TestClient(create_app(agent=agent)) as c:
        yield c


def test_api_ask_and_traces(client):
    r = client.post("/api/ask", json={"question": "What is the daily meal allowance in Lisbon?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and body["usage"]["llm_calls"] > 0 and "contexts" not in body
    t = client.get(f"/api/traces/{body['trace_id']}")
    assert t.status_code == 200 and t.json()["root"]["name"] == "agent.run"
    assert client.get("/api/traces/nope").status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"


def test_api_stream_emits_steps_tokens_and_final(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "Who founded Helix Freight?"}) as r:
        body = "".join(r.iter_text())
    types = re.findall(r"^event: (\w+)$", body, re.M)
    assert types[0] == "step" and "token" in types and types[-1] == "final"
    final = json.loads(body.strip().split("data: ")[-1])
    assert final["answer"] and final["trace_id"]


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Agent reasoning" in r.text


def test_provider_failure_returns_502(index):
    class ProviderDown(Exception):
        pass

    ProviderDown.__module__ = "openai"

    def handler(messages, json_mode):
        raise ProviderDown("connection refused")

    agent, _ = make_agent(index, handler)
    with TestClient(create_app(agent=agent), raise_server_exceptions=False) as c:
        r = c.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 502 and "VERITY_LLM_BASE_URL" in r.json()["error"]
