"""The agentic RAG loop: an explicit, inspectable state machine (no framework magic).

            ┌──────────────┐   small talk   ┌──────────┐
 question ─►│ contextualize│─► route ──────►│ chitchat │
            └──────────────┘     │ kb       └──────────┘
                                 ▼
                         ┌──────────────┐
                    ┌───►│   retrieve   │  hybrid + expansion + rerank + MMR
                    │    └──────┬───────┘
          refine    │           ▼
          query     │    ┌──────────────┐  corrective RAG: grade evidence,
          (missing) └────│    grade     │  detect what is still missing
                         └──────┬───────┘
                                ▼ sufficient / budget spent
                         ┌──────────────┐
                         │   generate   │  streamed, every sentence cited [n]
                         └──────┬───────┘
                                ▼
                         ┌──────────────┐  self-check: are all claims supported?
                         │    verify    │  if not, regenerate in strict mode
                         └──────────────┘
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import BaseModel

from .. import prompts
from ..ingest.chunking import Chunk
from ..llm import LLM, structured
from ..retrieval.retriever import HybridRetriever
from ..tracing import span, trace


class Route(BaseModel):
    needs_retrieval: bool
    reason: str = ""


class Standalone(BaseModel):
    standalone_question: str


class Grade(BaseModel):
    relevant_ids: list[int]
    sufficient: bool
    missing: str = ""


class Verdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = []


@dataclass
class Event:
    type: str  # "step" | "token" | "revision" | "final"
    data: dict


@dataclass
class Citation:
    n: int
    doc_id: str
    title: str
    section: str
    snippet: str


@dataclass
class AgentAnswer:
    question: str
    standalone_question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[Chunk] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    iterations: int = 0
    grounded: bool | None = None
    trace_id: str | None = None
    steps: list[dict] = field(default_factory=list)


def _sources_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{n}] ({c.title})\n{c.text}" for n, c in enumerate(chunks, 1))


class KnowledgeAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLM, *, max_iterations: int = 2, verify: bool = True,
                 max_context_chunks: int = 8):
        self.retriever, self.llm = retriever, llm
        self.max_iterations, self.verify, self.max_context = max_iterations, verify, max_context_chunks

    # ------------------------------------------------------------------ nodes
    def _contextualize(self, question: str, history: list[dict]) -> str:
        if not history:
            return question
        with span("agent.contextualize"):
            convo = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-6:])
            try:
                return structured(self.llm, [
                    {"role": "system", "content": prompts.CONTEXTUALIZE},
                    {"role": "user", "content": f"HISTORY:\n{convo}\n\nFOLLOW-UP QUESTION: {question}"}],
                    Standalone).standalone_question
            except Exception:
                return question

    def _route(self, question: str) -> Route:
        with span("agent.route"):
            try:
                return structured(self.llm, [{"role": "system", "content": prompts.ROUTE},
                                             {"role": "user", "content": question}], Route)
            except Exception:
                return Route(needs_retrieval=True, reason="router failed; defaulting to retrieval")

    def _grade(self, question: str, chunks: list[Chunk]) -> Grade:
        with span("agent.grade", n=len(chunks)) as s:
            try:
                g = structured(self.llm, [
                    {"role": "system", "content": prompts.GRADE},
                    {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n{_sources_block(chunks)}"}], Grade)
            except Exception:
                g = Grade(relevant_ids=list(range(1, len(chunks) + 1)), sufficient=True)
            g.relevant_ids = [i for i in dict.fromkeys(g.relevant_ids) if 1 <= i <= len(chunks)]
            s.set(relevant=g.relevant_ids, sufficient=g.sufficient, missing=g.missing)
            return g

    def _verify(self, chunks: list[Chunk], answer: str) -> Verdict:
        with span("agent.verify"):
            try:
                return structured(self.llm, [
                    {"role": "system", "content": prompts.VERIFY},
                    {"role": "user", "content": f"SOURCES:\n{_sources_block(chunks)}\n\nDRAFT ANSWER:\n{answer}"}], Verdict)
            except Exception:
                return Verdict(grounded=True)

    # ------------------------------------------------------------------ main loop
    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[Event]:
        """Run the agent, yielding progress events and answer tokens. Last event is ``final``."""
        history = history or []
        with trace("agent.run") as t:
            t.root.set(question=question)
            steps: list[dict] = []

            def step(name: str, **data) -> Event:
                steps.append({"step": name, **data})
                return Event("step", {"step": name, **data})

            standalone = self._contextualize(question, history)
            if standalone != question:
                yield step("contextualize", standalone_question=standalone)

            route = self._route(standalone)
            yield step("route", needs_retrieval=route.needs_retrieval, reason=route.reason)
            if not route.needs_retrieval:
                text = ""
                with span("agent.chitchat"):
                    for tok in self.llm.stream([{"role": "system", "content": prompts.CHITCHAT}, *history[-6:],
                                                {"role": "user", "content": question}]):
                        text += tok
                        yield Event("token", {"text": tok})
                ans = AgentAnswer(question, standalone, text, trace_id=t.id, steps=steps)
                yield Event("final", {"answer": ans})
                return

            # ---- corrective retrieval loop
            evidence: list[Chunk] = []
            seen: set[str] = set()
            retrieved_docs: list[str] = []
            relevant: list[Chunk] = []
            queries: list[str] | None = None
            iterations = 0
            for iterations in range(1, self.max_iterations + 1):
                res = self.retriever.retrieve(standalone, queries=queries)
                for d in res.doc_ids():
                    if d not in retrieved_docs:
                        retrieved_docs.append(d)
                new = [h.chunk for h in res.hits if h.chunk.id not in seen]
                seen.update(c.id for c in new)
                evidence.extend(new)
                yield step("retrieve", iteration=iterations, queries=res.queries,
                           hits=[{"id": h.chunk.id, "title": h.chunk.title} for h in res.hits])

                pool = (relevant + [c for c in evidence if c not in relevant])[:14]
                grade = self._grade(standalone, pool)
                relevant = [pool[i - 1] for i in grade.relevant_ids]
                yield step("grade", iteration=iterations, relevant=[c.id for c in relevant],
                           sufficient=grade.sufficient, missing=grade.missing)
                if grade.sufficient or not grade.missing.strip():
                    break
                if iterations < self.max_iterations:
                    queries = [grade.missing, f"{standalone} {grade.missing}"]
                    yield step("refine", query=grade.missing)

            context = (relevant or evidence)[: self.max_context]
            # top up with unused evidence so the generator has some breathing room
            for c in evidence:
                if len(context) >= min(self.max_context, max(4, len(relevant))):
                    break
                if c not in context:
                    context.append(c)

            # ---- generate (streamed)
            msgs = [{"role": "system", "content": prompts.ANSWER},
                    {"role": "user", "content": f"SOURCES:\n{_sources_block(context)}\n\nQUESTION: {standalone}"}]
            answer = ""
            with span("agent.generate", context_chunks=len(context)):
                for tok in self.llm.stream(msgs):
                    answer += tok
                    yield Event("token", {"text": tok})

            # ---- verify (self-reflection)
            grounded: bool | None = None
            if self.verify and context:
                v = self._verify(context, answer)
                grounded = v.grounded
                yield step("verify", grounded=v.grounded, unsupported=v.unsupported_claims)
                if not v.grounded and v.unsupported_claims:
                    strict = [{"role": "system", "content": prompts.ANSWER_STRICT.format(
                        claims="; ".join(v.unsupported_claims))}, msgs[1]]
                    with span("agent.regenerate"):
                        answer = self.llm.complete(strict)
                    grounded = self._verify(context, answer).grounded
                    yield Event("revision", {"text": answer, "grounded": grounded})

            used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(context)})
            cites = [Citation(n, context[n - 1].doc_id, context[n - 1].title, context[n - 1].section,
                              context[n - 1].text[:240]) for n in used]
            t.root.set(iterations=iterations, grounded=grounded, citations=len(cites))
            ans = AgentAnswer(question, standalone, answer.strip(), cites, context, retrieved_docs, iterations,
                              grounded, t.id, steps)
            yield Event("final", {"answer": ans})

    def ask(self, question: str, history: list[dict] | None = None) -> AgentAnswer:
        final = None
        for ev in self.stream(question, history):
            if ev.type == "final":
                final = ev.data["answer"]
        assert final is not None
        return final


def build_agent(settings=None, index=None, llm=None) -> KnowledgeAgent:
    """Wire an agent from settings (used by the CLI and API)."""
    from ..config import get_settings
    from ..index.store import KnowledgeIndex
    from ..llm import OpenAICompatLLM

    s = settings or get_settings()
    if index is None:
        index = KnowledgeIndex.load(s.index_dir, s) if (s.index_dir / "meta.json").exists() else KnowledgeIndex.from_settings(s)
    llm = llm or OpenAICompatLLM(s.llm_model, s.llm_base_url, s.llm_api_key, s.llm_timeout_s)
    retriever = HybridRetriever(index, llm, mode=s.retrieval_mode, top_k=s.top_k, candidate_k=s.candidate_k,
                                expand=s.use_query_expansion, rerank=s.use_rerank, use_mmr=s.use_mmr)
    return KnowledgeAgent(retriever, llm, max_iterations=s.max_iterations, verify=s.verify_answers)


