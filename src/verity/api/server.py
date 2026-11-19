"""FastAPI service.

POST /api/ask           -> full JSON answer with citations and agent steps
POST /api/ask/stream    -> Server-Sent Events: step / token / revision / final
GET  /api/traces        -> recent traces (latency, tokens, cost per request)
GET  /api/traces/{id}   -> full span tree for one request
POST /api/reindex       -> rebuild the index from the knowledge base folder
GET  /                  -> single-page chat UI that shows the agent's reasoning live
"""
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.agent import AgentAnswer, KnowledgeAgent, build_agent
from ..config import Settings, get_settings
from ..index.store import KnowledgeIndex
from ..tracing import STORE

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


def _answer_json(a: AgentAnswer) -> dict:
    d = asdict(a)
    d.pop("contexts")
    trace = STORE.get(a.trace_id) if a.trace_id else None
    d["usage"] = asdict(trace.usage) if trace else None
    d["latency_ms"] = round(trace.root.duration_ms, 1) if trace and trace.root else None
    return d


def create_app(settings: Settings | None = None, agent: KnowledgeAgent | None = None) -> FastAPI:
    s = settings or get_settings()
    state: dict = {"agent": agent}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["agent"] is None:
            if not (s.index_dir / "meta.json").exists():
                KnowledgeIndex.from_settings(s).save(s.index_dir)
            llm = None
            if s.demo:
                from ..llm import FakeLLM, extractive_demo_handler

                llm = FakeLLM(extractive_demo_handler, model="offline extractive demo", delay=0.03)
            state["agent"] = build_agent(s, llm=llm)
        STORE.path = s.trace_file
        yield

    app = FastAPI(title="Verity", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def provider_errors(request, exc: Exception):
        from fastapi.responses import JSONResponse

        name = type(exc).__name__
        if name.startswith("API") or "openai" in type(exc).__module__:
            return JSONResponse({"error": f"LLM provider error ({name}): {exc}. Check VERITY_LLM_BASE_URL / VERITY_LLM_API_KEY."},
                                status_code=502)
        return JSONResponse({"error": f"{name}: {exc}"}, status_code=500)

    @app.get("/api/health")
    def health():
        a: KnowledgeAgent = state["agent"]
        return {"status": "ok", "chunks": len(a.retriever.index.chunks), "model": a.llm.model,
                "mode": a.retriever.mode}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        return _answer_json(state["agent"].ask(req.question, req.history))

    @app.post("/api/ask/stream")
    async def ask_stream(req: AskRequest):
        agent: KnowledgeAgent = state["agent"]
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        # Run the whole synchronous agent in ONE worker thread so its tracing context stays
        # intact, and hand events to the event loop through a queue.
        def worker():
            try:
                for ev in agent.stream(req.question, req.history):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception as e:  # surface errors to the client instead of hanging
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        threading.Thread(target=worker, daemon=True).start()

        async def events():
            while True:
                ev = await queue.get()
                if ev is done:
                    break
                if isinstance(ev, Exception):
                    yield f"event: error\ndata: {json.dumps({'error': str(ev)})}\n\n"
                    break
                data = _answer_json(ev.data["answer"]) if ev.type == "final" else ev.data
                yield f"event: {ev.type}\ndata: {json.dumps(data, default=str)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/traces")
    def traces(n: int = 20):
        return [{"id": t.id, "question": (t.root.attrs.get("question") if t.root else None),
                 "latency_ms": round(t.root.duration_ms, 1) if t.root else None, "usage": asdict(t.usage)}
                for t in STORE.recent(n)]

    @app.get("/api/traces/{trace_id}")
    def trace_detail(trace_id: str):
        t = STORE.get(trace_id)
        if not t:
            raise HTTPException(404, "trace not found")
        return t.to_dict()

    @app.post("/api/reindex")
    def reindex():
        idx = KnowledgeIndex.from_settings(s)
        idx.save(s.index_dir)
        state["agent"] = build_agent(s, index=idx, llm=state["agent"].llm)
        return {"chunks": len(idx.chunks), "documents": len(idx.doc_titles)}

    @app.get("/")
    def index_page():
        return FileResponse(STATIC / "index.html")

    return app

