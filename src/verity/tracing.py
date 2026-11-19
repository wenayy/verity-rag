"""Tiny, dependency-free tracing: nested spans with timings, token usage and cost.

A trace is opened per request; any code can call ``span("name")`` and the span nests
under whatever span is currently active (tracked with a contextvar, so it is safe
across threads and asyncio tasks). Traces are kept in memory for the API and can be
appended to a JSONL file for offline analysis.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("verity_span", default=None)
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("verity_trace", default=None)


@dataclass
class Span:
    name: str
    start: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_ms": round(self.duration_ms, 2), "attrs": self.attrs,
                "children": [c.to_dict() for c in self.children]}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0


@dataclass
class Trace:
    name: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    root: Span | None = None
    usage: Usage = field(default_factory=Usage)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "created_at": self.created_at,
                "usage": asdict(self.usage), "root": self.root.to_dict() if self.root else None}


class TraceStore:
    def __init__(self, maxlen: int = 200, path: Path | None = None):
        self._items: deque[Trace] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.path = path

    def add(self, t: Trace) -> None:
        with self._lock:
            self._items.append(t)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(t.to_dict()) + "\n")

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            return next((t for t in self._items if t.id == trace_id), None)

    def recent(self, n: int = 20) -> list[Trace]:
        with self._lock:
            return list(self._items)[-n:][::-1]


STORE = TraceStore()


@contextmanager
def trace(name: str, store: TraceStore | None = STORE) -> Iterator[Trace]:
    t = Trace(name)
    tok_t = _current_trace.set(t)
    with span(name) as root:
        t.root = root
        try:
            yield t
        finally:
            pass
    _current_trace.reset(tok_t)
    if store is not None:
        store.add(t)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    s = Span(name, attrs=dict(attrs))
    parent = _current_span.get()
    if parent is not None:
        parent.children.append(s)
    tok = _current_span.set(s)
    try:
        yield s
    except Exception as e:  # record and re-raise
        s.set(error=f"{type(e).__name__}: {e}")
        raise
    finally:
        s.duration_ms = (time.perf_counter() - s.start) * 1000
        _current_span.reset(tok)


def record_usage(prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    t = _current_trace.get()
    s = _current_span.get()
    if s is not None:
        s.set(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=round(cost_usd, 6))
    if t is not None:
        t.usage.prompt_tokens += prompt_tokens
        t.usage.completion_tokens += completion_tokens
        t.usage.llm_calls += 1
        t.usage.cost_usd += cost_usd


def current_trace() -> Trace | None:
    return _current_trace.get()
