"""Provider-agnostic LLM client.

Works with any OpenAI-compatible Chat Completions endpoint (OpenAI, Groq, OpenRouter,
Together, Ollama, vLLM, LM Studio). Structured outputs are requested as JSON, then
validated with Pydantic and retried once with the validation error fed back, which
works even on providers that lack native JSON-schema enforcement.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .tracing import record_usage, span

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]

# USD per 1M tokens (input, output). Unknown models are costed at 0.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def _cost(model: str, pin: int, pout: int) -> float:
    p_in, p_out = PRICES.get(model, (0.0, 0.0))
    return (pin * p_in + pout * p_out) / 1_000_000


class LLM(Protocol):
    model: str

    def complete(self, messages: list[Message], *, temperature: float = 0.0, json_mode: bool = False) -> str: ...

    def stream(self, messages: list[Message], *, temperature: float = 0.0) -> Iterator[str]: ...


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a reply (handles ```json fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def structured(llm: LLM, messages: list[Message], schema: type[T], *, retries: int = 1) -> T:
    """Ask for JSON matching ``schema``; validate; on failure, show the model its error and retry."""
    schema_hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    msgs = list(messages)
    msgs[0] = {**msgs[0], "content": msgs[0]["content"]
               + f"\n\nRespond ONLY with a JSON object that matches this JSON schema:\n{schema_hint}"}
    last_err: Exception | None = None
    for _ in range(retries + 1):
        raw = llm.complete(msgs, json_mode=True)
        try:
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, ValueError) as e:
            last_err = e
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": f"That was not valid. Error: {e}. Return corrected JSON only."}]
    raise ValueError(f"structured output failed for {schema.__name__}: {last_err}")


@dataclass
class OpenAICompatLLM:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed", timeout=self.timeout_s,
                              max_retries=0)

    def _with_retry(self, fn: Callable):
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:  # rate limits / transient network
                status = getattr(e, "status_code", None)
                if attempt == self.max_retries - 1 or (status is not None and status < 500 and status != 429):
                    raise
                time.sleep(delay)
                delay *= 2

    def complete(self, messages: list[Message], *, temperature: float = 0.0, json_mode: bool = False) -> str:
        with span("llm.complete", model=self.model, json_mode=json_mode):
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = self._with_retry(lambda: self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, **kwargs))
            u = resp.usage
            if u:
                record_usage(u.prompt_tokens, u.completion_tokens, _cost(self.model, u.prompt_tokens, u.completion_tokens))
            return resp.choices[0].message.content or ""

    def stream(self, messages: list[Message], *, temperature: float = 0.0) -> Iterator[str]:
        with span("llm.stream", model=self.model):
            resp = self._with_retry(lambda: self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, stream=True,
                stream_options={"include_usage": True}))
            for chunk in resp:
                if chunk.usage:
                    u = chunk.usage
                    record_usage(u.prompt_tokens, u.completion_tokens,
                                 _cost(self.model, u.prompt_tokens, u.completion_tokens))
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class FakeLLM:
    """Deterministic stand-in for tests and offline demos.

    ``handler(messages, json_mode) -> str`` decides the reply. The default handler
    returns minimal valid JSON for every structured call the agent makes, and
    answers by echoing the first retrieved context line.
    """

    model = "fake"

    def __init__(self, handler: Callable[[list[Message], bool], str] | None = None):
        self.handler = handler or default_fake_handler
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, temperature: float = 0.0, json_mode: bool = False) -> str:
        self.calls.append(messages)
        with span("llm.complete", model=self.model, json_mode=json_mode):
            out = self.handler(messages, json_mode)
            record_usage(sum(len(m["content"]) for m in messages) // 4, len(out) // 4, 0.0)
            return out

    def stream(self, messages: list[Message], *, temperature: float = 0.0) -> Iterator[str]:
        text = self.complete(messages)
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word


def default_fake_handler(messages: list[Message], json_mode: bool) -> str:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "TASK:ROUTE" in system:
        return json.dumps({"needs_retrieval": True, "reason": "factual question"})
    if "TASK:CONTEXTUALIZE" in system:
        q = user.split("FOLLOW-UP QUESTION:")[-1].strip()
        return json.dumps({"standalone_question": q})
    if "TASK:EXPAND" in system:
        q = user.split("QUESTION:")[-1].strip()
        return json.dumps({"queries": [q]})
    if "TASK:RERANK" in system:
        ids = re.findall(r"\[(\d+)\]", user)
        return json.dumps({"ranked_ids": [int(i) for i in ids]})
    if "TASK:GRADE" in system:
        ids = re.findall(r"\[(\d+)\]", user)
        return json.dumps({"relevant_ids": [int(i) for i in ids[:3]], "sufficient": True, "missing": ""})
    if "TASK:VERIFY" in system:
        return json.dumps({"grounded": True, "unsupported_claims": []})
    if "TASK:JUDGE" in system:
        return json.dumps({"accuracy": 4, "completeness": 4, "relevance": 5, "feedback": "ok"})
    # answer generation: cite the first source
    m = re.search(r"\[1\][^\n]*\n(.+)", user)
    return (m.group(1).strip()[:200] if m else "I don't know.") + " [1]"
