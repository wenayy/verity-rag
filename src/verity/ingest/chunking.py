"""Chunking strategies.

* ``fixed``     – sliding character window snapped to word boundaries (baseline).
* ``recursive`` – split on paragraphs, then sentences, then words until chunks fit
                  (the LangChain ``RecursiveCharacterTextSplitter`` idea, from scratch).
* ``markdown``  – heading-aware: each section becomes its own unit, long sections are
                  split recursively, and every chunk is prefixed with a *contextual
                  header* ("Doc title > Section") so a chunk that only says "starts
                  2022-04-01" still carries *whose* contract it is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loaders import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str  # text that is embedded / indexed (includes contextual header)
    section: str = ""
    metadata: dict = field(default_factory=dict)


def _fixed(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    out, cur, i = [], [], 0
    while i < len(words):
        cur, length, j = [], 0, i
        while j < len(words) and length + len(words[j]) + 1 <= size:
            cur.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if not cur:  # single word longer than size
            cur, j = [words[i]], i + 1
        out.append(" ".join(cur))
        if j >= len(words):
            break
        # step back so consecutive windows share ~overlap characters
        back, k = 0, j
        while k > i + 1 and back < overlap:
            k -= 1
            back += len(words[k]) + 1
        i = k
    return out


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive(text: str, size: int, overlap: int, seps: list[str] | None = None) -> list[str]:
    seps = _SEPARATORS if seps is None else seps
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not seps:
        return _fixed(text, size, overlap)
    sep, rest = seps[0], seps[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) == 1:
        return _recursive(text, size, overlap, rest)
    out: list[str] = []
    buf = ""
    for p in parts:
        cand = f"{buf}{sep}{p}" if buf else p
        if len(cand) <= size:
            buf = cand
            continue
        if buf:
            out.append(buf)
        if len(p) > size:
            out.extend(_recursive(p, size, overlap, rest))
            buf = ""
        else:
            # carry a tail of the previous chunk forward as overlap
            tail = buf[-overlap:] if overlap and buf else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            buf = f"{tail}{sep}{p}" if tail and len(tail) + len(p) + len(sep) <= size else p
    if buf:
        out.append(buf)
    return [o.strip() for o in out if o.strip()]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, body) pairs."""
    sections: list[tuple[str, str]] = []
    h1, h2, body = "", "", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if any(b.strip() for b in body):
                sections.append((" > ".join(x for x in (h1, h2) if x), "\n".join(body)))
            body = []
            if len(m.group(1)) == 1:
                h1, h2 = m.group(2).strip(), ""
            else:
                h2 = m.group(2).strip()
        else:
            body.append(line)
    if any(b.strip() for b in body):
        sections.append((" > ".join(x for x in (h1, h2) if x), "\n".join(body)))
    return sections


def chunk_document(doc: Document, strategy: str = "markdown", size: int = 700, overlap: int = 120) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []  # (section, text)
    if strategy == "fixed":
        pieces = [("", t) for t in _fixed(doc.text, size, overlap)]
    elif strategy == "recursive":
        pieces = [("", t) for t in _recursive(doc.text, size, overlap)]
    elif strategy == "markdown":
        for section, body in _sections(doc.text) or [(doc.title, doc.text)]:
            for t in _recursive(body, size, overlap):
                pieces.append((section, t))
    else:
        raise ValueError(f"unknown chunk strategy {strategy!r}")

    chunks = []
    for i, (section, t) in enumerate(pieces):
        header = section if strategy == "markdown" and section else doc.title
        text = f"{header}\n{t}" if strategy == "markdown" else t
        chunks.append(Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, title=doc.title, text=text, section=section,
                            metadata={**doc.metadata, "position": i}))
    return chunks


def chunk_documents(docs: list[Document], strategy: str = "markdown", size: int = 700, overlap: int = 120) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, strategy, size, overlap)]
