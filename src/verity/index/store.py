"""The knowledge index: chunks + BM25 + dense vectors, persisted to a directory."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import Settings
from ..ingest.chunking import Chunk, chunk_documents
from ..ingest.loaders import Document, load_documents
from ..tracing import span
from .bm25 import BM25
from .embeddings import Embedder, HashingEmbedder, LSAEmbedder, OpenAIEmbedder


def make_embedder(s: Settings) -> Embedder:
    if s.embed_provider == "openai":
        return OpenAIEmbedder(s.embed_model, s.embed_base_url or s.llm_base_url, s.embed_api_key or s.llm_api_key)
    return LSAEmbedder(s.lsa_dims)


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], bm25: BM25, vectors: np.ndarray, embedder: Embedder):
        self.chunks, self.bm25, self.vectors, self.embedder = chunks, bm25, vectors, embedder
        self.doc_titles = {c.doc_id: c.title for c in chunks}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, docs: list[Document], *, strategy: str = "markdown", size: int = 700, overlap: int = 120,
              embedder: Embedder | None = None) -> KnowledgeIndex:
        with span("index.build", docs=len(docs), strategy=strategy):
            chunks = chunk_documents(docs, strategy, size, overlap)
            texts = [c.text for c in chunks]
            embedder = embedder or LSAEmbedder()
            embedder.fit(texts)
            vectors = embedder.embed(texts)
            return cls(chunks, BM25().fit(texts), vectors, embedder)

    @classmethod
    def from_settings(cls, s: Settings, embedder: Embedder | None = None) -> KnowledgeIndex:
        return cls.build(load_documents(s.kb_dir), strategy=s.chunk_strategy, size=s.chunk_size,
                         overlap=s.chunk_overlap, embedder=embedder or make_embedder(s))

    # ------------------------------------------------------------------ search primitives
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.embed([query])[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, k)

    def chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        np.save(path / "vectors.npy", self.vectors)
        (path / "bm25.json").write_text(json.dumps(self.bm25.to_dict()))
        meta = {"embedder": self.embedder.name, "n_chunks": len(self.chunks)}
        if isinstance(self.embedder, LSAEmbedder):
            self.embedder.save(path / "lsa.pkl")
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path, s: Settings | None = None) -> KnowledgeIndex:
        path = Path(path)
        chunks = [Chunk(**json.loads(line)) for line in (path / "chunks.jsonl").open()]
        meta = json.loads((path / "meta.json").read_text())
        if meta["embedder"] == "lsa":
            emb: Embedder = LSAEmbedder.load(path / "lsa.pkl")
        elif meta["embedder"] == "hashing":
            emb = HashingEmbedder()
        else:
            emb = make_embedder(s or Settings())
        return cls(chunks, BM25.from_dict(json.loads((path / "bm25.json").read_text())),
                   np.load(path / "vectors.npy"), emb)
