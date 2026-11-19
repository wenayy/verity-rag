"""Embedding providers. All return L2-normalised float32 matrices.

* ``LSAEmbedder``     – TF-IDF (word 1-2 grams + char 3-5 grams) reduced with truncated SVD.
                        Runs fully offline, fits on the corpus, and is a surprisingly strong
                        dense baseline for small domain corpora.
* ``OpenAIEmbedder``  – any OpenAI-compatible ``/embeddings`` endpoint (OpenAI, Ollama
                        ``nomic-embed-text``, Together, vLLM...).
* ``HashingEmbedder`` – stateless, used in unit tests.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from ..tracing import span


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims
        self._pipe = None

    def fit(self, texts: list[str]) -> LSAEmbedder:
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        x = hstack([self._word.fit_transform(texts), self._char.fit_transform(texts)]).tocsr()
        n = max(2, min(self.dims, x.shape[0] - 1, x.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=0).fit(x)
        self._pipe = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import hstack

        if self._pipe is None:
            raise RuntimeError("LSAEmbedder must be fit() before embed()")
        with span("embed.lsa", n=len(texts)):
            x = hstack([self._word.transform(texts), self._char.transform(texts)]).tocsr()
            return _normalise(self._svd.transform(x))

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> LSAEmbedder:
        return pickle.loads(path.read_bytes())


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dims: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._v = HashingVectorizer(n_features=dims, ngram_range=(1, 2), alternate_sign=False, norm=None)

    def fit(self, texts: list[str]) -> HashingEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(self._v.transform(texts).toarray())


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, batch: int = 128):
        from openai import OpenAI

        self.model, self.batch = model, batch
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def fit(self, texts: list[str]) -> OpenAIEmbedder:
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        with span("embed.openai", model=self.model, n=len(texts)):
            for i in range(0, len(texts), self.batch):
                resp = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch])
                out.extend(d.embedding for d in resp.data)
        return _normalise(np.array(out))
