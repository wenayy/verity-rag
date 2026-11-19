"""Okapi BM25 over an inverted index, implemented from scratch (no rank_bm25 dependency).

The tokenizer keeps numbers intact ("78,000" -> "78000", "99.9" -> "99.9") because
exact figures, dates and IDs are precisely where lexical search beats embeddings.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = frozenset("""a an and are as at be by for from has have he her his i in is it its of on or our she that the
their them they this to was were will with what which who whom how when where does do did can i'm me my you your""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN.findall(text.lower()):
        t = t.replace(",", "")
        if t not in STOPWORDS:
            toks.append(t)
            if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
                toks.append(t[:-1])  # cheap plural folding: "contracts" also matches "contract"
    return toks


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> BM25:
        for i, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, c in tf.items():
                self.postings[term].append((i, c))
        n = len(texts)
        self.avgdl = sum(self.doc_len) / max(n, 1)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        return self

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: -x[1])[:k]

    # --- persistence helpers (plain dicts -> JSON)
    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "postings": self.postings, "doc_len": self.doc_len,
                "idf": self.idf, "avgdl": self.avgdl}

    @classmethod
    def from_dict(cls, d: dict) -> BM25:
        m = cls(d["k1"], d["b"])
        m.postings = defaultdict(list, {k: [tuple(x) for x in v] for k, v in d["postings"].items()})
        m.doc_len, m.idf, m.avgdl = d["doc_len"], d["idf"], d["avgdl"]
        return m
