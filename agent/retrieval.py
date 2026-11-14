"""Hybrid search: BM25 keywords + dense embeddings, fused with reciprocal rank fusion (RRF).

Keywords catch exact terms ("Section 18", "corpus fund"); embeddings catch meaning
("get my money back" ~ "refund"). If the embedding model can't load, search falls back to
BM25 alone and says so in ``mode``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from .documents import Passage

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60  # standard RRF constant: score = sum(1 / (RRF_K + rank))
CANDIDATES = 20  # how many hits each method contributes before fusion

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or "
    "the their there this to was what when where which who will with would you your me we".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case words without stopwords, cut to 6 letters as a crude stem ("cancels",
    "cancellation" -> "cancel"), so keyword search and the fallback grader match word forms."""
    return [t[:6] for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class HybridRetriever:
    def __init__(self, passages: Sequence[Passage], use_embeddings: bool = True):
        self.passages = list(passages)
        self.bm25 = BM25Okapi([tokenize(f"{p.title} {p.text}") for p in self.passages])
        self.embedder = None
        self.vectors = None
        self.mode = "keyword only"
        if use_embeddings:
            try:
                import numpy as np
                from fastembed import TextEmbedding

                self.embedder = TextEmbedding(EMBED_MODEL)
                docs = [f"{p.title}. {p.text}" for p in self.passages]
                self.vectors = np.array(list(self.embedder.embed(docs)))
                self.mode = "hybrid (keywords + embeddings)"
            except Exception as exc:  # no network or model: keyword search still works
                self.embedder = None
                self.mode = f"keyword only (embeddings unavailable: {type(exc).__name__})"

    def _bm25_ranking(self, query: str, allowed: list[int]) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:CANDIDATES]

    def _dense_ranking(self, query: str, allowed: list[int]) -> list[int]:
        if self.embedder is None:
            return []
        import numpy as np

        q = np.array(list(self.embedder.query_embed(query))[0])
        sims = self.vectors[allowed] @ q
        order = np.argsort(-sims)[:CANDIDATES]
        return [allowed[i] for i in order]

    def search(self, query: str, source: str = "both", k: int = 5) -> list[Passage]:
        allowed = [i for i, p in enumerate(self.passages) if source in ("both", p.source)]
        fused: dict[int, float] = {}
        for ranking in (self._bm25_ranking(query, allowed), self._dense_ranking(query, allowed)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [self.passages[i] for i in best]
