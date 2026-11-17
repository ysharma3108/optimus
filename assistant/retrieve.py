"""Search the knowledge base.

BM25 keyword search (Robertson & Zaragoza 2009) is the first ranking. It
needs no model and is exact on API names such as `create_bowl`, and the BEIR
benchmark (Thakur et al. 2021) found it a strong zero-shot baseline. A second
ranking, for example from a local embedding model, can be merged with
Reciprocal Rank Fusion (Cormack et al. 2009), which needs no tuning.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from . import config
from .ingest import Piece

# A small English stop list. Kept short on purpose: words like "not" or
# "how" can matter in technical questions.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have i in is it its "
    "me my of on or that the this to was we what when which with you your".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> List[str]:
    """Lower-case words. Identifiers are kept whole AND split into parts,
    so "create_bowl" matches both "create_bowl" and "bowl"."""
    out = []
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", text):
        low = ident.lower()
        parts = [p.lower() for p in _WORD.findall(ident)]
        if "_" in ident and len(parts) > 1:
            out.append(low.strip("_"))
        for p in parts:
            if p not in STOPWORDS:
                out.append(p)
    return out


@dataclass
class Hit:
    piece: Piece
    score: float
    rank: int


class BM25:
    """Okapi BM25 over a fixed list of pieces."""

    def __init__(
        self,
        pieces: Sequence[Piece],
        k1: float = config.BM25_K1,
        b: float = config.BM25_B,
        tokenizer: Callable[[str], List[str]] = tokenize,
    ):
        self.pieces = list(pieces)
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer
        self.docs = [Counter(tokenizer(p.text)) for p in self.pieces]
        self.lengths = [sum(d.values()) for d in self.docs]
        n = len(self.docs)
        self.avg_len = (sum(self.lengths) / n) if n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        # The "+1" inside the log keeps idf positive for very common terms
        # (the Lucene form of BM25 idf).
        self.idf: Dict[str, float] = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def scores(self, query: str) -> List[float]:
        terms = self.tokenizer(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else 0.0
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def search(self, query: str, k: int = config.TOP_K) -> List[Hit]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [Hit(self.pieces[i], sc[i], r + 1) for r, i in enumerate(order[:k]) if sc[i] > 0]


def rrf(rankings: Sequence[Sequence[Hit]], k: int = config.RRF_K, top: int = config.TOP_K) -> List[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    fused: Dict[str, float] = {}
    by_id: Dict[str, Piece] = {}
    for ranking in rankings:
        for hit in ranking:
            fused[hit.piece.id] = fused.get(hit.piece.id, 0.0) + 1.0 / (k + hit.rank)
            by_id[hit.piece.id] = hit.piece
    order = sorted(fused, key=lambda pid: (-fused[pid], pid))
    return [Hit(by_id[pid], fused[pid], r + 1) for r, pid in enumerate(order[:top])]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class DenseIndex:
    """Ranking by embedding similarity. `embed` is any function from a list
    of texts to a list of vectors, e.g. llm.OllamaClient.embed."""

    def __init__(self, pieces: Sequence[Piece], embed: Callable[[List[str]], List[List[float]]]):
        self.pieces = list(pieces)
        self.embed = embed
        self.vectors = embed([p.text for p in self.pieces]) if self.pieces else []

    def search(self, query: str, k: int = config.TOP_K) -> List[Hit]:
        q = self.embed([query])[0]
        sc = [cosine(q, v) for v in self.vectors]
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))
        return [Hit(self.pieces[i], sc[i], r + 1) for r, i in enumerate(order[:k])]


class Retriever:
    """BM25 alone, or BM25 fused with a dense ranking when one is given."""

    def __init__(self, pieces: Sequence[Piece], dense: Optional[DenseIndex] = None):
        self.bm25 = BM25(pieces)
        self.dense = dense

    def search(self, query: str, k: int = config.TOP_K) -> List[Hit]:
        # Take a deeper list from each ranker so fusion has something to merge.
        depth = max(k * 3, 20)
        lexical = self.bm25.search(query, depth)
        if self.dense is None:
            return lexical[:k]
        return rrf([lexical, self.dense.search(query, depth)], top=k)

    def best_lexical_score(self, query: str) -> float:
        hits = self.bm25.search(query, 1)
        return hits[0].score if hits else 0.0
