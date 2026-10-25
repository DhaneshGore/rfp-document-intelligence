"""BM25 keyword index (rank_bm25) with an identifier-friendly tokenizer.

Exact identifiers matter in RFPs (bid numbers like JA-207652, SKUs like 210-BLYZ,
models like WD22TB4), so hyphenated/alphanumeric tokens are kept whole *and* split
into parts.
"""
from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from search.filters import matches
from schemas import Chunk, SearchFilters

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.#][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have if in into is it its of on or
    such that the their then there these they this to was were will with what which who
    when where how does do did can could should would shall may i you we our your""".split()
)


# Document labels followed by a number ("Table 5", "Clause 7.78", "Addendum No. 2") also get a
# compound token ("table_5"). The parts alone are weak signals - every caption says "table"
# and small numbers are everywhere - so without it a shorter, unlabelled chunk with the same
# words outranks the labelled one through BM25 length normalisation.
_LABEL_WORDS = frozenset(
    "table exhibit schedule figure appendix annex attachment section clause article item addendum amendment".split()
)
_LABEL_NUM = re.compile(r"^[a-z]?\d+(?:[.\-]\d+)*[a-z]?$")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    raw = _TOKEN_RE.findall(text.lower())
    for i, tok in enumerate(raw):
        if any(sep in tok for sep in "-/.#"):
            tokens.append(tok)  # whole identifier, e.g. "ja-207652"
            tokens.extend(p for p in re.split(r"[-/.#]", tok) if p and p not in STOPWORDS)
        elif tok not in STOPWORDS:
            tokens.append(tok)
        if tok in _LABEL_WORDS:
            j = i + 1 + (i + 1 < len(raw) and raw[i + 1] in ("no", "nr", "number"))
            if j < len(raw) and _LABEL_NUM.match(raw[j]):
                tokens.append(f"{tok}_{raw[j]}")
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._bm25: Optional[BM25Okapi] = None
        corpus = [tokenize(c.text) for c in chunks]
        self._token_sets = [set(toks) for toks in corpus]
        if chunks:
            self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self, query: str, n: int, filters: Optional[SearchFilters] = None
    ) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        q_set = set(q)
        # Keep every chunk sharing a query term. Okapi IDF can be <= 0 for terms found
        # in most chunks of a small corpus, so the score alone is not a match test.
        ranked = sorted(
            (
                (c, float(s))
                for c, s, toks in zip(self.chunks, scores, self._token_sets)
                if toks & q_set and matches(c, filters)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]
