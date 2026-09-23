"""稀疏词法检索 (M2).

BM25 (rank-bm25) + 中文 char/skip-gram 分词. 中文无空格, 用字符
unigram+bigram 近似词项, 英文按正则分词.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..protocol import Candidate, SparseIndex

_CJK_RE = re.compile(r"[一-鿿]")
_ALNUM_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """中文 char/skip-gram + 英文词 分词."""
    text = text.lower()
    toks: List[str] = []
    cjk = _CJK_RE.findall(text)
    toks.extend(cjk)
    toks.extend(a + b for a, b in zip(cjk, cjk[1:]))
    toks.extend(_ALNUM_RE.findall(text))
    return toks


class BM25Sparse:
    """BM25 稀疏索引. doc_id 由 hybrid 层用 chunk_id 还原."""

    def __init__(self):
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._tk: Dict[str, List[str]] = {}
        self._bm25 = None

    @property
    def count(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], docs: Sequence[str]) -> None:
        for i, d in zip(ids, docs):
            self._ids.append(i)
            self._docs.append(d)
            self._tk[i] = tokenize(d)
        from rank_bm25 import BM25Okapi  # type: ignore

        self._bm25 = BM25Okapi([self._tk[i] for i in self._ids])

    def search(self, query: str, k: int) -> List[Candidate]:
        if self._bm25 is None or not self._ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Candidate] = []
        for i in order[:k]:
            if scores[i] <= 0:
                continue
            out.append(
                Candidate(doc_id="", chunk_id=self._ids[i],
                          text=self._docs[i], score=float(scores[i]))
            )
        return out

    def reset(self) -> None:
        self._ids.clear()
        self._docs.clear()
        self._tk.clear()
        self._bm25 = None
