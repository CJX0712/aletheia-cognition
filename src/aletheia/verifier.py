"""答案验证器 (M5).

无需金标准(gold-free)的自洽质量评估. 综合三项可量化分量:
- citation_valid: 答案引用的 [n] 编号是否都合法
- coverage: 答案与上下文的 char-bigram Jaccard 覆盖
- relevance: 若注入 cross-encoder, 用 reranker(query, answer) 打分代理语义相关
"""
from __future__ import annotations

import re
from typing import List, Optional

from .protocol import Candidate, Reranker, Score, Verifier

_CIT_RE = re.compile(r"\[(\d+)\]")


def _bigrams(text: str) -> set:
    text = "".join(ch for ch in text if ch.strip())
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class RuleVerifier:
    """规则 + 可选 cross-encoder 验证器."""

    def __init__(self, reranker: Optional[Reranker] = None):
        self._reranker = reranker

    def score(self, query: str, answer: str,
              contexts: List[Candidate]) -> Score:
        cited = {int(m) for m in _CIT_RE.findall(answer)}
        max_idx = len(contexts)
        citation_valid = 1.0 if (not cited or max(cited) <= max_idx) else 0.0
        coverage = _jaccard(answer, " ".join(c.text for c in contexts))
        if self._reranker is not None and answer.strip():
            sc = self._reranker.rerank(query,
                                      [Candidate("", "a", answer, 0.0)], 1)
            relevance = sc[0].score if sc else coverage
        else:
            relevance = coverage
        value = (0.3 * citation_valid + 0.4 * coverage + 0.3 * relevance)
        return Score(
            value=float(value),
            components={
                "citation_valid": citation_valid,
                "coverage": round(float(coverage), 4),
                "relevance": round(float(relevance), 4),
            },
            method="rule" + ("+rerank" if self._reranker else ""),
        )
