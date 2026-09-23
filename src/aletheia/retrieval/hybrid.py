"""混合检索编排 (M2).

dense(HNSW) + sparse(BM25) 经 RRF 倒数排名融合, 再用 cross-encoder
重排. 维护 chunk_id -> Candidate 映射以还原 doc_id.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..protocol import Candidate, Reranker, Retriever, Scored
from .embed import Embedder
from .sparse import BM25Sparse
from .store import VectorStore

_RRF_K = 60


class HybridRetriever:
    """dense+sparse RRF 融合 + 可选重排."""

    def __init__(self, embedder: Embedder, store: VectorStore,
                 sparse: BM25Sparse, reranker: Optional[Reranker] = None,
                 top_k: int = 8, rerank_k: int = 4):
        self._embedder = embedder
        self._store = store
        self._sparse = sparse
        self._reranker = reranker
        self._top_k = top_k
        self._rerank_k = rerank_k
        self._payloads: Dict[str, Candidate] = {}

    @property
    def count(self) -> int:
        return self._store.count

    def add(self, candidates: List[Candidate]) -> None:
        if not candidates:
            return
        ids = [c.chunk_id for c in candidates]
        texts = [c.text for c in candidates]
        vectors = self._embedder.embed(texts)
        self._store.add(ids, vectors, candidates)
        self._sparse.add(ids, texts)
        for c in candidates:
            self._payloads[c.chunk_id] = c

    def _fuse(self, query: str, k: int) -> List[Candidate]:
        qv = self._embedder.embed([query])[0]
        dense = self._store.search(qv, k)
        sparse = self._sparse.search(query, k)
        rrf: Dict[str, float] = {}
        for lst in (dense, sparse):
            for rank, c in enumerate(lst, start=1):
                rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
        ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        out: List[Candidate] = []
        for cid, _ in ordered:
            base = self._payloads.get(cid)
            if base is None:
                continue
            out.append(
                Candidate(base.doc_id, base.chunk_id, base.text,
                          rrf[cid], dict(base.meta)))
        return out

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Scored]:
        k = k or self._top_k
        fused = self._fuse(query, k)
        if self._reranker is not None and fused:
            reranked = self._reranker.rerank(query, fused, self._rerank_k)
            # 未被重排覆盖的候选保留其 RRF 分
            seen = {s.candidate.chunk_id for s in reranked}
            for c in fused:
                if c.chunk_id not in seen:
                    reranked.append(Scored(candidate=c, score=c.score))
            reranked.sort(key=lambda s: s.score, reverse=True)
            return reranked[:k]
        return [Scored(candidate=c, score=c.score) for c in fused]

    def reset(self) -> None:
        self._store.reset()
        self._sparse.reset()
        self._payloads.clear()
