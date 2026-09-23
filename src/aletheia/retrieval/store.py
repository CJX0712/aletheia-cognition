"""稠密向量索引 (M2).

生产: FaissStore (HNSW, faiss-cpu).
兜底: MemoryStore (暴力余弦, 零依赖).
"""
from __future__ import annotations

import math
from typing import List, Sequence

from ..protocol import Candidate, VectorStore

__all__ = ["FaissStore", "MemoryStore"]


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MemoryStore:
    """零依赖暴力余弦索引."""

    def __init__(self, dim: int = 512):
        self._dim = dim
        self._ids: List[str] = []
        self._vecs: List[List[float]] = []
        self._payloads: List[Candidate] = []

    @property
    def count(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
            payloads: Sequence[Candidate]) -> None:
        for i, v, p in zip(ids, vectors, payloads):
            if len(v) != self._dim:
                raise ValueError(
                    f"vector dim {len(v)} != store dim {self._dim} for {i}")
            self._ids.append(i)
            self._vecs.append(list(v))
            self._payloads.append(p)

    def search(self, vector: Sequence[float], k: int) -> List[Candidate]:
        scored = [
            (c, _cosine(vector, v)) for c, v in zip(self._payloads, self._vecs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[Candidate] = []
        for c, s in scored[:k]:
            nc = Candidate(c.doc_id, c.chunk_id, c.text, float(s), dict(c.meta))
            out.append(nc)
        return out

    def reset(self) -> None:
        self._ids.clear()
        self._vecs.clear()
        self._payloads.clear()


class FaissStore:
    """HNSW 近似最近邻索引 (faiss-cpu)."""

    def __init__(self, dim: int = 512):
        import faiss  # type: ignore

        self._dim = dim
        self._index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = 64
        self._payloads: List[Candidate] = []

    @property
    def count(self) -> int:
        return self._index.ntotal

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
            payloads: Sequence[Candidate]) -> None:
        import numpy as np

        vecs = np.array([list(v) for v in vectors], dtype="float32")
        if vecs.shape[1] != self._dim:
            raise ValueError(
                f"vector dim {vecs.shape[1]} != store dim {self._dim}")
        # HNSW 用内积, 需先归一化
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._index.add(vecs)
        for p in payloads:
            self._payloads.append(
                Candidate(p.doc_id, p.chunk_id, p.text, 0.0, dict(p.meta)))

    def search(self, vector: Sequence[float], k: int) -> List[Candidate]:
        import numpy as np

        v = np.array([list(vector)], dtype="float32")
        n = math.sqrt(float(np.dot(v[0], v[0]))) or 1.0
        v = v / n
        kk = min(k, self.count)
        if kk == 0:
            return []
        scores, idx = self._index.search(v, kk)
        out: List[Candidate] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self._payloads):
                continue
            p = self._payloads[i]
            out.append(
                Candidate(p.doc_id, p.chunk_id, p.text, float(s), dict(p.meta)))
        return out

    def reset(self) -> None:
        import faiss  # type: ignore

        self._index = faiss.IndexHNSWFlat(
            self._dim, 32, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = 64
        self._payloads.clear()
