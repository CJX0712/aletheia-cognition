"""运行时装配 (assembly) — 把契约组件注入为可运行系统.

所有实现通过工厂构造, 并优先 onnx/gguf, 失败自动回退到
hash/mock/rrf, 保证离线零依赖也能跑通 (health 报告真实 provider).
"""
from __future__ import annotations

from typing import Optional

from .budget import ComplexityRouter
from .config import embed_path, llm_path, rerank_path, resolve
from .generator import build_llm
from .ingest import chunk_document
from .protocol import Config, Strategy
from .reasoning import ReasoningEngine
from .rerank import build_reranker
from .retrieval import BM25Sparse, FaissStore, HybridRetriever, build_embedder
from .verifier import RuleVerifier


class AletheiaEngine:
    """系统主引擎: 装配各模块并暴露 ingest / ask / health."""

    def __init__(self, cfg: Config, embedder_provider: str = "onnx",
                 reranker_provider: str = "onnx", llm_provider: str = "gguf"):
        cfg = resolve(cfg)
        self.cfg = cfg
        self.embedder, self._ep, _ = build_embedder(
            embedder_provider, embed_path(cfg))
        store = FaissStore(self.embedder.dim)
        sparse = BM25Sparse()
        self.reranker, self._rp, _ = build_reranker(
            reranker_provider, rerank_path(cfg))
        self.retriever = HybridRetriever(
            self.embedder, store, sparse, self.reranker,
            cfg.top_k, cfg.rerank_k)
        self.generator, self._lp, _ = build_llm(
            llm_provider, llm_path(cfg), cfg.llm_ctx, cfg.llm_threads)
        self.verifier = RuleVerifier(self.reranker if self._rp == "onnx" else None)
        self.reasoner = ReasoningEngine(self.generator, self.verifier)
        self.router = ComplexityRouter()

    def ingest(self, text: str, doc_id: Optional[str] = None) -> dict:
        cands = chunk_document(text, doc_id)
        self.retriever.add(cands)
        return {
            "ingested_chunks": len(cands),
            "total_chunks": self.retriever.count,
            "doc_id": cands[0].doc_id if cands else None,
        }

    def ask(self, query: str, strategy=None, n: Optional[int] = None,
            budget: float = 1.0, seed: int = 42):
        if isinstance(strategy, str) and strategy == "auto":
            plan = self.router.plan(query, budget=budget)
            strategy, n = plan.strategy, n or plan.n
        contexts = []
        if self.retriever.count > 0:
            contexts = [s.candidate for s in
                        self.retriever.retrieve(query, self.cfg.top_k)]
        if strategy is None:
            strategy = self.cfg.default_strategy
        return self.reasoner.solve(query, contexts, strategy, n, seed)

    def health(self) -> dict:
        return {
            "status": "ok",
            "components": {
                "embed": self._ep,
                "store": "faiss",
                "rerank": self._rp,
                "llm": self._lp,
            },
            "chunks": self.retriever.count,
        }
