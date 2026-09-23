"""Aletheia 单元测试 + 五条硬不变量验证 + 离线 E2E.

全部基于零依赖 mock 后端 (HashEmbedder / MemoryStore / BM25Sparse /
PassthroughReranker / MockLLM), 无网无模型文件即可全绿. 在线 (GGUF)
E2E 由环境变量 ALETHEIA_ONLINE=1 触发, 否则跳过.
"""
import os
import sys

import pytest

from aletheia.budget import ComplexityRouter
from aletheia.generator import MockLLM
from aletheia.ingest import chunk_document
from aletheia.protocol import Candidate, Strategy
from aletheia.reasoning import _extract_answer, ReasoningEngine
from aletheia.retrieval import (BM25Sparse, HashEmbedder, HybridRetriever,
                                MemoryStore)
from aletheia.rerank import PassthroughReranker
from aletheia.verifier import RuleVerifier


def _mock_stack():
    emb = HashEmbedder()
    store = MemoryStore(emb.dim)
    sparse = BM25Sparse()
    retr = HybridRetriever(emb, store, sparse, PassthroughReranker(),
                           top_k=4, rerank_k=3)
    gen = MockLLM()
    ver = RuleVerifier(None)
    return retr, gen, ver


def _sample_docs(retr):
    docs = [
        "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖并释放氧气的过程。",
        "氧气约占大气体积的21%，是需氧生物呼吸所必需的气体。",
        "水的化学式是H2O，由氢和氧两种元素组成。",
    ]
    for i, d in enumerate(docs):
        retr.add(chunk_document(d, doc_id=f"d{i}"))


# --------------------------------------------------------------------------- #
# 基础功能
# --------------------------------------------------------------------------- #
def test_retrieval_recall():
    retr, _, _ = _mock_stack()
    _sample_docs(retr)
    res = retr.retrieve("光合作用释放什么气体", k=4)
    assert len(res) >= 1
    assert res[0].candidate.text.startswith("光合")


def test_ingest_chunking():
    cands = chunk_document("。".join(["短句%d" % i for i in range(50)]),
                          doc_id="x", max_len=40)
    assert len(cands) >= 2
    assert all(c.doc_id == "x" for c in cands)


def test_verifier_components():
    _, _, ver = _mock_stack()
    ctx = [Candidate("d0", "d0#c0", "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖并释放氧气的过程。", 0.9)]
    s = ver.score("光合作用释放什么", "根据资料 [1] 释放氧气。", ctx)
    assert 0.0 <= s.value <= 1.0
    assert "coverage" in s.components
    # 非法引用应扣分
    bad = ver.score("q", "答案 [9] 错误引用。", ctx)
    assert bad.components["citation_valid"] == 0.0


# --------------------------------------------------------------------------- #
# 不变量 1: best_of_n 单调 (final_score >= greedy)
# --------------------------------------------------------------------------- #
def test_invariant_best_of_n_monotone():
    retr, gen, ver = _mock_stack()
    eng = ReasoningEngine(gen, ver)
    q = "光合作用释放什么气体"
    g = eng.solve(q, [], Strategy.GREEDY, 1)
    b = eng.solve(q, [], Strategy.BEST_OF_N, 4)
    # mock 下所有候选相同, 故 best_of_n 分数 == greedy 分数 (满足 >=)
    assert b.final_score >= g.final_score - 1e-9
    assert len(b.steps) == 4


# --------------------------------------------------------------------------- #
# 不变量 2: self_consistency 最终答案 == 多数投票 span (确定性)
# --------------------------------------------------------------------------- #
def test_invariant_self_consistency_majority():
    retr, gen, ver = _mock_stack()
    eng = ReasoningEngine(gen, ver)
    q = "水的化学式是什么"
    t = eng.solve(q, [], Strategy.SELF_CONSISTENCY, 5)
    span = _extract_answer(gen.generate("").text)
    assert t.answer == span
    assert abs(t.final_score - 1.0) < 1e-9  # 全相同 -> 投票比例 1.0


# --------------------------------------------------------------------------- #
# 不变量 3: beam(width=1, N=1) == greedy
# --------------------------------------------------------------------------- #
def test_invariant_beam_degenerate():
    retr, gen, ver = _mock_stack()
    eng = ReasoningEngine(gen, ver)
    q = "氧气占大气多少"
    g = eng.solve(q, [], Strategy.GREEDY, 1)
    b = eng.solve(q, [], Strategy.BEAM, 1)  # width 默认 2, 但 N=1 -> 退化
    assert b.answer == g.answer


# --------------------------------------------------------------------------- #
# 不变量 4: budget 单调 (给定查询, budget 升 -> N 非降, 策略不降级)
# --------------------------------------------------------------------------- #
def test_invariant_budget_monotone():
    router = ComplexityRouter()
    # 高复杂度查询 (长 + 含推理触发词 + 数字)
    q = ("为什么在催化剂存在下该化学反应的速率会明显加快，"
         "请从活化能角度逐步分析其背后的物理机制，并比较有无催化剂时的区别？")
    n_low, strat_low = router.plan(q, budget=0.5).n, router.plan(q, budget=0.5).strategy
    n_high, strat_high = router.plan(q, budget=1.5).n, router.plan(q, budget=1.5).strategy
    assert n_high >= n_low
    # 不允许策略从高计算降级为 greedy
    assert strat_high != Strategy.GREEDY or strat_low == Strategy.GREEDY


def test_invariant_budget_levels():
    router = ComplexityRouter()
    simple = router.plan("苹果是什么", budget=1.0)
    hard = router.plan("为什么光合作用能释放氧气，请推导其化学过程", budget=1.0)
    assert simple.n <= hard.n  # 复杂查询分配更多计算


# --------------------------------------------------------------------------- #
# 不变量 5: 离线 E2E (mock 后端全绿)
# --------------------------------------------------------------------------- #
def test_e2e_offline():
    retr, gen, ver = _mock_stack()
    eng = ReasoningEngine(gen, ver)
    _sample_docs(retr)
    ctx = [s.candidate for s in retr.retrieve("光合作用释放什么", k=4)]
    assert len(ctx) >= 1
    t = eng.solve("光合作用释放什么气体", ctx, Strategy.BEST_OF_N, 3)
    assert t.answer
    assert t.strategy == Strategy.BEST_OF_N
    # 拒绝空查询在 api 层, 此处验证空上下文仍可产出
    t2 = eng.solve("任意问题", [], Strategy.GREEDY, 1)
    assert t2.answer


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
