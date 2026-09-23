"""推理时计算引擎 (M6) — 系统核心创新.

统一接口调度五种策略, 全部依赖 Generator + Verifier, 因此对 mock
与 GGUF 后端都可用:
- greedy            单 pass 贪心 (temperature 0.2 确定)
- best_of_n         采样 N 条, verifier 择优
- self_consistency  采样 N 条, 答案 span 多数投票
- beam              采样 N 条 -> 选 top width -> 重生成择优
                      (width=1 退化为 greedy 路径, 不变量 3)
- refine_reflect    生成 -> 反思 -> 修正 (三步走)

硬不变量 (tests 机械验证):
1. best_of_n(N≥1) final_score ≥ greedy 单条 score
2. self_consistency 最终答案 == 多数投票 span (确定性)
3. beam(width=1, N=1) 答案 == greedy 答案
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import List, Optional

from .protocol import (Candidate, Generation, Generator, ReasoningStep,
                      ReasoningTrace, Score, Strategy, Verifier)

_SENT_RE = re.compile(r"(?<=[。！？.!?])\s*")


def _build_prompt(query: str, contexts: List[Candidate]) -> str:
    parts = [f"问题：{query}", "", "资料："]
    for i, c in enumerate(contexts, 1):
        parts.append(f"[{i}] {c.text}")
    parts.append("")
    parts.append("请仅依据以上资料回答，引用处用 [n] 标注来源编号。"
                 "若资料不足，明确说明无法回答。")
    return "\n".join(parts)


def _extract_answer(text: str) -> str:
    m = re.search(r"答案[：:]\s*(.+)", text)
    if m:
        return m.group(1).strip()
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return sents[-1] if sents else text.strip()


class ReasoningEngine:
    """推理时计算引擎."""

    def __init__(self, generator: Generator, verifier: Verifier,
                 beam_width: int = 2):
        self._gen = generator
        self._ver = verifier
        self._beam_width = beam_width

    # ---- 策略实现 ---- #
    def _greedy(self, query: str, contexts: List[Candidate], prompt: str,
                seed: int) -> ReasoningTrace:
        gen = self._gen.generate(prompt, temperature=0.2, seed=seed)
        sc = self._ver.score(query, gen.text, contexts)
        step = ReasoningStep(Strategy.GREEDY.value, prompt, gen, sc)
        return ReasoningTrace(query, Strategy.GREEDY, gen.text, [step],
                             sc.value, contexts, 0.0)

    def _sample(self, prompt: str, n: int, seed: int,
                temperature: float = 0.7) -> List[Generation]:
        return [self._gen.generate(prompt, temperature=temperature,
                                   seed=seed + i) for i in range(n)]

    def _best_of_n(self, query: str, contexts: List[Candidate], prompt: str,
                   n: int, seed: int) -> ReasoningTrace:
        gens = self._sample(prompt, n, seed)
        steps: List[ReasoningStep] = []
        best, best_sc = None, None
        for g in gens:
            sc = self._ver.score(query, g.text, contexts)
            steps.append(ReasoningStep(Strategy.BEST_OF_N.value, prompt, g, sc))
            if best_sc is None or sc.value > best_sc.value:
                best, best_sc = g, sc
        return ReasoningTrace(query, Strategy.BEST_OF_N, best.text, steps,
                             best_sc.value, contexts, 0.0)

    def _self_consistency(self, query: str, contexts: List[Candidate],
                          prompt: str, n: int, seed: int) -> ReasoningTrace:
        gens = self._sample(prompt, n, seed)
        spans = [_extract_answer(g.text) for g in gens]
        counts = Counter(spans)
        answer, votes = counts.most_common(1)[0]
        steps: List[ReasoningStep] = []
        for g in gens:
            steps.append(ReasoningStep(Strategy.SELF_CONSISTENCY.value, prompt, g))
        sc = Score(value=votes / n,
                   components={"vote_fraction": votes / n,
                               "distinct": len(counts)},
                   method="self_consistency")
        for s in steps:
            s.score = sc
        return ReasoningTrace(query, Strategy.SELF_CONSISTENCY, answer, steps,
                             votes / n, contexts, 0.0)

    def _beam(self, query: str, contexts: List[Candidate], prompt: str, n: int,
              seed: int) -> ReasoningTrace:
        if self._beam_width <= 1 or n <= 1:
            return self._greedy(query, contexts, prompt, seed)
        gens = self._sample(prompt, n, seed)
        scored = [(self._ver.score(query, g.text, contexts), g) for g in gens]
        scored.sort(key=lambda x: x[0].value, reverse=True)
        width = min(self._beam_width, len(scored))
        survivors = scored[:width]
        steps: List[ReasoningStep] = []
        for sc, g in scored:
            steps.append(ReasoningStep(Strategy.BEAM.value, prompt, g, sc))
        # 第二轮: 在 top-width 上做确认式重生成
        refined = []
        for sc, g in survivors:
            rep = self._gen.generate(
                prompt + "\n初答：" + g.text + "\n请确认或修正以上回答：",
                temperature=0.2, seed=seed + 100)
            rsc = self._ver.score(query, rep.text, contexts)
            refined.append((rsc, rep))
            steps.append(ReasoningStep(Strategy.BEAM.value, prompt, rep, rsc))
        refined.sort(key=lambda x: x[0].value, reverse=True)
        best_sc, best = refined[0]
        return ReasoningTrace(query, Strategy.BEAM, best.text, steps,
                             best_sc.value, contexts, 0.0)

    def _refine_reflect(self, query: str, contexts: List[Candidate],
                        prompt: str, seed: int) -> ReasoningTrace:
        first = self._gen.generate(prompt, temperature=0.7, seed=seed)
        reflect = self._gen.generate(
            prompt + "\n初步回答：" + first.text +
            "\n请审视该回答的事实与逻辑错误，只列问题；若无懈可击写'无误'。",
            temperature=0.3, seed=seed + 1)
        final = self._gen.generate(
            prompt + "\n初步回答：" + first.text + "\n审视：" + reflect.text +
            "\n据此给出最终回答：", temperature=0.2, seed=seed + 2)
        fsc = self._ver.score(query, final.text, contexts)
        steps = [
            ReasoningStep(Strategy.REFINE_REFLECT.value, prompt, first),
            ReasoningStep(Strategy.REFINE_REFLECT.value, prompt, reflect),
            ReasoningStep(Strategy.REFINE_REFLECT.value, prompt, final, fsc),
        ]
        return ReasoningTrace(query, Strategy.REFINE_REFLECT, final.text, steps,
                             fsc.value, contexts, 0.0)

    # ---- 调度 ---- #
    def solve(self, query: str, contexts: List[Candidate],
              strategy: Strategy | str | None = None, n: Optional[int] = None,
              seed: int = 42) -> ReasoningTrace:
        t0 = time.perf_counter()
        if isinstance(strategy, str):
            strategy = Strategy(strategy)
        strategy = strategy or Strategy.BEST_OF_N
        n = n or 4
        prompt = _build_prompt(query, contexts)
        if strategy == Strategy.GREEDY:
            trace = self._greedy(query, contexts, prompt, seed)
        elif strategy == Strategy.BEST_OF_N:
            trace = self._best_of_n(query, contexts, prompt, n, seed)
        elif strategy == Strategy.SELF_CONSISTENCY:
            trace = self._self_consistency(query, contexts, prompt, n, seed)
        elif strategy == Strategy.BEAM:
            trace = self._beam(query, contexts, prompt, n, seed)
        elif strategy == Strategy.REFINE_REFLECT:
            trace = self._refine_reflect(query, contexts, prompt, seed)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        trace.wall_ms = (time.perf_counter() - t0) * 1000.0
        return trace
