"""自适应计算预算路由 (M7).

按查询复杂度分配推理时计算策略与采样数 N. 关键不变量:
给定查询, budget 单调递增 ⇒ N 非降且策略不降级 (见 tests).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .protocol import Strategy

_TRIGGERS = ["为什么", "如何", "怎么", "计算", "证明", "比较", "区别",
             "原因", "分析", "推导", "如果", "是否", "推理", "步骤", "判断对错"]
_HARD = ["计算", "证明", "推导", "判断对错", "步骤"]


@dataclass
class Plan:
    strategy: Strategy
    n: int
    complexity: float
    effective: float


class ComplexityRouter:
    """复杂度 × budget → 推理计划."""

    def __init__(self, base_n: int = 4):
        self._base_n = base_n

    def complexity(self, query: str) -> float:
        q = query.strip()
        score = min(len(q) / 60.0, 0.4)
        for t in _TRIGGERS:
            if t in q:
                score += 0.15
        if re.search(r"\d", q):
            score += 0.1
        return min(score, 1.0)

    def plan(self, query: str, budget: float = 1.0,
             override: Strategy | str | None = None) -> Plan:
        c = self.complexity(query)
        eff = max(0.0, min(1.0, c * budget))
        if override is not None:
            strat = override if isinstance(override, Strategy) else Strategy(override)
            n = max(1, int(round(self._base_n * budget)))
        elif eff < 0.2:
            strat, n = Strategy.GREEDY, 1
        elif eff < 0.5:
            strat, n = Strategy.BEST_OF_N, max(2, int(round(2 * budget)))
        elif any(k in query for k in _HARD):
            strat, n = Strategy.SELF_CONSISTENCY, max(3, int(round(4 * budget)))
        else:
            strat, n = Strategy.BEAM, max(3, int(round(4 * budget)))
        return Plan(strategy=strat, n=n, complexity=c, effective=eff)
