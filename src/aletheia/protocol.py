"""Aletheia 契约层 (M1).

定义全系统共享的数据结构与组件协议(Protocol). 任何模块实现都需满足
对应 Protocol, 由 assembly 在运行时注入, 因此可独立替换与单测.

设计原则:
- 数据类(Immutable-ish dataclass)承载跨模块传递的载荷
- Protocol 仅声明"能力", 不约束具体技术选型
- 每个 Protocol 都有零依赖的 mock 兜底实现(见各模块 *_mock)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "Strategy",
    "Candidate",
    "Scored",
    "Generation",
    "ReasoningStep",
    "ReasoningTrace",
    "Score",
    "Config",
    "AletheiaError",
    "ConfigError",
    "ModelLoadError",
    "RetryableError",
    "Embedder",
    "VectorStore",
    "SparseIndex",
    "Reranker",
    "Generator",
    "Retriever",
    "Verifier",
    "Reasoner",
]


class Strategy(str, enum.Enum):
    """推理时计算策略. 统一接口, 由 reasoning 引擎调度."""

    GREEDY = "greedy"                      # 单 pass 贪心
    BEST_OF_N = "best_of_n"                # 采样 N 条, 验证器择优
    SELF_CONSISTENCY = "self_consistency"  # 多数投票一致
    BEAM = "beam"                          # 步级束搜索
    REFINE_REFLECT = "refine_reflect"      # 生成 -> 反思 -> 修正


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """检索返回的单个候选片段."""

    doc_id: str
    chunk_id: str
    text: str
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scored:
    """经重排后的候选(分数已归一化比较)."""

    candidate: Candidate
    score: float


@dataclass
class Generation:
    """一次生成的产出."""

    text: str
    tokens: int
    finish_reason: str
    latency_ms: float
    model: str


@dataclass
class Score:
    """验证器对某答案的打分."""

    value: float
    components: Dict[str, float]
    method: str


@dataclass
class ReasoningStep:
    """推理轨迹中的一步."""

    strategy: str
    prompt: str
    generation: Generation
    score: Optional[Score] = None


@dataclass
class ReasoningTrace:
    """完整推理轨迹 — 系统核心产出的可观测载体."""

    query: str
    strategy: Strategy | str
    answer: str
    steps: List[ReasoningStep]
    final_score: float
    contexts: List[Candidate]
    wall_ms: float


@dataclass
class Config:
    """全局配置. 装配时由 assembly 统一注入."""

    model_dir: str
    embed_model: str = "bge-small-zh"      # ONNX 文件名(不含扩展)
    rerank_model: str = "bge-reranker"
    llm_model: str = "qwen2.5-1.5b-instruct-q4_k_m"
    device: str = "cpu"
    llm_threads: int = 4                   # CPU 线程锁定(见 ADR-0003)
    llm_ctx: int = 4096
    default_strategy: Strategy = Strategy.BEST_OF_N
    default_n: int = 4
    top_k: int = 8                         # 融合后召回数
    rerank_k: int = 4                       # 重排后进入生成数
    seed: int = 42


# --------------------------------------------------------------------------- #
# 错误体系
# --------------------------------------------------------------------------- #
class AletheiaError(Exception):
    """所有 Aletheia 异常的基类."""


class ConfigError(AletheiaError):
    """配置不合法."""


class ModelLoadError(AletheiaError):
    """模型文件缺失或加载失败."""


class RetryableError(AletheiaError):
    """可重试的瞬时错误."""


# --------------------------------------------------------------------------- #
# 组件协议 (能力声明, 非实现)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> "list[list[float]]":
        """批量编码, 返回与 texts 同序的向量列表."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    def add(self, ids: List[str], vectors: "list[list[float]]",
            payloads: List[Candidate]) -> None:
        ...

    def search(self, vector: "list[float]", k: int) -> List[Candidate]:
        ...


@runtime_checkable
class SparseIndex(Protocol):
    def add(self, ids: List[str], docs: List[str]) -> None:
        ...

    def search(self, query: str, k: int) -> List[Candidate]:
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: List[Candidate],
               k: int) -> List[Scored]:
        ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 512, seed: int = 42) -> Generation:
        ...

    async def stream(self, prompt: str, *, temperature: float = 0.7,
                     max_tokens: int = 512, seed: int = 42) -> AsyncIterator[str]:
        ...


@runtime_checkable
class Retriever(Protocol):
    """混合检索编排入口."""

    def retrieve(self, query: str, k: int) -> List[Scored]:
        ...


@runtime_checkable
class Verifier(Protocol):
    """答案质量验证器(无需金标准)."""

    def score(self, query: str, answer: str,
              contexts: List[Candidate]) -> Score:
        ...


@runtime_checkable
class Reasoner(Protocol):
    """推理时计算引擎入口."""

    def solve(self, query: str, contexts: List[Candidate], *,
              strategy: Strategy | str | None = None,
              n: int | None = None, seed: int | None = None) -> ReasoningTrace:
        ...
