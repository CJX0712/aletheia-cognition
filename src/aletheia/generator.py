"""文本生成后端 (M4).

生产: LlamaCppLLM (llama-cpp-python, 本地 GGUF, 无需服务进程).
CPU 线程锁 4: 小量化模型瓶颈在内存带宽, 默认线程数(cpu_count-1)
反而更慢 (见 ADR-0003).
兜底: MockLLM, 确定性抽取式合成 — 无模型文件整体可验证.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import AsyncIterator, List, Optional, Tuple

from .protocol import Generation, Generator

_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")
_SYSTEM = (
    "你是严谨的问答助手。只依据给定资料回答，"
    "引用处用 [n] 标注来源编号；资料不足就明说。"
)


class MockLLM:
    """确定性兜底生成器: 抽取上下文首句 + 模板."""

    model = "mock"

    def generate(self, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 512, seed: int = 42) -> Generation:
        t0 = time.perf_counter()
        ctx = _extract_context(prompt)
        if not ctx:
            text = "未检索到相关资料，无法回答。"
        else:
            sents: List[str] = []
            for block in ctx:
                for s in _SENT_SPLIT.split(block):
                    s = s.strip()
                    if s:
                        sents.append(s)
                    if sum(len(x) for x in sents) >= max_tokens * 2:
                        break
                if sum(len(x) for x in sents) >= max_tokens * 2:
                    break
            text = "根据检索到的资料：" + "".join(sents[:6])[: max_tokens * 2]
        return Generation(text=text, tokens=len(text),
                          finish_reason="stop", latency_ms=0.0,
                          model=self.model)

    async def stream(self, prompt: str, *, temperature: float = 0.7,
                     max_tokens: int = 512, seed: int = 42) -> AsyncIterator[str]:
        gen = self.generate(prompt, temperature=temperature,
                            max_tokens=max_tokens, seed=seed)
        for ch in [gen.text[i:i + 12] for i in range(0, len(gen.text), 12)]:
            yield ch
            await asyncio.sleep(0)


class LlamaCppLLM:
    """本地 GGUF 对话模型 via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4):
        if not os.path.exists(model_path):
            raise FileNotFoundError("GGUF model missing: " + model_path)
        from llama_cpp import Llama

        self._llm = Llama(model_path=model_path, n_ctx=n_ctx,
                          n_threads=n_threads, verbose=False)
        self.model = os.path.basename(model_path)

    def _messages(self, prompt: str):
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, *, temperature: float = 0.7,
                 max_tokens: int = 512, seed: int = 42) -> Generation:
        t0 = time.perf_counter()
        resp = self._llm.create_chat_completion(
            messages=self._messages(prompt), max_tokens=max_tokens,
            temperature=temperature, seed=seed)
        ch = resp["choices"][0]
        text = ch["message"]["content"].strip()
        usage = resp.get("usage", {})
        tokens = usage.get("completion_tokens", len(text))
        return Generation(
            text=text, tokens=int(tokens),
            finish_reason=ch.get("finish_reason", "stop"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model=self.model)

    async def stream(self, prompt: str, *, temperature: float = 0.7,
                     max_tokens: int = 512, seed: int = 42) -> AsyncIterator[str]:
        gen = self._llm.create_chat_completion(
            messages=self._messages(prompt), max_tokens=max_tokens,
            temperature=temperature, seed=seed, stream=True)
        for chunk in gen:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta
                await asyncio.sleep(0)


def _extract_context(prompt: str) -> List[str]:
    """从 RAG 提示中抽取 [n] 前缀的上下文块."""
    blocks: List[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        if re.match(r"^\[\d+\]", line):
            blocks.append(line)
    return blocks


def build_llm(provider: str, model_path: str, n_ctx: int,
              n_threads: int) -> Tuple[Generator, str, Optional[str]]:
    """工厂: 返回 (llm, 实际provider, 错误信息)."""
    if provider == "gguf":
        try:
            return LlamaCppLLM(model_path, n_ctx, n_threads), "gguf", None
        except Exception as exc:  # noqa: BLE001
            return MockLLM(), "mock", str(exc)
    return MockLLM(), "mock", None
