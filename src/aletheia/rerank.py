"""交叉编码器重排 (M3).

生产: OnnxReranker (bge-reranker-base, ONNX Runtime + tokenizers;
输入名运行时探测, 因不同导出可能含/不含 token_type_ids).
兜底: PassthroughReranker 保持融合召回顺序.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from .protocol import Candidate, Reranker, Scored

__all__ = ["PassthroughReranker", "OnnxReranker", "build_reranker"]


class PassthroughReranker:
    """保持候选顺序 (离线 / CI 使用)."""

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: int) -> List[Scored]:
        out = [Scored(candidate=c, score=c.score) for c in candidates]
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:k]


class OnnxReranker:
    """cross-encoder 重排 via ONNX Runtime (logit 经 sigmoid)."""

    def __init__(self, model_dir: str, max_length: int = 512, batch_size: int = 8):
        model_path = os.path.join(model_dir, "model.onnx")
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not (os.path.exists(model_path) and os.path.exists(tok_path)):
            raise FileNotFoundError("reranker model files missing in " + model_dir)
        import numpy as np  # noqa: F401
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._batch_size = batch_size

    def _score_batch(self, query: str, texts: Sequence[str]) -> List[float]:
        import numpy as np

        encs = self._tokenizer.encode_batch([(query, t) for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encs], dtype=np.int64)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        logits = self._session.run(None, feeds)[0].reshape(-1)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]

    def rerank(self, query: str, candidates: Sequence[Candidate],
               k: int) -> List[Scored]:
        if not candidates:
            return []
        scores: List[float] = []
        texts = [c.text for c in candidates]
        for i in range(0, len(texts), self._batch_size):
            scores.extend(self._score_batch(query, texts[i:i + self._batch_size]))
        out = [Scored(candidate=c, score=s)
               for c, s in zip(candidates, scores)]
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:k]


def build_reranker(provider: str, model_dir: str) -> Tuple[Reranker, str, Optional[str]]:
    """工厂: 返回 (reranker, 实际provider, 错误信息)."""
    if provider == "onnx":
        try:
            return OnnxReranker(model_dir), "onnx", None
        except Exception as exc:  # noqa: BLE001
            return PassthroughReranker(), "rrf", str(exc)
    return PassthroughReranker(), "rrf", None
