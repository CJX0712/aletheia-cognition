"""Aletheia 端到端评测 (M9).

加载真实模型(GGUF/ONNX) 或零依赖 mock, 对中文客观题集做机械判分:
期望答案字符串出现在生成答案中即计为命中. 输出准确率与逐题明细.

用法:
  python scripts/eval.py            # 在线 (真实模型)
  ALETHEIA_OFFLINE=1 python scripts/eval.py   # 离线 (mock, 仅验证链路健康)
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aletheia.assembly import AletheiaEngine  # noqa: E402
from aletheia.config import default_config  # noqa: E402
from aletheia.generator import MockLLM  # noqa: E402
from aletheia.protocol import Strategy  # noqa: E402
from aletheia.rerank import PassthroughReranker  # noqa: E402
from aletheia.retrieval import (BM25Sparse, HashEmbedder, HybridRetriever,  # noqa: E402
                                MemoryStore)
from aletheia.verifier import RuleVerifier  # noqa: E402


def load_questions():
    with open(os.path.join(ROOT, "data", "eval_questions.json"),
              encoding="utf-8") as f:
        return json.load(f)


def build_engine(offline: bool):
    if offline:
        emb = HashEmbedder()
        retr = HybridRetriever(emb, MemoryStore(emb.dim), BM25Sparse(),
                               PassthroughReranker())
        gen = MockLLM()
        ver = RuleVerifier(None)
        eng = AletheiaEngine.__new__(AletheiaEngine)
        eng.retriever = retr
        eng.reasoner = __import__("aletheia.reasoning",
                                  fromlist=["ReasoningEngine"]).ReasoningEngine(gen, ver)
        eng.router = __import__("aletheia.budget",
                                fromlist=["ComplexRouter"]).ComplexityRouter()
        eng.ask = lambda q, s=None, n=None, b=1.0, sd=42: eng.reasoner.solve(
            q, [c.candidate for c in retr.retrieve(q, 4)] if retr.count else [],
            s or Strategy.BEST_OF_N, n or 3, sd)
        return eng, "offline"
    return AletheiaEngine(default_config()), "online"


def main():
    offline = os.environ.get("ALETHEIA_OFFLINE") == "1"
    eng, mode = build_engine(offline)
    qs = load_questions()
    for i, q in enumerate(qs):
        eng.ingest(q["doc"], doc_id=f"d{i}")
    hits, rows = 0, []
    t0 = time.perf_counter()
    for q in qs:
        trace = eng.ask(q["q"], Strategy.BEST_OF_N, 3)

        def norm(s):
            s = s.replace(" ", "").replace("　", "")
            for k, v in {"零": "0", "一": "1", "二": "2", "两": "2",
                         "三": "3", "四": "4", "五": "5", "六": "6",
                         "七": "7", "八": "8", "九": "9", "十": "10"}.items():
                s = s.replace(k, v)
            return s

        ok = norm(q["expect"]) in norm(trace.answer)
        hits += int(ok)
        rows.append({"q": q["q"], "expect": q["expect"],
                     "hit": ok, "score": round(trace.final_score, 3),
                     "answer": trace.answer[:60]})
    elapsed = time.perf_counter() - t0
    acc = hits / len(qs)
    report = {
        "mode": mode,
        "total": len(qs),
        "hits": hits,
        "accuracy": round(acc, 4),
        "elapsed_s": round(elapsed, 2),
        "details": rows,
    }
    out = os.path.join(ROOT, "data", "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {out}")
    if offline:
        return 0  # 离线仅验证链路健康, 不卡准确率
    return 0 if acc >= 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())
