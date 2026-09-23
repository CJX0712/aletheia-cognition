"""HTTP API 层 (M8) — FastAPI.

/healthz  健康检查 (含真实 provider)
/ingest   文档摄入
/ask      推理问答 (返回完整 ReasoningTrace, 含推理轨迹与评分)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .assembly import AletheiaEngine
from .config import default_config
from .protocol import Strategy

_engine: Optional[AletheiaEngine] = None


class AskReq(BaseModel):
    query: str
    strategy: Optional[str] = None
    n: Optional[int] = None
    budget: float = 1.0
    seed: int = 42


class IngestReq(BaseModel):
    text: str
    doc_id: Optional[str] = None


def _trace_to_dict(t):
    return {
        "answer": t.answer,
        "strategy": t.strategy.value if isinstance(t.strategy, Strategy) else str(t.strategy),
        "final_score": t.final_score,
        "wall_ms": t.wall_ms,
        "contexts": [
            {"doc_id": c.doc_id, "chunk_id": c.chunk_id,
             "text": c.text, "score": c.score} for c in t.contexts
        ],
        "steps": [
            {"strategy": s.strategy, "text": s.generation.text,
             "score": (s.score.value if s.score else None)} for s in t.steps
        ],
    }


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _engine
    _engine = AletheiaEngine(default_config())
    yield
    _engine = None


app = FastAPI(title="Aletheia 澄明", version="1.0.0", lifespan=_lifespan)


@app.get("/healthz")
def healthz():
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return _engine.health()


@app.post("/ingest")
def ingest(req: IngestReq):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    return _engine.ingest(req.text, req.doc_id)


@app.post("/ask")
def ask(req: AskReq):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    trace = _engine.ask(req.query, req.strategy, req.n, req.budget, req.seed)
    return _trace_to_dict(trace)
