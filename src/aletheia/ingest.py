"""文档摄入与分块 (M0 支持模块).

支持 txt / md / pdf. 中文按标点切句, 累积至 max_len, 相邻块 overlap
以防跨块语义断裂. 产出 Candidate 列表供 retriever.add.
"""
from __future__ import annotations

import re
import uuid
from typing import List

from .protocol import Candidate

_SENT_RE = re.compile(r"(?<=[。！？.!?；;])\s*")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def chunk_text(text: str, max_len: int = 400, overlap: int = 60) -> List[str]:
    text = normalize(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    chunks: List[str] = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= max_len:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            while len(s) > max_len:
                chunks.append(s[:max_len])
                s = s[max_len:]
            cur = s
    if cur:
        chunks.append(cur)
    if overlap > 0 and len(chunks) > 1:
        merged: List[str] = []
        for i, c in enumerate(chunks):
            if i > 0:
                prev = chunks[i - 1]
                c = (prev[-overlap:] + " " + c).strip()
            merged.append(c)
        chunks = merged
    return chunks


def chunk_document(text: str, doc_id: str | None = None,
                  max_len: int = 400, overlap: int = 60) -> List[Candidate]:
    doc_id = doc_id or uuid.uuid4().hex[:12]
    chunks = chunk_text(text, max_len, overlap)
    return [
        Candidate(doc_id=doc_id, chunk_id=f"{doc_id}#c{i}",
                  text=c, score=0.0, meta={"index": i})
        for i, c in enumerate(chunks)
    ]
