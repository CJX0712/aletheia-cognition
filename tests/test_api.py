"""API 层测试: 用 mock 引擎注入, 验证 HTTP 端点契约 (无模型/无网)."""
import sys

sys.path.insert(0, "src")

import pytest
from fastapi.testclient import TestClient

import aletheia.api as api
from aletheia.generator import MockLLM
from aletheia.protocol import Strategy
from aletheia.reasoning import ReasoningEngine
from aletheia.verifier import RuleVerifier


class FakeEngine:
    def __init__(self):
        self._eng = ReasoningEngine(MockLLM(), RuleVerifier(None))
        self.chunks = 0

    def health(self):
        return {"status": "ok", "components": {"embed": "hash", "store": "memory",
                                               "rerank": "rrf", "llm": "mock"},
                "chunks": self.chunks}

    def ingest(self, text, doc_id=None):
        self.chunks += 1
        return {"ingested_chunks": 1, "total_chunks": self.chunks,
                "doc_id": doc_id or "x"}

    def ask(self, query, strategy=None, n=None, budget=1.0, seed=42):
        return self._eng.solve(query, [], strategy or Strategy.BEST_OF_N, n or 3, seed)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "AletheiaEngine", FakeEngine)
    api._engine = FakeEngine()
    return TestClient(api.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"]["llm"] == "mock"


def test_ingest_empty_rejected(client):
    r = client.post("/ingest", json={"text": "   "})
    assert r.status_code == 400


def test_ask_empty_rejected(client):
    r = client.post("/ask", json={"query": ""})
    assert r.status_code == 400


def test_ask_returns_trace(client):
    r = client.post("/ask", json={"query": "光合作用释放什么气体", "strategy": "best_of_n", "n": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and body["answer"]
    assert body["strategy"] == "best_of_n"
    assert len(body["steps"]) == 3
    assert body["final_score"] >= 0.0


def test_ingest_then_ask(client):
    r = client.post("/ingest", json={"text": "氧气约占大气21%。"})
    assert r.json()["ingested_chunks"] >= 1
    r = client.post("/ask", json={"query": "氧气占比"})
    assert r.status_code == 200
