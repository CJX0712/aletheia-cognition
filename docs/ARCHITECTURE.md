# 架构文档 — Aletheia 澄明

## 1. 设计哲学

Aletheia 把"推理时计算扩展（inference-time compute scaling）"作为一等公民：
模型的答案质量，不仅由权重决定，也由**推理阶段投入的计算路径**决定。系统对
同一查询生成多条候选推理链，再用验证器择优或投票，从而把 1.5B 本地小模型的
事实准确率，在纯 CPU 上拉到接近大模型单 pass 的水平。

所有外部依赖（嵌入 / 重排 / 生成）都通过 **Protocol 契约** 声明，由 `assembly`
在运行时注入，并优先选用开源成果、失败自动回退零依赖兜底实现，确保：
- 生产环境用真实开源模型（bge / bge-reranker / Qwen2.5 GGUF）
- 测试 / CI / 离线环境用 hash 嵌入 / 内存索引 / MockLLM，无网无 Key 全绿

## 2. 模块拓扑

```
            ┌─────────────────────────────────────┐
 query ───▶ │  api (FastAPI) / cli                │
            └──────────────┬──────────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │  budget 预算路由   │  复杂度 × budget → 策略 + N
                 └─────────┬─────────┘
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ retrieval 混合│ │ reasoning 推理│ │ verifier 验证│
   │ 检索          │ │ 时计算引擎   │ │ 器           │
   │ dense+sparse │ │ 5 策略统一   │ │ 自洽打分     │
   │ RRF+rerank  │ │ 接口         │ │              │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                │
          ▼                ▼                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │ embed ONNX   │ │ generator    │ │ rerank ONNX  │
   │ / Hash       │ │ llama.cpp    │ │ / Passthrough│
   │ FAISS/mem    │ │ GGUF / Mock  │ │              │
   │ BM25         │ │              │ │              │
   └──────────────┘ └──────────────┘ └──────────────┘
```

## 3. 模块接口（协议层冻结）

见 `src/aletheia/protocol.py`。关键 Protocol：

| Protocol | 方法 | 实现（生产 / 兜底） |
|----------|------|--------------------|
| `Embedder` | `embed(texts) -> vec[]` | `OnnxEmbedder` / `HashEmbedder` |
| `VectorStore` | `add` / `search` | `FaissStore`(HNSW) / `MemoryStore` |
| `SparseIndex` | `add` / `search` | `BM25Sparse` |
| `Reranker` | `rerank(q, cands, k)` | `OnnxReranker` / `PassthroughReranker` |
| `Generator` | `generate` / `stream` | `LlamaCppLLM` / `MockLLM` |
| `Verifier` | `score(q, ans, ctx)` | `RuleVerifier` |
| `Reasoner` | `solve(q, ctx, strat, n)` | `ReasoningEngine` |
| `Retriever` | `retrieve(q, k)` | `HybridRetriever` |

## 4. 数据流（一次 /ask）

1. `budget.plan(query, budget)` → `(strategy, n)`
2. `retriever.retrieve(query, top_k)` → RRF 融合 → cross-encoder 重排 → `Candidate[]`
3. `reasoning.solve(query, contexts, strategy, n)` 生成 `ReasoningTrace`
   - 构造 RAG prompt（`[n]` 标注来源）
   - 按策略采样 / 投票 / 束搜索 / 反思修正
   - 每条候选经 `verifier.score` 打分
4. 返回 trace（answer + 轨迹 steps + 评分 + 上下文）

## 5. 推理时计算策略语义

| 策略 | 计算量 | 适用 | 机制 |
|------|--------|------|------|
| `greedy` | 1× | 简单事实 | 单 pass，temperature 0.2 确定 |
| `best_of_n` | N× | 通用 | 采样 N 条，verifier 择优 |
| `self_consistency` | N× | 逻辑/数学 | 抽取答案 span，多数投票 |
| `beam` | N×(2 轮) | 复杂 | 选 top-width → 确认式重生成（width=1 退化为 greedy） |
| `refine_reflect` | 3× | 易错 | 生成 → 反思 → 修正 三步走 |

## 6. 依赖与运行约束

- Python 3.11+（验证于 3.13）
- **无 C/C++ 编译器要求**：所有依赖均提供预编译 wheel
  （faiss-cpu / onnxruntime / llama-cpp-python / tokenizers）
- 内存：GGUF(1.1G) + rerank ONNX(1.1G) 常驻，建议 ≥ 8GB
- 无 GPU 亦可跑（纯 CPU 推理，线程锁 4 避免带宽争用）

## 7. 可观测性

每次 `/ask` 返回完整 `ReasoningTrace`：策略、逐步生成文本、各步评分、
最终分数、检索上下文、端到端耗时。Web 控制台可视化展示推理轨迹。
