# 规格契约 — Aletheia 澄明 v1.0

> 状态：已确认（基于 MVP 专家团三文档 + 用户需求）
> 本文档是开发唯一依据；范围外功能一律不做。

## 1. 产品定义

- **一句话**：在纯 CPU 上，用推理时计算扩展把 1.5B 本地小模型的推理质量拉到接近 7B 单 pass 水平的端到端 RAG + 推理系统。
- **目标用户**：需要本地/私有化、离线可用、零外部 API 的 AI 应用开发者与研究者。
- **核心问题**：小模型单 pass 事实准确率低；通过推理时计算（采样+验证）提升质量而不依赖更大模型或 GPU。

## 2. MVP 范围（锁定）

| 优先级 | 功能 | 验收摘要 |
|--------|------|----------|
| P0 | 混合检索（dense+sparse+RRF+rerank） | recall@k≥0.9 on 评测集 |
| P0 | 本地生成（GGUF） | /ask 返回 grounded 答案 |
| P0 | 推理时计算 5 策略 | 不变量 1–4 全通过 |
| P0 | 自适应预算路由 | 不变量 4 通过 |
| P0 | 答案验证器（gold-free） | 返回 0–1 分数 |
| P0 | HTTP API + CLI | /healthz /ingest /ask |
| P1 | Web 控制台（单文件 HTML） | 展示 trace + 录入 |
| P1 | 中文客观评测集 + 脚本 | 在线准确率 ≥ 0.9 |

## 3. 明确不做（Out-of-Scope）

| 不做 | 原因 | 何时考虑 |
|------|------|----------|
| 微调 / 训练 | 本机无 GPU，超出 MVP | v2.0 |
| 多模态（图/音） | 模块边界外 | v2.0 |
| 分布式部署 | 单节点 CPU 足矣 | 规模化时 |
| 用户账户体系 | MVP 单机私有化 | 多租户时 |

## 4. 技术架构（版本锁定）

| 层 | 技术 | 版本 | 锁定原因 |
|----|------|------|----------|
| 语言 | Python | 3.13 | 已验证 wheel 栈 |
| 嵌入 | bge-small-zh ONNX | — | 中文句向量，ONNX Runtime 无 torch |
| 向量库 | faiss-cpu | 1.15.0 | HNSW，预编译 wheel |
| 稀疏 | rank-bm25 | 0.2.2 | 纯 Python，零依赖 |
| 重排 | bge-reranker-base ONNX | — | cross-encoder，ONNX Runtime |
| 生成 | llama-cpp-python + Qwen2.5-1.5B Q4_K_M | 0.3.19 | 本地 GGUF，无需服务进程 |
| 服务 | fastapi + uvicorn | 0.141.1 / 0.53.0 | ASGI 标准栈 |
| 测试 | pytest | 8.4.2 | 不变量验证 |

## 5. API 端点（锁定）

| Method | Path | 功能 | 认证 | 请求体 | 响应 |
|--------|------|------|------|--------|------|
| GET | /healthz | 健康检查 | 无 | — | `{status, components, chunks}` |
| POST | /ingest | 摄入文档 | 无 | `{text, doc_id?}` | `{ingested_chunks, total_chunks, doc_id}` |
| POST | /ask | 推理问答 | 无 | `{query, strategy?, n?, budget?, seed?}` | `ReasoningTrace` JSON |

`/ask` 返回：`{answer, strategy, final_score, wall_ms, contexts[], steps[]}`。

## 6. 数据库 / 存储

无传统数据库。状态存储：
- 向量索引（FAISS）+ 稀疏索引（BM25）+ 文档分块：进程内存（`HybridRetriever`）
- 模型权重：`models/`（不入库，挂载/放置）

## 7. 页面清单（Web 控制台）

| 页面 | 路由 | 核心组件 | 设计 Token |
|------|------|----------|-----------|
| 控制台 | `web/console.html` | 录入区 / 提问区 / 轨迹展示 | 蓝青中性色，浅色主题 |

## 8. 设计 Token（Web 控制台）

- 主色：`#185FA5`（蓝）/ `#0F6E56`（青）
- 背景：`#FFFFFF` / 表面 `#F1EFE8`
- 边框：`rgba(0,0,0,0.15)`
- 字体：系统无衬线 + `Noto Sans SC`
- 图标：内联 SVG（描边，16/20/24px），**禁用 emoji**
- 主题：浅色
- 约束：禁用紫粉渐变、禁用硬编码颜色、禁用 AI 模板味文案

## 9. 验收标准（锁定 — QA 唯一依据）

| 编号 | 功能 | 验收（EARS） |
|------|------|--------------|
| AC-01 | 推理 | While 用户提交合法 query，系统**必须**返回含 answer 的 trace |
| AC-02 | 推理 | If 查询为空，系统**必须**返回 400 |
| AC-03 | 摄入 | If 文本为空，系统**必须**返回 400 |
| AC-04 | 不变量1 | best_of_n(N≥1) 的 final_score **必须** ≥ greedy 单条分数 |
| AC-05 | 不变量2 | self_consistency 的 answer **必须**等于多数投票 span |
| AC-06 | 不变量3 | beam(width=1,N=1) 的 answer **必须**等于 greedy answer |
| AC-07 | 不变量4 | 给定查询，budget 增大**必须**使 N 非降且策略不降级 |
| AC-08 | 在线质量 | 客观评测集准确率**应该** ≥ 0.9 |

## 10. 边界与约束

- 不支持 IE 浏览器
- 上下文窗口：GGUF n_ctx=4096
- 性能目标：单问（best_of_n N=3）端到端 < 10s（CPU）
- 内存 ≥ 8GB（模型常驻）

## 11. 内嵌已知坑（从记忆/实践）

| 坑 | 指纹 | 根因 | 修法 |
|----|------|------|------|
| llama.cpp 线程过多反慢 | llama-cpp-python | 小量化模型瓶颈在内存带宽 | 线程锁 4（ADR-0003） |
| doc_id 重复导致 chunk 冲突 | HybridRetriever | 多次 ingest 同 doc_id 覆盖 chunk_id | 每文档唯一 doc_id |
| ONNX 输入名不一致 | onnxruntime | 不同导出含/不含 token_type_ids | 运行时探测 input names |
| 中文零空格分词 | BM25Sparse | 英文分词器对中文失效 | char/skip-gram 分词 |

## 12. 端到端验证步骤

```bash
# 1. 安装
make install
# 2. 单元测试 + 不变量 (离线零依赖)
make test
# 3. 离线链路健康检查
make offline-eval
# 4. 在线评测 (需 models/ 权重)
make eval
# 5. 启动服务
make serve
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"光合作用释放什么气体","strategy":"best_of_n","n":3}'
```

## 13. 变更记录

| 日期 | 变更 | 原因 | 影响 |
|------|------|------|------|
| 2026-09-24 | 初始 v1.0 | 新建 | 全量 |
