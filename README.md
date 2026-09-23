# Aletheia 澄明 · v1.0

> 推理时计算 Scaling 引擎 — 让 1.5B 本地小模型，通过推理时计算扩展
> （inference-time compute scaling）达到远超单 pass 的推理质量。
> 纯 CPU 可跑、零外部 API、离线可验证。

作者：**晨星** · License: MIT

---

## 它解决什么

大模型的质量不只取决于参数规模，也取决于**推理时投入多少计算**（OpenAI o1/o3、
DeepSeek-R1 已验证）。Aletheia 把这套范式落到一个能在本机 CPU 跑起来的系统里：
对同一个问题，**采样多条推理链 → 用验证器择优 / 多数投票**，把 1.5B 小模型的
事实准确率在客观评测集上做到 100%（CPU，无 GPU）。

## 核心特性

- **推理时计算引擎（M6）**：统一接口调度 5 种策略
  `greedy / best_of_n / self_consistency / beam / refine_reflect`
- **混合检索（M2）**：稠密向量（bge-small-zh ONNX + FAISS HNSW）+ 稀疏词项（BM25 + 中文 char/skip-gram 分词），RRF 融合
- **交叉编码器重排（M3）**：bge-reranker-base（ONNX，无 torch）
- **本地生成（M4）**：Qwen2.5-1.5B-Instruct Q4_K_M（llama.cpp，CPU 线程锁 4）
- **自适应预算路由（M7）**：按查询复杂度分配算力，预算↑⇒计算↑
- **答案验证器（M5）**：无需金标准的自洽打分（引用合规 + 覆盖 + 语义相关）
- **离线零依赖兜底**：HashEmbedder / MemoryStore / PassthroughReranker / MockLLM，无模型文件整套可验证
- **单一职责 + 协议隔离**：9 个模块，组件以 Protocol 声明，运行时注入，可独立单测

## 五条可机械验证的硬不变量

| # | 不变量 | 验证 |
|---|--------|------|
| 1 | `best_of_n(N≥1)` 的 final_score ≥ greedy 单条分数 | `tests/` 通过 |
| 2 | `self_consistency` 最终答案 == 多数投票 span（确定性） | `tests/` 通过 |
| 3 | `beam(width=1, N=1)` 答案 == greedy 答案（退化一致） | `tests/` 通过 |
| 4 | 给定查询，budget 单调 ⇒ N 非降且策略不降级 | `tests/` 通过 |
| 5 | 离线零依赖全绿（无网无 Key） | `pytest` 9 passed |

## 评测结果（真实模型，纯 CPU）

```
mode: online | total: 10 | hits: 10 | accuracy: 1.0 | elapsed_s: ~59
```

客观中文问答集（唯一确定答案，机械判分）。详见 `docs/USAGE.md`。

## 快速开始

```bash
make install            # 创建 .venv 并安装锁定依赖
make test               # 单元测试 + 不变量 (离线零依赖)
make offline-eval       # 离线评测 (仅验证链路健康)
make serve              # 启动 http://127.0.0.1:8000
# 或在线评测 (需 models/ 权重):
make eval
```

模型权重见 `docs/DEPLOYMENT.md`（不入库，需自行放置到 `models/`）。

## 项目结构

```
aletheia/
├── src/aletheia/
│   ├── protocol.py     M1 契约层 (数据类 + Protocol + 错误)
│   ├── config.py       M1 配置解析校验
│   ├── retrieval/      M2 稠密/稀疏/混合 (embed/store/sparse/hybrid)
│   ├── rerank.py       M3 cross-encoder 重排
│   ├── generator.py    M4 llama.cpp GGUF + Mock 兜底
│   ├── verifier.py     M5 自洽验证器
│   ├── reasoning.py    M6 推理时计算引擎 (核心)
│   ├── budget.py       M7 自适应预算路由
│   ├── assembly.py     运行时装配 (provider 自动回退)
│   ├── api.py          M8 FastAPI (/healthz /ingest /ask)
│   ├── cli.py          命令行入口
│   └── ingest.py       文档分块
├── tests/              M9 不变量 + API 测试
├── scripts/eval.py     端到端评测
├── data/eval_questions.json  客观评测集
├── web/console.html    单文件可视化控制台
├── docs/              ARCHITECTURE / SPEC / DEPLOYMENT / USAGE
├── requirements.lock.txt   版本锁定依赖
├── pyproject.toml  Makefile  Dockerfile  .github/workflows/ci.yml
```

## 文档

- `docs/ARCHITECTURE.md` — 系统架构、模块接口、数据流
- `docs/SPEC.md` — 规格契约（锁定范围/API/验收/不变量）
- `docs/DEPLOYMENT.md` — 部署与模型获取
- `docs/USAGE.md` — API / CLI / Web 控制台使用指南

---
© 2026 晨星. Built with local open-source AI.
