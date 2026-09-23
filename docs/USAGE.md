# 使用指南 — Aletheia 澄明

## 1. 命令行

```bash
# 摄入文档 (txt / md / pdf)
aletheia ingest path/to/doc.txt

# 提问（默认走自适应预算路由）
aletheia ask "光合作用释放什么气体"

# 指定策略与采样数
aletheia ask "为什么天空是蓝色的" --strategy self_consistency --n 5
aletheia ask "计算 12*8 等于多少" --strategy best_of_n --n 4 --budget 1.5

# 启动服务
aletheia serve --host 0.0.0.0 --port 8000
```

## 2. HTTP API

```bash
# 健康检查
curl http://127.0.0.1:8000/healthz
# -> {"status":"ok","components":{"embed":"onnx","store":"faiss","rerank":"onnx","llm":"gguf"},"chunks":0}

# 摄入
curl -X POST http://127.0.0.1:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text":"光合作用是植物利用光能将二氧化碳和水转化为葡萄糖并释放氧气的过程。"}'

# 提问（返回完整推理轨迹）
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"光合作用释放什么气体","strategy":"best_of_n","n":3}'
```

`/ask` 响应字段：

| 字段 | 含义 |
|------|------|
| `answer` | 最终答案（含 `[n]` 来源引用） |
| `strategy` | 实际使用的推理策略 |
| `final_score` | 验证器综合分（0–1） |
| `wall_ms` | 端到端耗时（毫秒） |
| `contexts[]` | 检索到的上下文片段（doc_id/chunk_id/text/score） |
| `steps[]` | 推理轨迹逐步（strategy/text/score） |

## 3. Web 控制台

浏览器打开 `web/console.html`（单文件，零依赖），默认请求 `http://127.0.0.1:8000`：
- 文档录入区：粘贴文本 → 录入
- 提问区：输入问题、选策略、调 N、提交
- 结果区：最终答案、分数进度条、耗时、检索上下文、推理轨迹展开

## 4. 策略选择建议

| 场景 | 推荐策略 | N |
|------|----------|---|
| 简单事实（是谁/是什么） | `greedy`（或 auto） | 1 |
| 通用问答 | `best_of_n`（或 auto） | 3–4 |
| 逻辑/数学/多选 | `self_consistency` | 5–8 |
| 复杂开放推理 | `beam` / `refine_reflect` | 3–4 |
| 不确定 | `auto`（预算路由自动选） | — |

`budget` 参数（默认 1.0）整体放大计算投入：预算越高，复杂查询分配更多 N、更强策略。

## 5. 评测

```bash
ALETHEIA_OFFLINE=1 python scripts/eval.py   # 离线（仅链路健康）
python scripts/eval.py                        # 在线（真实模型，准确率）
```

评测集：`data/eval_questions.json`（中文客观题，唯一确定答案，机械判分）。
结果写入 `data/eval_report.json`。

## 6. 测试

```bash
python -m pytest tests/ -q     # 9 项不变量 + 5 项 API 契约，离线零依赖全绿
```
