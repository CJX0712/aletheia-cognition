# 部署指南 — Aletheia 澄明

## 0. 环境要求

- Python 3.11+（推荐 3.13，已验证）
- pip 可访问 PyPI 或国内镜像（所有依赖均有预编译 wheel，**无需 C/C++ 编译器**）
- 内存 ≥ 8GB（GGUF 1.1G + rerank ONNX 1.1G 常驻）
- 磁盘 ≥ 4GB（模型权重）

## 1. 安装依赖

```bash
git clone <repo> && cd aletheia
make install          # 或: python -m venv .venv && .venv/bin/pip install -r requirements.lock.txt
```

国内镜像：`pip install -r requirements.lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`

## 2. 获取模型权重（不入库）

系统需要 3 个本地模型文件，放到 `models/` 目录：

| 用途 | 文件 | 来源 | 说明 |
|------|------|------|------|
| 嵌入 | `models/embed/model.onnx` + `tokenizer.json` | bge-small-zh-v1.5 ONNX 导出 | 中文句向量，dim=512 |
| 重排 | `models/rerank/model.onnx` + `tokenizer.json` | bge-reranker-base ONNX 导出 | cross-encoder |
| 生成 | `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Qwen2.5-1.5B-Instruct Q4_K_M | llama.cpp 格式 |

获取方式（任选）：
- **HuggingFace**：从 `BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base` 下载并 ONNX 导出；从 `Qwen/Qwen2.5-1.5B-Instruct` 用 `llama.cpp/convert` 转 GGUF 并量化。
- **已验证快照**：若运行环境已具备（如本仓库开发机），直接硬链接/复制 `models/` 即可。

> 缺失任一模型时，系统自动回退到零依赖兜底（hash 嵌入 / 内存索引 / MockLLM / 直通重排），
> 仍能启动并跑通离线测试，仅生产质量下降。

## 3. 运行

```bash
# 开发 / 本地
make serve                                   # http://127.0.0.1:8000
aletheia serve --host 0.0.0.0 --port 8000    # 命令行等价

# 录入与提问（CLI）
aletheia ingest docs/foo.txt
aletheia ask "光合作用释放什么气体" --strategy best_of_n --n 4
```

## 4. Docker

```bash
docker build -t aletheia .
# models 通过卷挂载 (不入库)
docker run -p 8000:8000 -v /abs/path/models:/app/models aletheia
```

## 5. 环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `ALETHEIA_OFFLINE` | 强制零依赖兜底（无模型也跑） | 未设 |
| `ALETHEIA_ONLINE` | 触发在线 E2E 测试 | 未设 |

## 6. 干净环境一键复现

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m pytest tests/ -q          # 离线零依赖全绿
ALETHEIA_OFFLINE=1 .venv/bin/python scripts/eval.py   # 链路健康
```

## 7. 生产化建议

- 用 gunicorn + uvicorn workers 横向扩展（注意每 worker 加载一份模型，内存 ×N）
- 向量索引可替换为持久化（FAISS 落盘 / 外接向量库）以支持重启保留
- 加反向代理（nginx）做 TLS 与限流
