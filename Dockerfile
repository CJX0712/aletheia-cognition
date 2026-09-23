# Aletheia 澄明 — 容器镜像
# 作者: 晨星
# 注意: 模型权重 (models/) 不入库, 运行时通过挂载卷提供
#   docker build -t aletheia .
#   docker run -p 8000:8000 -v /path/to/models:/app/models aletheia
#
# llama-cpp-python (本地 GGUF 推理) 需要编译工具链, 默认不安装 —— 镜像默认使用
# 离线 MockLLM 后端, 开箱即用。需要真实本地推理时:
#   docker build --build-arg WITH_LLAMACPP=1 -t aletheia .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

ARG WITH_LLAMACPP=0

WORKDIR /app

COPY requirements.lock.txt .
RUN set -eux; \
    if [ "$WITH_LLAMACPP" = "1" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends build-essential cmake; \
        rm -rf /var/lib/apt/lists/*; \
        pip install -r requirements.lock.txt; \
    else \
        grep -viE '^[[:space:]]*llama-cpp-python' requirements.lock.txt > /tmp/requirements.lock; \
        pip install -r /tmp/requirements.lock; \
    fi

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/eval_questions.json ./data/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# models 由外部挂载至 /app/models
VOLUME ["/app/models"]
EXPOSE 8000

CMD ["python", "-m", "aletheia.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
