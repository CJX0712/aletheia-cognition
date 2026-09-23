# Aletheia 澄明 — 容器镜像
# 注意: 模型权重 (models/) 不入库, 运行时通过挂载卷提供
#   docker build -t aletheia .
#   docker run -p 8000:8000 -v /path/to/models:/app/models aletheia
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/eval_questions.json ./data/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# models 由外部挂载至 /app/models
VOLUME ["/app/models"]
EXPOSE 8000

CMD ["python", "-m", "aletheia.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
