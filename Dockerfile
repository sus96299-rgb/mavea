# ==================== MAVEA Dockerfile ====================
# 多阶段构建：builder 安装依赖，runner 精简运行

FROM python:3.10-slim AS builder

# 系统依赖：FFmpeg + OpenCV/SciPy 运行时库
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存层
COPY pyproject.toml requirements.txt ./
COPY src ./src

# 安装项目（不装 dev 依赖）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ==================== Runner ====================
FROM python:3.10-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r mavea && useradd -r -g mavea -m -d /home/mavea mavea

WORKDIR /app

# 从 builder 复制已安装的 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/mavea /usr/local/bin/mavea
COPY --from=builder /usr/local/bin/mavea-mcp /usr/local/bin/mavea-mcp
COPY --from=builder /usr/local/bin/mavea-web /usr/local/bin/mavea-web

# 复制项目源码（MCP 入口、RAG 模板等包内数据需要）
COPY --from=builder /app/src ./src
COPY pyproject.toml ./

# 工作目录与数据卷
RUN mkdir -p /app/workspace/output /app/workspace/temp /app/data/vector_db \
    && chown -R mavea:mavea /app
USER mavea

ENV MAVEA_WORKSPACE_DIR=/app/workspace \
    MAVEA_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Gradio 默认端口
EXPOSE 7860
# MCP SSE 默认端口
EXPOSE 8765

# 默认启动 Gradio WebUI
CMD ["mavea-web", "--host", "0.0.0.0", "--port", "7860"]
