# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
        libmagic1 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch keeps image smaller and avoids OOM on Render free tier
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/.cache/huggingface \
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libpoppler-cpp-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

COPY src/ src/
COPY frontend/ frontend/
COPY data/ data/
COPY start.sh start.sh

RUN mkdir -p data/uploads /tmp/.cache/huggingface && \
    chmod +x start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

CMD ["./start.sh"]
