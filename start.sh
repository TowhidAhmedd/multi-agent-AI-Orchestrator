#!/bin/sh
set -e

PORT="${PORT:-8000}"

echo "Starting Multi-Agent AI Orchestrator on 0.0.0.0:${PORT}"

exec uvicorn src.api.fastapi_app:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 75
