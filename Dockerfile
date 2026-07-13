FROM python:3.12-slim

ARG BAKE_OCR_MODELS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PORT=8001 \
    OCR_WARMUP_ON_STARTUP=true \
    PADDLE_CPU_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /home/appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

USER appuser

RUN if [ "$BAKE_OCR_MODELS" = "true" ]; then \
        python -c "from app.ocr.paddle_engine import PaddleOcrEngine; PaddleOcrEngine(cpu_threads=2)"; \
    else \
        echo "Skipping OCR model bake"; \
    fi

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
