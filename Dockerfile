FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the model to prevent cold starts
ARG MODEL_ID=onnx-community/gemma-3-270m-it-ONNX
RUN hf download "${MODEL_ID}" --include "onnx/model_q4f16.onnx*" --include "*.json" --include "*.model"

RUN adduser --disabled-password --gecos '' appuser && \
    mkdir -p ${HF_HOME} && \
    chown -R appuser:appuser /app ${HF_HOME}
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
