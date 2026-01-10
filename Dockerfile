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
ARG MODEL_ID=onnx-community/SmolLM2-135M-ONNX
ARG ONNX_FILE=model_q4f16.onnx
ARG EXECUTION_PROVIDER=

ENV MODEL_ID=${MODEL_ID}
ENV ONNX_FILE=${ONNX_FILE}
ENV EXECUTION_PROVIDER=${EXECUTION_PROVIDER}

# Download metadata and model separately to ensure all files are captured correctly
RUN hf download ${MODEL_ID} --include "*.json" --include "*.txt" --include "*.model" --include "*.py" && \
    ONNX_BASE=$(echo ${ONNX_FILE} | sed 's/\.onnx$//') && \
    hf download ${MODEL_ID} --include "**/${ONNX_BASE}.onnx*" --include "**/*.data"

RUN adduser --disabled-password --gecos '' appuser && \
    mkdir -p ${HF_HOME} && \
    chown -R appuser:appuser /app ${HF_HOME}
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
