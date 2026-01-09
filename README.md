# ONNX Inference for LLMs

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/onnx--runtime-1.23.2-green.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)](TESTING.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **A reference implementation for embedded LLM inference using ONNX Runtime.**

This project demonstrates how to run Large Language Models locally with hardware acceleration (CUDA, CoreML, CPU) and smart caching. It is designed to provide **easy, direct access** to inference from your own code for embedded applications.

*Note: This is a sample demonstration for local/embedded use cases, rather than a dedicated high-throughput production server like vLLM or TGI.*

---

## 🌟 Highlights

- **🚀 Fast Startup**: Zero network requests when cached
- **⚡ Hardware Acceleration**: Auto-selects best provider (CUDA, CoreML, or CPU)
- **💰 Cost-Effective**: Works with CPU-only, GPU optional
- **📦 Serverless-Ready**: Built for Google Cloud Run with scale-to-zero
- **🔌 Multiple Interfaces**: Python API, FastAPI server, or CLI

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Models](#-tested-models)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Python API](#1-python-api)
  - [Quickstart Examples](#2-quickstart-examples)
  - [FastAPI Server](#3-fastapi-server)
- [Deployment](#-deployment)
  - [Docker](#docker)
  - [Google Cloud Run](#google-cloud-run)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

Get started in 3 simple steps:

```bash
# 1. Clone the repository
git clone https://github.com/kweinmeister/onnx-inference.git
cd onnx-inference

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run your first inference
python -c "
from inference import OnnxTextGenerator
gen = OnnxTextGenerator()
result = gen.generate('The future of AI is', max_new_tokens=30)
print(result['generated_text'])
"
```

**First run**: Downloads models and caches locally  
**Subsequent runs**: Instant load from cache, zero network requests ⚡

**Hardware Acceleration**: Automatically uses the best available execution provider:

- **CUDA** (NVIDIA GPUs) for maximum performance
- **CoreML** (Apple Silicon) for efficient M1/M2/M3 acceleration  
- **CPU** as universal fallback

> [!NOTE]
> For CUDA GPU support, you must replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt` and ensure your host machine has the appropriate CUDA drivers installed. The default configuration uses the CPU-compatible package to ensure broad compatibility.

Or explore the examples:

```bash
python quickstart.py
```

---

## 🤖 Tested Models

This project has been tested with these ONNX-optimized models:

- **[gemma-3-270m-it-ONNX](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX)**
- **[SmolLM2-135M-ONNX](https://huggingface.co/onnx-community/SmolLM2-135M-ONNX)**

The inference engine is model-agnostic and should work with any ONNX-compatible language model.

---

## 📦 Installation

### Prerequisites

You'll need Python 3.12 or higher and at least 4GB of RAM for inference.

### Install

```bash
# Clone the repository
git clone https://github.com/kweinmeister/onnx-inference.git
cd onnx-inference

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Python API

The most direct way to use the models:

```python
from inference import OnnxTextGenerator

# Initialize with default model
generator = OnnxTextGenerator()

# Basic generation
result = generator.generate(
    prompt="Explain quantum computing in simple terms:",
    max_new_tokens=50,
    temperature=0.7
)
print(result['generated_text'])
print(f"Tokens: {result['tokens_generated']}, Reason: {result['finish_reason']}")
```

#### Streaming Generation

Perfect for real-time applications:

```python
# Stream tokens as they're generated
for chunk, metadata in generator.stream_generate(
    prompt="Write a short story:",
    max_new_tokens=100,
    temperature=0.8
):
    print(chunk, end='', flush=True)
```

#### Using Different Models

```python
# Use a different ONNX model
generator = OnnxTextGenerator(
    model_id="onnx-community/SmolLM2-135M-ONNX",
    onnx_file="onnx/model.onnx",
    allow_patterns=["onnx/model.onnx*", "*.json"]
)

result = generator.generate("def fibonacci(n):", max_new_tokens=40)
print(result['generated_text'])
```

#### Parameters

```python
result = generator.generate(
    prompt="Your prompt here",
    max_new_tokens=50,      # Maximum tokens to generate
    temperature=0.7,        # 0.0 = deterministic, 1.0+ = creative
    top_p=0.9              # Nucleus sampling threshold
)
```

### 2. Quickstart Examples

Run interactive demonstrations:

```bash
# Run all examples sequentially
python quickstart.py

# Run specific examples
python quickstart.py basic     # Basic text generation
python quickstart.py stream    # Streaming demo
python quickstart.py temp      # Temperature comparison
python quickstart.py code      # Code completion
python quickstart.py custom    # Custom parameters
```

### 3. FastAPI Server

Production-ready API server:

```bash
# Start the server
python app.py

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Making API requests:**

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain microservices:",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

---

## 🐳 Deployment

### Docker

Build and run containerized version:

```bash
# Build image (includes model download at build time)
# Optionally specify MODEL_ID build argument
docker build -t onnx-inference --build-arg MODEL_ID=onnx-community/gemma-3-270m-it-ONNX .

# Run container
docker run -p 8080:8080 onnx-inference

# Access at http://localhost:8080
```

**Note:** The default Dockerfile bakes in the Gemma-3 model for faster cold starts. You can change this by passing the `MODEL_ID` build argument.

**Why bake the model into the image?** Baking the model eliminates cold-start downloads, speeds up container startup, ensures consistent deployments, and enables offline operation.

### Google Cloud Run

Deploy to serverless infrastructure:

```bash
# Set your region
REGION=us-central1

# Deploy from source (builds automatically)
gcloud run deploy onnx-inference \
  --source . \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 4 \
  --allow-unauthenticated
```

**Configuration:**

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 2Gi | Model + overhead |
| CPU | 4 | Optimal for ONNX Runtime |
| Concurrency | 4 | Balance parallelism & resources |

**Get your service URL:**

```bash
SERVICE_URL=$(gcloud run services describe onnx-inference \
  --region $REGION \
  --format 'value(status.url)')

echo $SERVICE_URL
```

---

## 📡 API Reference

### POST `/generate`

Generate text from a prompt.

**Request:**

```json
{
  "prompt": "string (required)",
  "max_new_tokens": 50,         // optional, default: 50
  "temperature": 0.7,           // optional, default: 0.7
  "top_p": 0.9                  // optional, default: 0.9
}
```

**Response:**

```json
{
  "generated_text": "string",
  "finish_reason": "stop|length",  // stop=EOS token, length=max tokens reached
  "tokens_generated": 30
}
```

**Examples:**

```bash
# Creative writing
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time in a digital realm,",
    "max_new_tokens": 100,
    "temperature": 0.9
  }'

# Code generation
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def factorial(n):",
    "max_new_tokens": 40,
    "temperature": 0.3
  }'

# Few-shot PII masking
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Mask PII:\nText: Call 555-1234\nMasked: Call [PHONE]\nText: Email user@example.com\nMasked:",
    "max_new_tokens": 20,
    "temperature": 0.1
  }'
```

---

## 🧪 Testing

Test suite with mocked dependencies for fast iteration:

```bash
# Run all tests
uv run pytest -v

# Run with detailed output
uv run pytest -v -s
```

---

## 🏗️ Project Structure

```
onnx-inference/
├── inference.py              # Core inference engine
├── app.py                    # FastAPI server
├── test_inference.py         # Test suite with mocks
├── requirements.txt          # Dependencies
├── Dockerfile                # Container definition
└── README.md                 # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For bug reports and feature requests, please [open an issue](https://github.com/kweinmeister/onnx-inference/issues).

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
