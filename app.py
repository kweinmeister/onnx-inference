import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from inference import OnnxTextGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store model instance
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization and cleanup."""
    # Load the model during startup
    logger.info("Initializing ONNX model...")
    try:
        import os

        model_id = os.getenv("MODEL_ID")
        onnx_file = os.getenv("ONNX_FILE")
        execution_provider = os.getenv("EXECUTION_PROVIDER")

        # Initialize with env vars if present, otherwise uses class defaults
        init_kwargs: Dict[str, Any] = {}
        if model_id:
            init_kwargs["model_id"] = model_id
        if onnx_file:
            init_kwargs["onnx_file"] = onnx_file
        if execution_provider:
            init_kwargs["execution_providers"] = execution_provider

        ml_models["generator"] = OnnxTextGenerator(**init_kwargs)
        logger.info(
            f"✓ Model loaded (ID: {model_id or 'default'}, File: {onnx_file or 'default'}, EP: {execution_provider or 'auto'})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize ONNX model: {e}")
        raise

    yield

    # Clean up resources
    logger.info("Cleaning up resources...")
    ml_models.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan,
    title="ONNX Text Generation API",
    description="FastAPI server for ONNX-based text generation with streaming support",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: int = Field(
        50, description="Maximum number of tokens to generate", ge=1, le=500
    )
    temperature: float = Field(
        0.7, description="Sampling temperature (0 for greedy)", ge=0.0, le=2.0
    )
    top_p: float = Field(0.9, description="Nucleus sampling parameter", ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    generated_text: str = Field(..., description="Generated text")
    finish_reason: str = Field(
        ...,
        description="Reason generation stopped: 'stop' (EOS) or 'length' (max tokens)",
    )
    tokens_generated: int = Field(..., description="Number of tokens generated")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "ONNX Text Generation API is running",
        "endpoints": ["/generate", "/stream_generate"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text completion for the given prompt.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        GenerateResponse with generated text
    """
    generator = ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="Model not initialized")

    logger.info(f"Generating text for prompt: '{request.prompt[:50]}...'")

    try:
        # Run synchronous generation in a separate thread to avoid blocking the event loop
        result = await run_in_threadpool(
            generator.generate,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        logger.info(
            f"Generated {result['tokens_generated']} tokens, finish_reason: {result['finish_reason']}"
        )

        return GenerateResponse(
            generated_text=result["generated_text"],
            finish_reason=result["finish_reason"],
            tokens_generated=result["tokens_generated"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream_generate")
async def stream_generate(request: GenerateRequest):
    """
    Stream generated text tokens for the given prompt.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        StreamingResponse with generated text chunks
    """
    generator = ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=503, detail="Model not initialized")

    logger.info(f"Streaming generation for prompt: '{request.prompt[:50]}...'")

    # Capture generator in closure
    gen = generator
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def producer():
        """Synchronous producer running in a thread pool."""
        try:
            # The synchronous generator runs entirely in this background thread
            iterator = gen.stream_generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            for chunk, _ in iterator:
                # Use call_soon_threadsafe/run_coroutine_threadsafe to push to queue from thread
                loop.call_soon_threadsafe(queue.put_nowait, chunk)

            # Sentinel for completion
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, f"\n[ERROR: {str(e)}]")
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start the generation loop in the background
    loop.run_in_executor(None, producer)

    async def consumer() -> AsyncGenerator[str, None]:
        """Asynchronous consumer yielding from the queue."""
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(consumer(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
