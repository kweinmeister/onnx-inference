from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import logging
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
        ml_models["generator"] = OnnxTextGenerator()
        logger.info("✓ Model loaded successfully")
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

    # Create a thread-safe iterator wrapper
    async def async_generator():
        """Async wrapper for the synchronous generator."""
        # Create the iterator in a thread to run the initial logic
        iterator = await run_in_threadpool(
            gen.stream_generate,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        while True:
            try:
                # Use a sentinel to detect end of iteration safely across threads
                # Wrap next in a lambda to satisfy linter argument count checks
                result = await run_in_threadpool(lambda: next(iterator, None))
                if result is None:
                    break

                chunk, _ = result
                yield chunk
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield f"\n[ERROR: {str(e)}]"
                break

    return StreamingResponse(async_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
