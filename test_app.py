import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app, lifespan

client = TestClient(app)


# Mock the ml_models dictionary to avoid loading the actual model
@pytest.fixture
def mock_ml_models():
    with patch("app.ml_models", {}) as mock_models:
        yield mock_models


@pytest.fixture
def mock_generator():
    generator = MagicMock()
    # Mock generate method
    generator.generate.return_value = {
        "generated_text": "Test response",
        "finish_reason": "length",
        "tokens_generated": 2,
    }

    # Mock stream_generate method
    def mock_stream(*args, **kwargs):
        yield "Test", {"tokens_generated": 1, "finish_reason": None}
        yield " response", {"tokens_generated": 2, "finish_reason": "length"}

    generator.stream_generate = mock_stream
    return generator


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_generate_endpoint(mock_ml_models, mock_generator):
    # Inject mock generator
    mock_ml_models["generator"] = mock_generator

    response = client.post(
        "/generate", json={"prompt": "Test prompt", "max_new_tokens": 10}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == "Test response"
    assert data["finish_reason"] == "length"
    assert data["tokens_generated"] == 2


def test_generate_model_not_initialized(mock_ml_models):
    # Ensure no generator present
    mock_ml_models.clear()

    response = client.post("/generate", json={"prompt": "Test prompt"})
    assert response.status_code == 503


def test_stream_generate_endpoint(mock_ml_models, mock_generator):
    mock_ml_models["generator"] = mock_generator

    with client.stream("POST", "/stream_generate", json={"prompt": "Test"}) as response:
        assert response.status_code == 200
        chunks = list(response.iter_text())
        assert "".join(chunks) == "Test response"


@pytest.mark.asyncio
async def test_lifespan_startup_shutdown(mock_ml_models):
    """Test standard startup and shutdown with default env."""
    mock_init = MagicMock()

    with (
        patch("app.OnnxTextGenerator", side_effect=mock_init) as mock_cls,
        patch.dict(os.environ, {}, clear=True),
    ):
        async with lifespan(app):
            # STARTUP
            mock_cls.assert_called_once()
            assert "generator" in mock_ml_models
            assert mock_ml_models["generator"] == mock_init.return_value

        # SHUTDOWN
        assert len(mock_ml_models) == 0


@pytest.mark.asyncio
async def test_lifespan_startup_env_vars(mock_ml_models):
    """Test startup with environment variable overrides."""
    mock_init = MagicMock()
    env = {
        "MODEL_ID": "custom/model",
        "ONNX_FILE": "custom.onnx",
        "EXECUTION_PROVIDER": "cuda",
    }

    with (
        patch("app.OnnxTextGenerator", side_effect=mock_init) as mock_cls,
        patch.dict(os.environ, env),
    ):
        async with lifespan(app):
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model_id"] == "custom/model"
            assert call_kwargs["onnx_file"] == "custom.onnx"
            assert call_kwargs["execution_providers"] == "cuda"


@pytest.mark.asyncio
async def test_lifespan_startup_failure(mock_ml_models):
    """Test startup exception handling."""
    with patch("app.OnnxTextGenerator", side_effect=RuntimeError("Startup Failed")):
        with pytest.raises(RuntimeError, match="Startup Failed"):
            async with lifespan(app):
                pass
