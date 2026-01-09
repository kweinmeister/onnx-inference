from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app import app
import pytest

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
