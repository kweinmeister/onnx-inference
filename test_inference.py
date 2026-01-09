import pytest
import numpy as np
from unittest.mock import Mock, patch
from inference import OnnxTextGenerator


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Create a mock tokenizer that behaves like the real one."""
    tokenizer = Mock()

    # Mock encoding object
    def mock_encode(text: str) -> Mock:
        encoding = Mock()
        tokens = text.split() if text else []
        encoding.ids = list(range(1, len(tokens) + 2))
        encoding.attention_mask = [1] * len(encoding.ids)
        return encoding

    tokenizer.encode = mock_encode

    tokenizer.token_to_id = lambda t: 2 if t == "</s>" else None
    tokenizer.decode = lambda ids, *args, **kwargs: " ".join(
        [f"word{i}" for i in range(len(ids))]
    )

    return tokenizer


@pytest.fixture
def mock_onnx_session() -> Mock:
    """Create a mock ONNX session that simulates model inference."""
    session = Mock()

    # Mock outputs
    output = Mock()
    output.name = "logits"

    outputs = [output]
    for i in range(4):
        out = Mock()
        out.name = f"present.{i}.key"
        outputs.append(out)
        out = Mock()
        out.name = f"present.{i}.value"
        outputs.append(out)

    session.get_outputs.return_value = outputs

    inputs = []
    # input_ids, attention_mask, position_ids
    for name in ["input_ids", "attention_mask", "position_ids"]:
        inp = Mock()
        inp.name = name
        inputs.append(inp)

    # past_key_values
    for i in range(4):
        for typ in ["key", "value"]:
            inp = Mock()
            inp.name = f"past_key_values.{i}.{typ}"
            inp.shape = [1, 8, 0, 64]
            inp.type = "tensor(float16)"
            inputs.append(inp)

    session.get_inputs.return_value = inputs

    def mock_run(output_names, ort_inputs):
        seq_len = ort_inputs["input_ids"].shape[1]
        logits = np.random.randn(1, seq_len, 50000).astype(np.float32)
        logits[0, -1, 100:110] += 5.0

        results = [logits]
        for _ in range(8):
            results.append(np.zeros((1, 8, 1, 64), dtype=np.float32))
        return results

    session.run = mock_run

    # Mock get_providers to return a list of available providers
    session.get_providers.return_value = ["CPUExecutionProvider"]

    return session


@pytest.fixture
def mock_snapshot_download():
    """Mock snapshot_download to avoid downloading models."""
    with patch("inference.snapshot_download") as mock_download:
        mock_download.return_value = "/fake/model/path"
        yield mock_download


@pytest.fixture
def mock_tokenizer_from_file(mock_tokenizer):
    """Mock Tokenizer.from_file to return our mock tokenizer."""
    with patch("inference.Tokenizer.from_file") as mock_from_file:
        mock_from_file.return_value = mock_tokenizer
        yield mock_from_file


@pytest.fixture
def mock_inference_session(mock_onnx_session):
    """Mock ort.InferenceSession to return our mock session."""
    with patch("inference.ort.InferenceSession") as mock_session:
        mock_session.return_value = mock_onnx_session
        yield mock_session


@pytest.fixture
def mocked_generator(
    mock_snapshot_download, mock_tokenizer_from_file, mock_inference_session
) -> OnnxTextGenerator:
    """Create a fully mocked OnnxTextGenerator."""
    # Mock existence of generation_config.json
    with patch("os.path.exists", return_value=False):
        return OnnxTextGenerator()


class TestOnnxTextGeneratorMocked:
    """Test suite for OnnxTextGenerator using mocks (no model downloads)."""

    def test_initialization(self, mocked_generator) -> None:
        """Test that the generator initializes correctly with mocks."""
        assert mocked_generator.tokenizer is not None
        assert mocked_generator.session is not None
        assert mocked_generator.eos_token_id == 2
        assert len(mocked_generator.output_names) > 0

    def test_basic_generation(self, mocked_generator) -> None:
        """Test basic text generation."""
        prompt = "Hello, my name is"
        result = mocked_generator.generate(prompt, max_new_tokens=5, temperature=0.7)

        assert result is not None
        assert isinstance(result, dict)
        assert "generated_text" in result
        assert "finish_reason" in result
        assert "tokens_generated" in result
        # With mocks, we should get some generated text
        assert len(result["generated_text"]) > 0

    def test_streaming_generation(self, mocked_generator) -> None:
        """Test streaming generation."""
        prompt = "Hello, my name is"
        chunks = []

        for chunk, metadata in mocked_generator.stream_generate(
            prompt, max_new_tokens=5, temperature=0.7
        ):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        # Should generate some chunks
        assert len(chunks) > 0
        full_output = "".join(chunks)
        assert len(full_output) > 0

    def test_greedy_generation(self, mocked_generator) -> None:
        """Test greedy generation (temperature=0)."""
        prompt = "The capital of France is"
        result = mocked_generator.generate(prompt, max_new_tokens=3, temperature=0)

        assert result is not None
        assert isinstance(result, dict)
        assert "generated_text" in result

    def test_greedy_is_deterministic(self, mocked_generator) -> None:
        """Test that greedy sampling is deterministic."""
        prompt = "Test prompt"

        result1 = mocked_generator.generate(prompt, max_new_tokens=3, temperature=0)
        result2 = mocked_generator.generate(prompt, max_new_tokens=3, temperature=0)

        # With temperature=0, results should be identical
        assert result1["generated_text"] == result2["generated_text"]

    def test_empty_prompt(self, mocked_generator) -> None:
        """Test generation with an empty prompt."""
        result = mocked_generator.generate("", max_new_tokens=3, temperature=0.7)
        assert isinstance(result, dict)

    def test_max_tokens_respected(self, mocked_generator) -> None:
        """Test that max_new_tokens limit is respected."""
        prompt = "Count: "
        max_tokens = 3

        chunks = []
        for chunk, metadata in mocked_generator.stream_generate(
            prompt, max_new_tokens=max_tokens, temperature=0.7
        ):
            chunks.append(chunk)

        # Should not exceed max_tokens (may be less if EOS encountered)
        assert len(chunks) <= max_tokens

    def test_different_temperature_values(self, mocked_generator) -> None:
        """Test generation with different temperature values."""
        prompt = "The weather is"

        # Greedy (temp=0)
        result_greedy = mocked_generator.generate(
            prompt, max_new_tokens=3, temperature=0
        )

        # Low temp
        result_low = mocked_generator.generate(
            prompt, max_new_tokens=3, temperature=0.3
        )

        # High temp
        result_high = mocked_generator.generate(
            prompt, max_new_tokens=3, temperature=1.5
        )

        assert isinstance(result_greedy, dict)
        assert isinstance(result_low, dict)
        assert isinstance(result_high, dict)

    def test_different_top_p_values(self, mocked_generator) -> None:
        """Test generation with different top_p values."""
        prompt = "AI is"

        result_low = mocked_generator.generate(
            prompt, max_new_tokens=3, temperature=0.7, top_p=0.5
        )
        result_high = mocked_generator.generate(
            prompt, max_new_tokens=3, temperature=0.7, top_p=0.95
        )

        assert isinstance(result_low, dict)
        assert isinstance(result_high, dict)

    def test_kv_cache_logic(self, mocked_generator) -> None:
        """Test that KV cache is being extracted and updated."""
        outputs: list[np.ndarray] = []
        # Create output with key/values
        outputs.append(np.zeros((1, 1, 10), dtype=np.float32))  # logits

        # Add 1 layer of KV
        # present.0.key
        outputs.append(np.zeros((1, 8, 5, 64), dtype=np.float32))
        # present.0.value
        outputs.append(np.zeros((1, 8, 5, 64), dtype=np.float32))

        # Setup output names map manually for test
        mocked_generator.output_names = ["logits", "present.0.key", "present.0.value"]
        mocked_generator.output_to_input_map = {
            "present.0.key": "past_key_values.0.key",
            "present.0.value": "past_key_values.0.value",
        }

        past = mocked_generator._extract_past_from_outputs(outputs)

        assert "past_key_values.0.key" in past
        assert "past_key_values.0.value" in past
        assert past["past_key_values.0.key"].shape == (1, 8, 5, 64)


class TestModelLoading:
    """Test model loading behavior without actual downloads."""

    @patch("inference.snapshot_download")
    @patch("inference.Tokenizer.from_file")
    @patch("inference.ort.InferenceSession")
    def test_cache_check_then_download(
        self, mock_session, mock_tokenizer_file, mock_snapshot
    ) -> None:
        """Test that model checks cache before downloading."""
        # First call: cache miss (raises exception)
        # Second call: successful download
        mock_snapshot.side_effect = [
            Exception("Not in cache"),  # local_files_only=True fails
            "/fake/model/path",  # Actual download succeeds
        ]

        mock_tokenizer = Mock()
        mock_tokenizer.token_to_id.return_value = 2
        mock_tokenizer_file.return_value = mock_tokenizer

        # Create mock session with proper structure
        session = Mock()
        output = Mock()
        output.name = "logits"
        session.get_outputs.return_value = [output]

        # Minimal inputs
        inputs = []
        for name in ["input_ids", "attention_mask"]:
            inp = Mock()
            inp.name = name
            inputs.append(inp)

        # Add one past_key_values input
        past_input = Mock()
        past_input.name = "past_key_values.0.key"
        past_input.shape = [1, 8, 0, 64]
        past_input.type = "tensor(float16)"
        inputs.append(past_input)

        session.get_inputs.return_value = inputs
        session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.return_value = session

        with patch("os.path.exists", return_value=False):
            OnnxTextGenerator()

        # Verify cache was checked first, then download happened
        assert mock_snapshot.call_count == 2
        first_call = mock_snapshot.call_args_list[0]
        assert first_call[1].get("local_files_only")

        second_call = mock_snapshot.call_args_list[1]
        assert "local_files_only" not in second_call[1] or not second_call[1].get(
            "local_files_only"
        )

    @patch("inference.snapshot_download")
    @patch("inference.Tokenizer.from_file")
    @patch("inference.ort.InferenceSession")
    def test_use_cached_model(
        self, mock_session, mock_tokenizer_file, mock_snapshot
    ) -> None:
        """Test that cached model is used when available."""
        # First call succeeds (cache hit)
        mock_snapshot.return_value = "/fake/cached/model/path"

        mock_tokenizer = Mock()
        mock_tokenizer.token_to_id.return_value = 2
        mock_tokenizer_file.return_value = mock_tokenizer

        # Create mock session
        session = Mock()
        output = Mock()
        output.name = "logits"
        session.get_outputs.return_value = [output]

        inputs = []
        for name in ["input_ids", "attention_mask"]:
            inp = Mock()
            inp.name = name
            inputs.append(inp)

        past_input = Mock()
        past_input.name = "past_key_values.0.key"
        past_input.shape = [1, 8, 0, 64]
        past_input.type = "tensor(float16)"
        inputs.append(past_input)

        session.get_inputs.return_value = inputs
        session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.return_value = session

        with patch("os.path.exists", return_value=False):
            OnnxTextGenerator()

        # Should only call snapshot_download once (cache hit)
        assert mock_snapshot.call_count == 1
        call_args = mock_snapshot.call_args_list[0]
        assert call_args[1].get("local_files_only")

    @patch("inference.snapshot_download")
    @patch("inference.Tokenizer.from_file")
    @patch("inference.ort.InferenceSession")
    def test_strict_kv_cache_error(
        self, mock_session, mock_tokenizer_file, mock_snapshot
    ) -> None:
        """Test that ValueError is raised if KV cache dimensions cannot be determined."""
        mock_snapshot.return_value = "/fake/model/path"

        # Mock config with missing dimensions
        with patch("builtins.open", side_effect=IOError):
            # Mock inputs without shape info
            session = Mock()
            output = Mock()
            output.name = "logits"
            session.get_outputs.return_value = [output]

            inputs = []
            past_input = Mock()
            past_input.name = "past_key_values.0.key"
            past_input.shape = [
                "batch",
                "heads",
                "seq",
                "dim",
            ]  # Dynamic/unknown shapes
            inputs.append(past_input)

            session.get_inputs.return_value = inputs
            session.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.return_value = session

            with patch("os.path.exists", return_value=False):
                with pytest.raises(
                    ValueError, match="Could not determine KV cache dimensions"
                ):
                    OnnxTextGenerator()

    def test_utf8_split_handling(self, mocked_generator) -> None:
        """Test that generation waits for complete UTF-8 characters."""
        # Mock behavior where:
        # Step 1: Token A -> "Hello \ufffd" (incomplete)
        # Step 2: Token A + B -> "Hello World" (complete)

        # We need to mock tokenizer.decode to return specific strings based on input length
        # mocked_generator.tokenizer is the mock object

        def split_behavior_decode(
            token_ids: list[int], skip_special_tokens: bool = False
        ) -> str:
            if len(token_ids) == 1:
                return "Hello \ufffd"
            return "Hello World"

        mocked_generator.tokenizer.decode = split_behavior_decode

        # Manually verify stream_generate logic with this mock
        # We simulate the loop inputs
        mocked_generator._prepare_generation_inputs = Mock(
            return_value=(
                np.array([[1]]),  # input_ids
                np.array([[1]]),  # attention_mask
                {},  # past_key_values
                [],  # all_token_ids (initially empty from prompt for this test setup?)
                # Actually _prepare returns current_text. Let's say ""
                "",
            )
        )

        # Mock inference step to return dummy data
        mocked_generator._run_inference_step = Mock(
            return_value=(np.array([[[0.1] * 50000]]), {})
        )
        mocked_generator._sample_token = Mock(
            side_effect=[
                np.array([[1]]),  # First token generated
                np.array([[3]]),  # Second token generated (avoid EOS=2)
            ]
        )

        # Run stream_generate for 2 steps
        chunks = []
        for chunk, _ in mocked_generator.stream_generate("prompt", max_new_tokens=2):
            chunks.append(chunk)

        # First step: "Hello \ufffd" -> ends with replacement char -> yield should be skipped (or empty)
        # Second step: "Hello World" -> "Hello World"[0:] -> "Hello World"

        # Wait, if current_text starts as "", and first decode is "Hello \ufffd".
        # Logic: len("Hello \ufffd") > 0. Ends with \ufffd. Condition `not full_text.endswith("\ufffd")` fails.
        # So nothing yielded. current_text remains "".

        # Second step: tokens [1, 2]. decode -> "Hello World".
        # Logic: len("Hello World") > 0. Not ends with \ufffd.
        # new_text_chunk = "Hello World"[0:] = "Hello World".
        # current_text becomes "Hello World".
        # yield "Hello World".

        assert len(chunks) == 1
        assert chunks[0] == "Hello World"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
