import json
from unittest.mock import MagicMock, Mock, mock_open, patch

import onnx
import pytest

from inference import OnnxTextGenerator, bundle_onnx_model


# Define a mock for onnxruntime_genai
@pytest.fixture
def mock_og():
    with patch("inference.og") as mock_og:
        # Mock Config
        mock_og.Config = Mock()

        # Mock Model
        mock_model = Mock()
        mock_og.Model.return_value = mock_model

        # Mock Tokenizer
        mock_tokenizer = Mock()
        mock_og.Tokenizer.return_value = mock_tokenizer

        # Mock GeneratorParams
        mock_params = Mock()
        mock_og.GeneratorParams.return_value = mock_params

        # Mock Generator
        mock_generator = Mock()
        mock_og.Generator.return_value = mock_generator

        yield mock_og


@pytest.fixture
def mock_snapshot_download():
    with patch("inference.snapshot_download") as mock_download:
        mock_download.return_value = "/mock/model/path"
        yield mock_download


class TestOnnxTextGenerator:
    def test_initialization_success(self, mock_og, mock_snapshot_download):
        """Test successful initialization including config check."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch.object(
                OnnxTextGenerator, "_detect_onnx_model", return_value="model.onnx"
            ),
        ):
            generator = OnnxTextGenerator()

            assert generator.model_folder == "/mock/model/path"
            mock_og.Config.assert_called_once_with("/mock/model/path")
            mock_og.Model.assert_called_once()
            mock_og.Tokenizer.assert_called_once()

    def test_initialization_generates_config(self, mock_og, mock_snapshot_download):
        """Test that genai_config is generated if missing."""

        with patch("inference.GenAIConfigGenerator") as mock_gen_class:
            mock_gen_instance = mock_gen_class.return_value
            mock_gen_instance.create_config.return_value = {"mock": "config"}

            # Mocking exists: root genai_config=False initially, then True after generation
            exists_results = {"genai_config.json": False}

            def mock_exists(p):
                # If we've already "created" it, return True
                if "genai_config.json" in p and exists_results.get("genai_config.json"):
                    return True
                # Initial check for config.json or other files
                if "genai_config.json" not in p:
                    return True
                return False

            with (
                patch("os.path.exists", side_effect=mock_exists),
                patch("os.path.getsize", return_value=100),
                patch("builtins.open", mock_open(read_data='{"model_type": "gemma"}')),
                patch.object(
                    OnnxTextGenerator, "_detect_onnx_model", return_value="model.onnx"
                ),
                patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            ):

                def side_effect_with_creation(*args, **kwargs):
                    exists_results["genai_config.json"] = True
                    return {"mock": "config"}

                mock_gen_instance.create_config.side_effect = side_effect_with_creation

                OnnxTextGenerator()

                # Verify create_config was called on the instance
                mock_gen_instance.create_config.assert_called()

    def test_generate_call(self, mock_og, mock_snapshot_download):
        """Test the generate wrapper method."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()

            # Setup mock stream_generate behavior
            with patch.object(generator, "stream_generate") as mock_stream:
                mock_stream.return_value = [
                    ("Hello", {"tokens_generated": 1, "finish_reason": None}),
                    (" World", {"tokens_generated": 2, "finish_reason": "length"}),
                ]

                result = generator.generate("Prompt")

                assert result["generated_text"] == "Hello World"
                assert result["finish_reason"] == "length"
                assert result["tokens_generated"] == 2

    def test_stream_generate(self, mock_og, mock_snapshot_download):
        """Test stream_generate logic."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator_instance = OnnxTextGenerator()

            # Mock Tokenizer behavior
            mock_tokenizer = mock_og.Tokenizer.return_value
            mock_input_tokens = MagicMock()
            # In some OGA versions shape might be a property or a method.
            # Our code uses `input_tokens.shape()[1]`.
            mock_input_tokens.shape.return_value = [1, 5]
            mock_tokenizer.encode_batch.return_value = mock_input_tokens

            mock_stream = Mock()
            mock_tokenizer.create_stream.return_value = mock_stream
            mock_stream.decode.side_effect = ["Hello", " World"]

            # Mock Generator behavior
            mock_gen = mock_og.Generator.return_value
            mock_gen.is_done.side_effect = [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ]
            mock_gen.get_next_tokens.return_value = [123]  # Mock token ID

            # Run stream_generate
            chunks = []
            for chunk, meta in generator_instance.stream_generate(
                "Prompt", max_new_tokens=2
            ):
                chunks.append(chunk)

            assert "Hello" in chunks

            # Verify GeneratorParams were set correctly
            mock_params = mock_og.GeneratorParams.return_value
            mock_params.set_search_options.assert_called()

    def test_beam_search_configuration(self, mock_og, mock_snapshot_download):
        """Test that beam search correctly configures the model."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CUDAExecutionProvider"],
            ),
        ):
            # Test with beam search and cuda
            OnnxTextGenerator(num_beams=3, execution_providers="CUDAExecutionProvider")

            # Verify provider was set
            mock_config = mock_og.Config.return_value
            assert mock_config.clear_providers.called
            mock_config.append_provider.assert_called_with("CUDAExecutionProvider")

    def test_coreml_initialization_no_unsupported_options(
        self, mock_og, mock_snapshot_download
    ):
        """Test that CoreML initialization does not set unsupported 'past_present_share_buffer'."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch(
                "onnxruntime.get_available_providers",
                return_value=["CoreMLExecutionProvider"],
            ),
        ):
            OnnxTextGenerator(execution_providers="CoreMLExecutionProvider")

            # Verify provider was set to CoreML
            mock_config = mock_og.Config.return_value
            mock_config.append_provider.assert_called_with("CoreMLExecutionProvider")

            # Verify set_provider_option was NOT called with past_present_share_buffer
            # Note: set_provider_option might be called for other things, so we check the calls
            for call in mock_config.set_provider_option.call_args_list:
                args, _ = call
                assert "past_present_share_buffer" not in args

    def test_verify_and_fix_config_filename(self, mock_og, mock_snapshot_download):
        """Test that incorrect model filename in config is fixed."""
        mock_config = {"model": {"decoder": {"filename": "wrong_model.onnx"}}}

        # _fix_paths_in_config will use _detect_onnx_model
        with (
            patch(
                "os.path.exists",
                side_effect=lambda p: "wrong_model" in p or "decoder.onnx" in p,
            ),
            patch("os.symlink"),
            patch("shutil.copy2"),
            patch("builtins.open", mock_open()),
            patch("json.load", return_value=mock_config),
            patch("json.dump") as mock_json_dump,
            patch.object(
                OnnxTextGenerator,
                "_detect_onnx_model",
                return_value="onnx/decoder.onnx",
            ),
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()
            generator.model_folder = "/mock/path"
            # Manually trigger the method under test
            generator._verify_and_fix_config_filename(
                "/mock/path/genai_config.json", "onnx/decoder.onnx"
            )

            # Verify it was promoted to root
            assert mock_json_dump.called
            saved_config = mock_json_dump.call_args[0][0]
            assert saved_config["model"]["decoder"]["filename"] == "onnx/decoder.onnx"

    def test_fix_paths_deeply_nested(self, mock_og, mock_snapshot_download):
        """Test recursive path fixing in complex nested config."""
        config = {
            "model": {
                "decoder": {"filename": "text.onnx"},
                "vision": {
                    "filename": "vision.onnx",
                    "adapter_filename": "vision.adapter",
                },
            }
        }

        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()
            generator.model_folder = "/mock/model/path"

            with (
                patch("os.path.exists", return_value=False),
                patch("os.symlink"),
                patch("shutil.copy2"),
                patch.object(
                    generator,
                    "_detect_onnx_model",
                    side_effect=lambda f, v, fallback: f"gpu/{v}",
                ),
            ):
                changed = generator._fix_paths_in_config(config)

                assert changed is True
                assert config["model"]["decoder"]["filename"] == "gpu/text.onnx"
                assert config["model"]["vision"]["filename"] == "gpu/vision.onnx"
                assert (
                    config["model"]["vision"]["adapter_filename"]
                    == "gpu/vision.adapter"
                )

    def test_verify_and_fix_config_saves_on_forced_filename(
        self, mock_og, mock_snapshot_download
    ):
        """Test that genai_config.json is saved if the primary decoder path is forced, even if no other paths change."""
        mock_config = {"model": {"decoder": {"filename": "old_path.onnx"}}}

        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()
            generator.model_folder = "/mock/path"

            # 1. Config has old_path.onnx
            # 2. We detect it should be new/path.onnx
            # 3. new/path.onnx exists (so _fix_paths_in_config returns False)
            # 4. Verify it still saves because of "any_changed = True" from the forced step.

            with (
                patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
                patch("json.load", return_value=mock_config),
                patch("json.dump") as mock_json_dump,
                patch("os.path.exists", return_value=True),
                patch("os.symlink"),
                patch("shutil.copy2"),
                patch.object(generator, "_fix_paths_in_config", return_value=False),
            ):
                generator._verify_and_fix_config_filename(
                    "/mock/path/genai_config.json", "new/path.onnx"
                )

                assert mock_json_dump.called
                saved_config = mock_json_dump.call_args[0][0]
                assert saved_config["model"]["decoder"]["filename"] == "new/path.onnx"

    def test_stream_generate_mutually_exclusive(self, mock_og, mock_snapshot_download):
        """Test that stream_generate raises ValueError if beam search and sampling are both requested."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()

            # Mock Tokenizer behavior for the beginning of stream_generate
            mock_tokenizer = mock_og.Tokenizer.return_value
            mock_input_tokens = MagicMock()
            mock_input_tokens.shape.return_value = [1, 5]
            mock_tokenizer.encode_batch.return_value = mock_input_tokens

            with pytest.raises(ValueError, match="mutually exclusive"):
                # Use list() to exhaust the generator and trigger the logic
                list(generator.stream_generate("Prompt", num_beams=2, do_sample=True))

    def test_detect_onnx_model_priorities(self, mock_og, mock_snapshot_download):
        """Test the priority logic in _detect_onnx_model."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()
            generator.model_folder = "/mock/path"

            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isfile", side_effect=lambda p: "exact.onnx" in p),
            ):
                # 1. Exact path match
                assert (
                    generator._detect_onnx_model("/mock/path", "exact.onnx")
                    == "exact.onnx"
                )

            with (
                patch("os.path.exists", return_value=True),
                patch(
                    "os.path.isfile",
                    side_effect=lambda p: "root.onnx" in p and "subdir" not in p,
                ),
            ):
                # 2. Root filename match
                assert (
                    generator._detect_onnx_model("/mock/path", "subdir/root.onnx")
                    == "root.onnx"
                )

            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isfile", return_value=False),
                patch("os.walk") as mock_walk,
            ):
                # 3. Recursive match
                mock_walk.return_value = [("/mock/path/subdir", [], ["recurse.onnx"])]
                assert (
                    generator._detect_onnx_model("/mock/path", "recurse.onnx")
                    == "subdir/recurse.onnx"
                )

            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isfile", return_value=False),
                patch("os.walk") as mock_walk,
            ):
                # 4. Strict non-default file failure
                mock_walk.return_value = [("/mock/path", [], ["other.onnx"])]
                with pytest.raises(
                    FileNotFoundError,
                    match="Specific ONNX model 'missing.onnx' not found",
                ):
                    generator._detect_onnx_model(
                        "/mock/path", "missing.onnx", fallback=False
                    )

    def test_ensure_tokenizer_files_copy(self, mock_og, mock_snapshot_download):
        """Test that tokenizer files are copied from repo root to model subdir if missing."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch("os.path.exists") as mock_exists,
            patch("shutil.copy2") as mock_copy,
            # Setup detection to return a model in a subdir, setting model_folder to subdir
            patch.object(
                OnnxTextGenerator,
                "_detect_onnx_model",
                return_value="subdir/model.onnx",
            ),
            patch.object(
                OnnxTextGenerator, "_get_model_path", return_value="/mock/path"
            ),
        ):
            # Simulate:
            # 1. target (subdir) tokenizer.json missing
            # 2. source (root) tokenizer.json exists
            def exists_side_effect(path):
                if "subdir" in path and "tokenizer.json" in path:
                    return False
                if "tokenizer.json" in path:  # Root
                    return True
                return True  # Other checks

            mock_exists.side_effect = exists_side_effect

            # Initialization will set model_folder = /mock/path/subdir
            # Then _ensure_tokenizer_files will check subdir, fail, then check repo root, succeed, and copy.
            OnnxTextGenerator()

            # verify copy called
            mock_copy.assert_called()
            # Expect copy from root to subdir
            mock_copy.assert_any_call(
                "/mock/path/tokenizer.json", "/mock/path/subdir/tokenizer.json"
            )

    def test_ensure_genai_config_copy_subdir(self, mock_og, mock_snapshot_download):
        """Test that genai_config.json is copied from subdir if missing in root."""
        with (
            patch("os.path.exists") as mock_exists,
            patch(
                "builtins.open",
                mock_open(
                    read_data='{"model": {"decoder": {"filename": "subdir/model.onnx"}}}'
                ),
            ),
            patch(
                "json.load",
                return_value={"model": {"decoder": {"filename": "subdir/model.onnx"}}},
            ),
            patch("json.dump") as mock_dump,
            patch.object(
                OnnxTextGenerator,
                "_detect_onnx_model",
                return_value="subdir/model.onnx",
            ),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch.object(OnnxTextGenerator, "_fix_paths_in_config"),
        ):
            # Simulate: root config missing, subdir config exists
            def exists_side_effect(path):
                if path.endswith("genai_config.json"):
                    if "onnx" in path:  # subdir
                        return True
                    return False  # root
                return True  # model file exists

            mock_exists.side_effect = exists_side_effect

            OnnxTextGenerator()

            # Verify root config was written (dump called)
            mock_dump.assert_called()

    def test_stream_generate_multimodal_constraint(
        self, mock_og, mock_snapshot_download
    ):
        """Test that beam search is forbidden for multimodal models."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch("builtins.open", mock_open(read_data='{"model": {"vision": {}}}')),
            patch("json.load", return_value={"model": {"vision": {}}}),
            patch("os.path.exists", return_value=True),
        ):
            generator = OnnxTextGenerator()

            with pytest.raises(ValueError, match="multimodal models"):
                # Trigger check by requesting beams > 1
                list(generator.stream_generate("prompt", num_beams=2))

    def test_stream_generate_sampling_params(self, mock_og, mock_snapshot_download):
        """Test correct propagation of sampling parameters."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()

            # Mock tokenizer and generator
            mock_og.Tokenizer.return_value.encode_batch.return_value.shape.return_value = [
                1,
                5,
            ]
            mock_gen = mock_og.Generator.return_value
            # is_done called once in while, once after loop
            mock_gen.is_done.side_effect = [True, True]

            list(
                generator.stream_generate(
                    "prompt", temperature=0.8, top_p=0.95, do_sample=True
                )
            )

            params = mock_og.GeneratorParams.return_value
            # Check set_search_options call
            params.set_search_options.assert_called()
            call_kwargs = params.set_search_options.call_args[1]
            assert call_kwargs["do_sample"] is True
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["top_p"] == 0.95

    def test_detect_onnx_model_fallback_recursive(
        self, mock_og, mock_snapshot_download
    ):
        """Test that _detect_onnx_model falls back to recursive search."""
        with patch.object(OnnxTextGenerator, "_ensure_genai_config"):
            generator = OnnxTextGenerator()

        with (
            patch("os.path.isfile", return_value=False),
            patch("os.walk") as mock_walk,
            patch("os.path.exists", return_value=True),
        ):  # Ensure folder exists check passes
            # Simulate: root/subdir/other.onnx found
            mock_walk.return_value = [
                ("/model", ["subdir"], []),
                ("/model/subdir", [], ["other.onnx"]),
            ]

            # Fallback enabled by default
            result = generator._detect_onnx_model("/model")
            assert result == "subdir/other.onnx"

    def test_detect_onnx_model_integrity_check(self, mock_og, mock_snapshot_download):
        """Test _get_model_path integrity check logic via internal helper usage."""
        with patch.object(OnnxTextGenerator, "_ensure_genai_config"):
            generator = OnnxTextGenerator()

        with (
            patch("os.path.exists") as mock_exists,
            patch("os.walk") as mock_walk,
            patch("os.path.getsize") as mock_getsize,
        ):
            # Configure snapshot_download to return our fake cache path
            mock_snapshot_download.return_value = "/cache"

            # Case 1: Specific onnx file requested, file exists but no .data, size > 100MB -> PASS
            mock_exists.return_value = True  # config exists
            # os.walk for specific file
            mock_walk.return_value = [("/cache", [], ["large.onnx"])]

            def exists_side_effect(path):
                if "config.json" in path:
                    return True
                if ".data" in path:
                    return False
                return True

            mock_exists.side_effect = exists_side_effect

            mock_getsize.return_value = 150 * 1024 * 1024  # 150MB

            path = generator._get_model_path("test/model", onnx_file="large.onnx")

            # If check_integrity passed, snapshot_download is NOT called for download
            # (It is called with local_files_only=True first, which returns the folder)
            # To distinguish, we mock snapshot_download to return "/cache" only if local_files_only=True

            assert path == "/cache"

    def test_stream_generate_runtime_error(self, mock_og, mock_snapshot_download):
        """Test error handling during generation loop."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            generator = OnnxTextGenerator()

            # Fix Mock: tokenizer.encode_batch(...).shape()[1] is accessed
            mock_tokenizer = mock_og.Tokenizer.return_value
            mock_input_tokens = MagicMock()
            mock_input_tokens.shape.return_value = [1, 5]
            mock_tokenizer.encode_batch.return_value = mock_input_tokens

            mock_gen = mock_og.Generator.return_value
            mock_gen.is_done.return_value = False
            mock_gen.generate_next_token.side_effect = RuntimeError("Generation failed")

            with pytest.raises(RuntimeError, match="Generation failed"):
                list(generator.stream_generate("Prompt"))

    def test_fix_paths_failure_logging(self, mock_og, mock_snapshot_download):
        """Test logging when path fixing fails completely."""
        config = {"model": {"decoder": {"filename": "missing.onnx"}}}

        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch("inference.logger") as mock_logger,
        ):
            generator = OnnxTextGenerator()
            generator.model_folder = "/mock/model"

            with (
                patch("os.path.exists", return_value=False),
                patch.object(
                    generator, "_detect_onnx_model", side_effect=FileNotFoundError
                ),
            ):
                changed = generator._fix_paths_in_config(config)

                assert changed is False
                assert mock_logger.error.called
                assert (
                    "Could not find replacement for missing.onnx"
                    in mock_logger.error.call_args[0][0]
                )

    def test_multimodal_beam_search_guard_more_checks(
        self, mock_og, mock_snapshot_download
    ):
        """Test beam search guardrail with speech/embedding models too."""
        for model_type in ["speech", "embedding"]:
            with (
                patch.object(OnnxTextGenerator, "_ensure_genai_config"),
                patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
                patch(
                    "builtins.open",
                    mock_open(read_data=f'{{"model": {{"{model_type}": {{}}}}}}'),
                ),
                patch("json.load", return_value={"model": {model_type: {}}}),
                patch("os.path.exists", return_value=True),
            ):
                generator = OnnxTextGenerator()
                with pytest.raises(ValueError, match="multimodal models"):
                    list(generator.stream_generate("prompt", num_beams=2))

    def test_default_model_id(self, mock_og, mock_snapshot_download):
        """Test that model_id defaults to DEFAULT_MODEL_ID if None."""
        with (
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
            patch.object(
                OnnxTextGenerator, "_get_model_path", return_value="/mock/path"
            ) as mock_get_path,
        ):
            OnnxTextGenerator(model_id=None)
            # Verify the default constant was passed
            mock_get_path.assert_called_with("onnx-community/SmolLM2-135M-ONNX", None)

    def test_init_generic_failure(self, mock_og, mock_snapshot_download):
        """Test that __init__ catches and logs generic exceptions during loading."""
        # Cause exception in model loading block
        mock_og.Config.side_effect = Exception("Load Failed")

        with (
            patch("inference.logger") as mock_logger,
            patch.object(OnnxTextGenerator, "_ensure_genai_config"),
            patch.object(OnnxTextGenerator, "_ensure_tokenizer_files"),
        ):
            with pytest.raises(Exception):
                OnnxTextGenerator()

            assert mock_logger.error.called
            assert "Failed to load model" in mock_logger.error.call_args[0][0]

    def test_ensure_genai_config_exists(self, mock_og, mock_snapshot_download):
        """Test _ensure_genai_config when config already exists."""
        with patch("os.path.exists", return_value=True):
            # Bypass _ensure_genai_config during init
            with patch.object(OnnxTextGenerator, "_ensure_genai_config"):
                generator = OnnxTextGenerator()

            generator.model_folder = "/mock/path"

            # Now test the method logic
            with (
                patch.object(generator, "_create_genai_config") as mock_create,
                patch.object(
                    generator, "_detect_onnx_model", return_value="model.onnx"
                ),
            ):
                # We need to call the real method, which was patched on the class during init.
                # But since we exited the 'with' block, it should be restored.
                generator._ensure_genai_config()

                mock_create.assert_not_called()

    def test_integrity_check_real_data(self, mock_og, mock_snapshot_download):
        """Test integrity check when .data file exists for valid ONNX model."""
        with patch.object(OnnxTextGenerator, "_ensure_genai_config"):
            generator = OnnxTextGenerator()

        with (
            patch("os.path.exists") as mock_exists,
            patch("os.walk") as mock_walk,
            patch("os.path.getsize", return_value=100),
        ):
            # Setup: model.onnx exists, model.onnx.data exists
            def exists_side_effect(p):
                if p.endswith("model.onnx"):
                    return True
                if p.endswith("model.onnx.data"):
                    return True
                if "genai_config.json" in p:
                    return True
                return False

            mock_exists.side_effect = exists_side_effect

            mock_walk.return_value = [("/path", [], ["model.onnx", "model.onnx.data"])]

            # Call private helper that invokes check_integrity
            mock_snapshot_download.return_value = "/path"

            path = generator._get_model_path("repo", "model.onnx")
            assert path == "/path"


class TestBundleOnnxModel:
    def test_bundle_onnx_model_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            bundle_onnx_model("/non/existent/path.onnx")

    def test_bundle_onnx_model_success(self):
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize", return_value=1024),
            patch("onnx.load") as mock_load,
            patch("onnx.save") as mock_save,
            patch("os.remove"),
        ):
            # Scenario: input exists, output does not
            def exists_side_effect(path):
                if path == "/input.onnx":
                    return True
                if path == "/output.onnx":
                    return False
                return False

            mock_exists.side_effect = exists_side_effect

            # Mock graph initializer with external data
            tensor = MagicMock()
            tensor.data_location = onnx.TensorProto.EXTERNAL
            entry = MagicMock()
            entry.key = "location"
            entry.value = "data.bin"
            tensor.external_data = [entry]

            mock_model = MagicMock()
            mock_model.graph.initializer = [tensor]
            mock_load.return_value = mock_model

            result = bundle_onnx_model("/input.onnx", "/output.onnx")

            assert result == "/output.onnx"
            mock_load.assert_called()
            mock_save.assert_called()

    def test_bundle_onnx_model_size_limit(self):
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize") as mock_getsize,
            patch("onnx.load") as mock_load,
            patch("inference.logger") as mock_logger,
        ):

            def exists_side_effect(path):
                if path == "/output.onnx":
                    return False
                return True

            mock_exists.side_effect = exists_side_effect

            # Base size slightly under 2GB, but with external data it goes over
            # 2GB = 2 * 1024^3 = 2147483648
            mock_getsize.side_effect = [1024 * 1024 * 1024, 1024 * 1024 * 1024 + 100]

            # Mock external data
            tensor = MagicMock()
            tensor.data_location = onnx.TensorProto.EXTERNAL
            entry = MagicMock()
            entry.key = "location"
            entry.value = "huge.bin"
            tensor.external_data = [entry]

            mock_model = MagicMock()
            mock_model.graph.initializer = [tensor]
            mock_load.return_value = mock_model

            result = bundle_onnx_model("/input.onnx", "/output.onnx")

            assert result is None
            assert mock_logger.warning.called
            assert "exceeds 2GB limit" in mock_logger.warning.call_args[0][0]

    def test_bundle_onnx_model_exception_cleanup(self):
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize", return_value=100),
            patch("onnx.load", side_effect=Exception("Boom")),
            patch("os.remove") as mock_remove,
        ):
            # 1. exists(input_path) -> True (line 50)
            # 2. exists(output_path) -> False (line 58)
            # ... exception ...
            # 3. exists(output_path) -> True (line 112)
            mock_exists.side_effect = [True, False, True]

            result = bundle_onnx_model("/input.onnx", "/output.onnx")

            assert result is None
            mock_remove.assert_called_with("/output.onnx")

    def test_bundle_onnx_model_default_output_path(self):
        """Test bundling when no output path is specified."""
        with (
            patch("os.path.exists") as mock_exists,
            patch("os.path.getsize", return_value=1024),
            patch("onnx.load") as mock_load,
            patch("onnx.save") as mock_save,
        ):
            # Input exists, output (default) does not
            def exists_side_effect(path):
                if path == "/input.onnx":
                    return True
                return False

            mock_exists.side_effect = exists_side_effect

            # Mock graph
            tensor = MagicMock()
            tensor.data_location = onnx.TensorProto.EXTERNAL
            tensor.external_data = [MagicMock(key="location", value="d.bin")]
            mock_load.return_value.graph.initializer = [tensor]

            result = bundle_onnx_model("/input.onnx")

            assert result == "/input_bundled.onnx"
            mock_save.assert_called()

    def test_bundle_onnx_model_early_return(self):
        """Test early return if output file already exists."""
        with patch("os.path.exists", return_value=True):
            result = bundle_onnx_model("/input.onnx", "/existing.onnx")
            assert result == "/existing.onnx"


class TestOnnxTextGeneratorCoreML:
    @patch("inference.og")
    def test_coreml_bundling_retry(self, mock_og):
        """Test that detecting 'Not a directory' with CoreML triggers bundling."""

        # Setup exceptions
        mock_model_init = mock_og.Model

        # We need to mock Config as well because successful loop uses it
        mock_config = MagicMock()
        mock_og.Config.return_value = mock_config

        # First attempt: raises generic Exception with "Not a directory"
        error = Exception("Error: Not a directory")

        # Second attempt: success
        mock_model_init.side_effect = [error, MagicMock()]

        with (
            patch("inference.OnnxTextGenerator._get_model_path", return_value="/model"),
            patch(
                "inference.OnnxTextGenerator._detect_onnx_model",
                return_value="model.onnx",
            ),
            patch("inference.os.path.exists", return_value=True),
            patch(
                "inference.onnxruntime.get_available_providers",
                return_value=["CoreMLExecutionProvider"],
            ),
            patch(
                "inference.OnnxTextGenerator._ensure_genai_config"
            ) as mock_ensure_config,
            patch("inference.OnnxTextGenerator._ensure_tokenizer_files"),
            patch(
                "inference.bundle_onnx_model",
                return_value="/model/model_coreml_bundled.onnx",
            ) as mock_bundle,
            patch("inference.logger"),
        ):
            OnnxTextGenerator(execution_providers="CoreMLExecutionProvider")

            # Assert bundling was triggered
            mock_bundle.assert_called()
            # Assert config was updated/re-ensured call count
            assert mock_ensure_config.call_count >= 2

            # Check the filename passed to second call
            calls = mock_ensure_config.call_args_list
            assert calls[-1][0][0] == "model_coreml_bundled.onnx"

    @patch("inference.og")
    def test_coreml_bundling_failure_logs_error(self, mock_og):
        """Test that if bundling fails, it logs error and re-raises original exception."""
        mock_og.Model.side_effect = Exception("Not a directory")
        mock_og.Config.return_value = MagicMock()

        with (
            patch("inference.OnnxTextGenerator._get_model_path", return_value="/model"),
            patch(
                "inference.OnnxTextGenerator._detect_onnx_model",
                return_value="model.onnx",
            ),
            patch("inference.os.path.exists", return_value=True),
            patch(
                "inference.onnxruntime.get_available_providers",
                return_value=["CoreMLExecutionProvider"],
            ),
            patch("inference.OnnxTextGenerator._ensure_genai_config"),
            patch("inference.OnnxTextGenerator._ensure_tokenizer_files"),
            patch(
                "inference.bundle_onnx_model", side_effect=Exception("Bundle failed")
            ) as mock_bundle,
            patch("inference.logger") as mock_logger,
        ):
            with pytest.raises(Exception, match="Not a directory"):
                OnnxTextGenerator(execution_providers="CoreMLExecutionProvider")

            mock_bundle.assert_called()
            assert mock_logger.error.called
            # Verify "Bundling failed" was logged
            assert any(
                "Bundling failed" in str(c) for c in mock_logger.error.call_args_list
            )

    @patch("inference.og")
    def test_cuda_optimization_exception(self, mock_og):
        """Test that exceptions during CUDA optimization setting are caught and logged."""
        mock_config = MagicMock()
        mock_og.Config.return_value = mock_config

        # Make set_provider_option raise exception
        mock_config.set_provider_option.side_effect = Exception("Not supported")

        with (
            patch("inference.OnnxTextGenerator._get_model_path", return_value="/model"),
            patch(
                "inference.OnnxTextGenerator._detect_onnx_model",
                return_value="model.onnx",
            ),
            patch("inference.os.path.exists", return_value=True),
            patch(
                "inference.onnxruntime.get_available_providers",
                return_value=["CUDAExecutionProvider"],
            ),
            patch("inference.OnnxTextGenerator._ensure_genai_config"),
            patch("inference.OnnxTextGenerator._ensure_tokenizer_files"),
            patch("inference.logger") as mock_logger,
        ):
            OnnxTextGenerator(execution_providers="CUDAExecutionProvider")

            assert mock_logger.debug.called
            assert (
                "Optional optimizations not supported"
                in mock_logger.debug.call_args[0][0]
            )


class TestStreamGenerateLoop:
    def test_stream_generate_early_termination(self):
        """Test loop termination when generator is done."""
        with (
            patch("inference.og") as mock_og,
            patch("inference.OnnxTextGenerator._ensure_genai_config"),
            patch("inference.OnnxTextGenerator._ensure_tokenizer_files"),
            patch(
                "inference.OnnxTextGenerator._get_model_path", return_value="/mock/path"
            ),
        ):
            generator = OnnxTextGenerator()
            mock_gen = mock_og.Generator.return_value
            mock_tokenizer = mock_og.Tokenizer.return_value
            mock_tokenizer.encode_batch.return_value.shape.return_value = [1, 5]

            # Mock behavior:
            # 1. is_done -> False
            # 2. generate_next_token()
            # 3. is_done -> True (Loop break condition line 360)
            mock_gen.is_done.side_effect = [False, True, True]

            mock_gen.get_next_tokens.return_value = [1]
            mock_tokenizer.create_stream.return_value.decode.return_value = "A"

            chunks = list(generator.stream_generate("Prompt", max_new_tokens=10))

            assert len(chunks) == 1
            assert chunks[0][0] == ""
            assert chunks[0][1]["finish_reason"] == "stop"

    def test_stream_generate_max_tokens_break(self):
        """Test breaking when max_new_tokens is reached."""
        with (
            patch("inference.og") as mock_og,
            patch("inference.OnnxTextGenerator._ensure_genai_config"),
            patch("inference.OnnxTextGenerator._ensure_tokenizer_files"),
            patch(
                "inference.OnnxTextGenerator._get_model_path", return_value="/mock/path"
            ),
        ):
            generator = OnnxTextGenerator()
            mock_gen = mock_og.Generator.return_value
            mock_tokenizer = mock_og.Tokenizer.return_value
            mock_tokenizer.encode_batch.return_value.shape.return_value = [1, 5]

            mock_gen.is_done.return_value = False
            mock_gen.get_next_tokens.return_value = [1]
            mock_tokenizer.create_stream.return_value.decode.side_effect = [
                "A",
                "B",
                "C",
            ]

            # max_new_tokens = 2
            chunks = list(generator.stream_generate("Prompt", max_new_tokens=2))

            assert len(chunks) == 3
            assert chunks[0][0] == "A"
            assert chunks[1][0] == "B"
            assert chunks[2][0] == ""
            assert chunks[2][1]["finish_reason"] == "length"
