import json
from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from config import GenAIConfigGenerator


@pytest.fixture
def config_generator():
    """Fixture to provide a configured GenAIConfigGenerator instance."""
    generator = GenAIConfigGenerator()
    generator.logger = MagicMock()
    # Suppress logging during tests
    generator.logger.setLevel("ERROR")
    return generator


@pytest.fixture
def create_minimal_onnx(tmp_path):
    """Factory fixture to create minimal valid ONNX models for testing."""

    def _create(
        filename,
        hidden_size=1024,
        num_heads=16,
        head_dim=64,
        vocab_size=32000,
        context_length=4096,
        directory=None,
    ):
        target_dir = directory if directory else tmp_path

        # 1. Create Inputs/Outputs
        # Input: input_ids [Batch, Seq]
        input_ids = helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch", "seq"]
        )
        # Output: logits [Batch, Seq, Vocab]
        logits = helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, ["batch", "seq", vocab_size]
        )

        # KV Cache Inputs (for num_key_value_heads detection)
        # Shape: [Batch, NumKVHeads, PastSeq, HeadDim]
        kv_input = helper.make_tensor_value_info(
            "past_key_values.0.key",
            TensorProto.FLOAT,
            ["batch", num_heads, "past_seq", head_dim],
        )

        # 2. Create Initializers (Weights)
        initializers = []

        # Hidden Size (from embed_tokens or norm)
        # model.embed_tokens.weight: [Vocab, Hidden]
        embed_weight = helper.make_tensor(
            "model.embed_tokens.weight",
            TensorProto.FLOAT,
            [vocab_size, hidden_size],
            [1.0] * (vocab_size * hidden_size),  # placeholder values
        )
        initializers.append(embed_weight)

        # Positional Embeddings (for context_length)
        # The code looks for POS_EMBED_CANDIDATES like "position_embeddings"
        pos_embed = helper.make_tensor(
            "model.embed_tokens.position_embeddings",  # Made up name matching candidate
            TensorProto.FLOAT,
            [context_length, hidden_size],
            [0.0],  # optimize size, we only care about dims
        )
        initializers.append(pos_embed)

        # Attention Heads (from q_proj)
        # q_proj.weight: [NumHeads * HeadDim, Hidden]
        q_proj = helper.make_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            TensorProto.FLOAT,
            [num_heads * head_dim, hidden_size],
            [0.1],
        )
        initializers.append(q_proj)

        # 3. Create Graph
        graph = helper.make_graph(
            nodes=[],
            name="test-graph",
            inputs=[input_ids, kv_input],
            outputs=[logits],
            initializer=initializers,
        )

        # 4. Create Model
        model = helper.make_model(graph, producer_name="onnx-inference-test")

        path = target_dir / filename
        # Ensure parent exists if filename contains dirs
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return str(path)

    return _create


def test_context_length_heuristic_safe_limit(config_generator, tmp_path):
    """Test that we pick the smaller value if it's > half the larger value (Safe Limit)."""
    # Config says 128k, ONNX says 132k -> Should pick 128k
    hf_config = {"context_length": 128000}

    with (
        patch.object(config_generator, "_inspect_onnx_model") as mock_inspect,
        patch("os.path.exists", return_value=True),
        patch.object(config_generator, "_validate_params"),
        patch.object(config_generator, "_handle_multimodal_sections"),
        patch.object(config_generator, "_infer_derived_params"),
    ):
        mock_inspect.return_value = {
            "inputs": [],
            "outputs": [],
            "dimensions": {"context_length": 132000},
        }

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))

        assert result["model"]["decoder"]["session_options"]["provider_options"] == []
        assert result["model"]["context_length"] == 128000


def test_context_length_heuristic_sliding_window(config_generator, tmp_path):
    """Test that we pick the larger value if smaller is < half (Sliding Window)."""
    # Config says 4096, ONNX says 131072 -> Should pick 131072
    hf_config = {"context_length": 4096}

    with (
        patch.object(config_generator, "_inspect_onnx_model") as mock_inspect,
        patch("os.path.exists", return_value=True),
        patch.object(config_generator, "_validate_params"),
        patch.object(config_generator, "_handle_multimodal_sections"),
        patch.object(config_generator, "_infer_derived_params"),
    ):
        mock_inspect.return_value = {
            "inputs": [],
            "outputs": [],
            "dimensions": {"context_length": 131072},
        }

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))
        assert result["model"]["context_length"] == 131072


def test_infer_derived_params_critical_missing(config_generator):
    """Test that critical errors are raised when params cannot be derived."""
    params = {"hidden_size": 1024}  # Missing head_dim AND num_heads
    with pytest.raises(ValueError, match="num_attention_heads"):
        config_generator._infer_derived_params(params)


def test_infer_derived_params_success(config_generator):
    """Test successful derivation of params."""
    params = {
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "context_length": 1024,
        "num_hidden_layers": 2,
    }
    config_generator._infer_derived_params(params)
    assert params["head_dim"] == 64  # 1024 / 16
    assert params["num_key_value_heads"] == 16  # Default to num_heads


def test_map_io_names_kv_cache(config_generator):
    """Test mapping of KV cache IO names."""
    inputs = ["input_ids", "past_key_values.0.key", "past_key_values.0.value"]
    mapping = config_generator._map_io_names(inputs, "inputs")
    assert mapping["input_ids"] == "input_ids"
    assert mapping["past_key_names"] == "past_key_values.%d.key"
    assert mapping["past_value_names"] == "past_key_values.%d.value"


def test_handle_multimodal_sections(config_generator, tmp_path):
    """Test detection of multimodal components."""
    params = {}
    model_folder = str(tmp_path)

    # Mock os.walk to return a vision model file
    with (
        patch("os.walk") as mock_walk,
        patch.object(config_generator, "_inspect_onnx_model") as mock_inspect,
    ):
        mock_walk.return_value = [
            (model_folder, [], ["vision_encoder.onnx", "vision_processor.json"])
        ]

        mock_inspect.return_value = {
            "inputs": ["pixel_values"],
            "outputs": ["image_features"],
            "dimensions": {},
        }

        config_generator._handle_multimodal_sections(params, model_folder)

        assert "vision" in params
        assert params["vision"]["filename"] == "vision_encoder.onnx"
        assert params["vision"]["config_filename"] == "vision_processor.json"
        assert params["vision"]["inputs"]["pixel_values"] == "pixel_values"


def test_validate_params_missing(config_generator):
    """Test that missing required parameters raises ValueError."""
    params = {"vocab_size": 100}  # Missing hidden_size etc.
    with pytest.raises(ValueError):
        config_generator._validate_params(params)


def test_context_length_aliases(config_generator):
    """Test that context_length can be derived from various aliases."""
    aliases = [
        ("max_position_embeddings", 2048),
        ("n_ctx", 1024),
        ("seq_length", 512),
        ("max_sequence_length", 4096),
    ]

    base_params = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_hidden_layers": 2,
    }

    for alias, val in aliases:
        params = base_params.copy()
        params[alias] = val
        config_generator._infer_derived_params(params)
        assert params["context_length"] == val, f"Failed to infer from {alias}"


def test_map_io_names_outputs(config_generator):
    """Test mapping of output names including present/KV cache."""
    outputs = ["logits", "present.0.key", "present.0.value"]
    mapping = config_generator._map_io_names(outputs, "outputs")
    assert mapping["present_key_names"] == "present.%d.key"
    assert mapping["present_value_names"] == "present.%d.value"


def test_pad_token_id_fallback(config_generator):
    """Test inference of pad_token_id from eos_token_id."""
    base_params = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 16,
        "context_length": 128,
    }

    # Case 1: EOS is int
    params = base_params.copy()
    params["eos_token_id"] = 2
    config_generator._infer_derived_params(params)
    assert params["pad_token_id"] == 2

    # Case 2: EOS is list
    params = base_params.copy()
    params["eos_token_id"] = [2, 3]
    config_generator._infer_derived_params(params)
    assert params["pad_token_id"] == 2


def test_pad_precedence_tokenizer_wins(config_generator, tmp_path):
    """Verify that Tokenizer Pad ID overrides HF Early Imputation."""
    hf_config = {
        "model_type": "test",
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "eos_token_id": [100, 101],  # Early logic would pick 100
        "context_length": 512,
    }

    with (
        patch.object(config_generator, "_load_tokenizer_data") as mock_tok,
        patch.object(config_generator, "_inspect_onnx_model") as mock_inspect,
        patch("os.path.exists", return_value=True),
    ):
        mock_tok.return_value = ({"pad_token_id": 999}, {})
        mock_inspect.return_value = {
            "inputs": [],
            "outputs": [],
            "dimensions": {
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "eos_token_id": 100,
                "context_length": 512,
            },
        }

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))
        assert result["model"]["pad_token_id"] == 999


def test_vocab_size_precedence_onnx_wins(config_generator, tmp_path):
    """Verify Vocab Size Precedence: HF < Tokenizer < ONNX."""
    hf_config = {
        "model_type": "test",
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 100,
        "eos_token_id": 1,
        "context_length": 512,
    }

    with (
        patch.object(config_generator, "_load_tokenizer_data") as mock_tok,
        patch.object(config_generator, "_inspect_onnx_model") as mock_inspect,
        patch("os.path.exists", return_value=True),
    ):
        mock_tok.return_value = ({"vocab_size": 200}, {})
        mock_inspect.return_value = {
            "inputs": [],
            "outputs": [],
            "dimensions": {
                "vocab_size": 300,
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "eos_token_id": 1,
                "context_length": 512,
            },
        }

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))
        assert result["model"]["vocab_size"] == 300


def test_eos_list_handling_and_pad_imputation(config_generator, tmp_path):
    """Test scenario where EOS is a list and Pad is derived from the first element."""
    hf_config = {
        "model_type": "llama",
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "eos_token_id": [128001, 128008, 128009],
        "num_hidden_layers": 16,
        # No pad_token_id explicitly set
    }

    # Mock ONNX inspection
    with patch.object(config_generator, "_inspect_onnx_model") as mock_inspect:
        mock_inspect.return_value = {
            "inputs": [
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values.0.key",
            ],
            "outputs": ["logits"],
            "dimensions": {
                "hidden_size": 2048,
                "num_attention_heads": 32,
                "num_hidden_layers": 16,
                "head_dim": 64,
                "context_length": 131072,
            },
        }

        # Create minimal model file to pass existence check
        (tmp_path / "model.onnx").write_text("placeholder")

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))

        model = result["model"]
        assert model["type"] == "llama"
        assert model["context_length"] == 131072
        assert model["eos_token_id"] == [128001, 128008, 128009]
        # Check pad_token_id derivation from first EOS
        assert model["pad_token_id"] == 128001


def test_head_size_alias_and_decoder_mapping(config_generator, tmp_path):
    """Test scenario handling head_size alias and ensuring it maps to decoder config."""
    hf_config = {
        "model_type": "gemma",
        "hidden_size": 2048,
        "num_attention_heads": 8,
        "vocab_size": 256000,
        "num_hidden_layers": 18,
        "head_dim": 256,  # Config might have it, or it comes from ONNX
        "eos_token_id": 1,
    }

    with patch.object(config_generator, "_inspect_onnx_model") as mock_inspect:
        mock_inspect.return_value = {
            "inputs": ["input_ids"],
            "outputs": ["logits"],
            "dimensions": {
                "hidden_size": 2048,
                "num_attention_heads": 8,
                "num_hidden_layers": 18,
                "head_dim": 256,
                "context_length": 8192,
            },
        }

        (tmp_path / "model.onnx").write_text("placeholder")

        result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))

        decoder = result["model"]["decoder"]
        assert "head_size" in decoder
        assert decoder["head_size"] == 256
        assert result["model"]["type"] == "gemma"


def test_multimodal_file_scanning_and_config_generation(config_generator, tmp_path):
    """Test scanning for multimodal components (vision, speech, embedding) and their config generation."""
    hf_config = {
        "model_type": "multimodal_test",
        "hidden_size": 3072,
        "num_attention_heads": 24,
        "vocab_size": 200064,
        "num_hidden_layers": 32,
        "eos_token_id": 199999,
    }

    # Create placeholder files with generic names
    (tmp_path / "gpu").mkdir()
    files = [
        "gpu/text-model.onnx",
        "gpu/vision-component.onnx",
        "gpu/vision_processor.json",
        "gpu/speech-component.onnx",
        "gpu/speech_processor.json",
        "gpu/embedding-component.onnx",
    ]
    for f in files:
        (tmp_path / f).write_text("placeholder")

    # Mock inspect to return different things for different files
    def side_effect(path):
        if "vision" in path:
            return {
                "inputs": ["pixel_values"],
                "outputs": ["image_features"],
                "dimensions": {},
            }
        elif "speech" in path:
            return {
                "inputs": ["audio_embeds"],
                "outputs": ["audio_features"],
                "dimensions": {},
            }
        elif "embedding" in path:
            return {
                "inputs": ["input_ids", "image_features", "audio_features"],
                "outputs": ["inputs_embeds"],
                "dimensions": {},
            }
        else:  # Text model
            return {
                "inputs": ["inputs_embeds"],
                "outputs": ["logits"],
                "dimensions": {
                    "hidden_size": 3072,
                    "num_attention_heads": 24,
                    "num_hidden_layers": 32,
                    "head_dim": 128,
                    "context_length": 131072,
                },
            }

    with patch.object(config_generator, "_inspect_onnx_model", side_effect=side_effect):
        result = config_generator.create_config(
            hf_config, "gpu/text-model.onnx", str(tmp_path)
        )

        model = result["model"]
        assert model["type"] == "multimodal_test"

        # Check Vision
        assert "vision" in model
        assert model["vision"]["filename"] == "gpu/vision-component.onnx"
        assert model["vision"]["config_filename"] == "vision_processor.json"

        # Check Speech
        assert "speech" in model
        assert model["speech"]["filename"] == "gpu/speech-component.onnx"

        # Check Embedding
        assert "embedding" in model
        assert model["embedding"]["filename"] == "gpu/embedding-component.onnx"


def test_load_generation_config_max_length(config_generator, tmp_path):
    """Test that generation_config.json overrides context_length."""
    gen_config = {"max_length": 8192, "pad_token_id": 99}
    with open(tmp_path / "generation_config.json", "w") as f:
        json.dump(gen_config, f)

    params = config_generator._load_generation_config(str(tmp_path))
    assert params["context_length"] == 8192
    assert params["pad_token_id"] == 99


def test_tokenizer_config_string_resolution(config_generator, tmp_path):
    """Test resolving string keys in tokenizer_config.json using vocab map."""
    # 1. Create tokenizer.json (Vocab Source)
    with open(tmp_path / "tokenizer.json", "w") as f:
        json.dump({"model": {"vocab": {"<unk>": 0, "<pad>": 1}}}, f)

    # 2. Create tokenizer_config.json (String References)
    tok_cfg = {
        "unk_token": "<unk>",  # String
        "pad_token": {"content": "<pad>"},  # Dict
    }
    with open(tmp_path / "tokenizer_config.json", "w") as f:
        json.dump(tok_cfg, f)

    params, _ = config_generator._load_tokenizer_data(str(tmp_path))

    assert params["unk_token_id"] == 0
    assert params["pad_token_id"] == 1


def test_flatten_text_config(config_generator, tmp_path):
    """Test flattening of nested text_config."""
    params = {
        "model_type": "gemma3",
        "text_config": {"vocab_size": 100, "hidden_size": 64},
    }
    # NOTE: The logic is inside create_config. Let's run create_config with a dummy HF config.
    with (
        patch.object(config_generator, "_load_generation_config", return_value={}),
        patch.object(config_generator, "_load_tokenizer_data", return_value=({}, {})),
        patch.object(
            config_generator,
            "_inspect_onnx_model",
            return_value={"inputs": [], "outputs": [], "dimensions": {}},
        ),
    ):
        # Create minimal model file to pass existence check
        (tmp_path / "model.onnx").write_text("placeholder")

        # We need valid params to pass validation later, so inject them
        res = config_generator.create_config(
            params,
            "model.onnx",
            str(tmp_path),
            num_attention_heads=4,
            num_hidden_layers=2,
            eos_token_id=1,
            head_dim=16,
            context_length=1024,
        )

        # Check if vocab_size was pulled up
        assert res["model"]["vocab_size"] == 100
        # Check if model_type was preserved
        assert res["model"]["type"] == "gemma3"


def test_added_tokens_loading(config_generator, tmp_path):
    """Test loading added_tokens.json."""
    with open(tmp_path / "added_tokens.json", "w") as f:
        json.dump({"<extra_0>": 1000}, f)

    params, vocab = config_generator._load_tokenizer_data(str(tmp_path))
    assert params["vocab_size"] == 1001  # 1000 + 1
    assert "<extra_0>" in vocab


def test_derived_param_aliases(config_generator):
    """Test that n_head, n_key_value_heads, etc. are correctly aliased."""
    params = {
        # Aliases
        "n_head": 8,
        "n_key_value_heads": 4,
        "head_size": 32,
        # Required to pass validation
        "hidden_size": 256,
        "context_length": 1024,
    }

    config_generator._infer_derived_params(params)

    assert params["num_attention_heads"] == 8
    assert params["num_key_value_heads"] == 4
    assert params["head_dim"] == 32

    # Test num_kv_heads alias
    params2 = {
        "num_attention_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 32,
        "context_length": 1024,
    }
    config_generator._infer_derived_params(params2)
    assert params2["num_key_value_heads"] == 2


def test_inspect_onnx_failure(config_generator):
    """Test inspection failure handling."""
    with patch("onnx.load", side_effect=Exception("Corrupt")):
        info = config_generator._inspect_onnx_model("bad.onnx")
        assert info["dimensions"] == {}


def test_inspect_simple_model(config_generator, create_minimal_onnx, tmp_path):
    """Verify that _inspect_onnx_model correctly deduces dimensions from a real graph."""
    create_minimal_onnx(
        "model.onnx",
        hidden_size=512,
        num_heads=8,
        head_dim=64,
        vocab_size=1000,
        context_length=4096,  # Must be >= 4096 to be detected as context_length
    )

    info = config_generator._inspect_onnx_model(str(tmp_path / "model.onnx"))
    dims = info["dimensions"]

    assert dims["hidden_size"] == 512
    assert dims["num_attention_heads"] == 8
    assert dims["vocab_size"] == 1000
    assert dims["context_length"] == 4096
    assert dims["head_dim"] == 64


def test_create_config_precedence_onnx_wins(
    config_generator, create_minimal_onnx, tmp_path
):
    """Test that ONNX dimensions override HF config when appropriate."""
    # Use small constants to avoid slow model creation
    kwargs = {"vocab_size": 100, "hidden_size": 16, "num_heads": 4, "head_dim": 4}

    create_minimal_onnx("model.onnx", context_length=8192, **kwargs)

    hf_config = {
        "model_type": "llama",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "vocab_size": 100,
        "context_length": 4096,  # HF says 4k
        "eos_token_id": 2,
    }

    result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))

    # ONNX (8192) > HF (4096), ratio < 2 -> no override (safe limit)
    assert result["model"]["context_length"] == 4096

    # Sliding window case: ONNX >> HF
    create_minimal_onnx("window.onnx", context_length=131072, **kwargs)
    result_window = config_generator.create_config(
        hf_config, "window.onnx", str(tmp_path)
    )
    assert result_window["model"]["context_length"] == 131072


def test_tokenizer_loading(config_generator, create_minimal_onnx, tmp_path):
    """Test recursive loading of tokenizer files."""
    # Create tokenizer.json
    tok_data = {
        "model": {"vocab": {"<unk>": 0, "<s>": 1, "</s>": 2}},
        "added_tokens": [{"id": 3, "content": "<pad>"}],
    }
    with open(tmp_path / "tokenizer.json", "w") as f:
        json.dump(tok_data, f)

    # Create empty minimal ONNX to pass inspection.
    # Set vocab_size=4 to match tokenizer so inspection doesn't fight it.
    create_minimal_onnx(
        "model.onnx", vocab_size=4, hidden_size=64, num_heads=4, head_dim=16
    )

    hf_config = {"hidden_size": 64, "num_attention_heads": 4, "model_type": "llama"}

    # Only tokenizer data provides eos/pad ids here (implicitly or explicitly?)
    # config.py _load_tokenizer_data populates params
    result = config_generator.create_config(hf_config, "model.onnx", str(tmp_path))

    # Vocab size should be max id + 1 = 4
    assert result["model"]["vocab_size"] == 4


def test_handle_multimodal_real_file(config_generator, tmp_path):
    """Test multimodal section scanning with actual files."""
    # Create vision/model.onnx
    vision_dir = tmp_path / "vision"
    vision_dir.mkdir()

    # Create a basic ONNX there
    graph = helper.make_graph(
        [],
        "vision",
        [
            helper.make_tensor_value_info(
                "pixel_values", TensorProto.FLOAT, [1, 3, 224, 224]
            )
        ],
        [helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, 768])],
        [],
    )
    model = helper.make_model(graph)
    onnx.save(model, str(vision_dir / "vision_tower.onnx"))

    # Create config
    with open(tmp_path / "vision_config.json", "w") as f:
        json.dump({}, f)

    params = {}
    config_generator._handle_multimodal_sections(params, str(tmp_path))

    assert "vision" in params
    assert params["vision"]["filename"] == "vision/vision_tower.onnx"
    assert params["vision"]["config_filename"] == "vision_config.json"
