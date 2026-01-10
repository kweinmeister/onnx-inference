import json
import logging
import os
from typing import Any, Dict, List, Tuple

import onnx


class GenAIConfigGenerator:
    """Generates genai_config.json configurations for ONNX Runtime GenAI models."""

    # Constants
    MODEL_PARAM_ALLOWLIST = {
        "type",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "decoder",
        "vision",
        "speech",
        "embedding",
        "context_length",
    }

    POS_EMBED_CANDIDATES = [
        "cos_cache",
        "sin_cache",
        "position_embeddings",
        "pos_embed",
        "position_ids",
    ]

    IO_CANDIDATES = {
        "inputs": [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_features",
            "audio_features",
            "image_sizes",
            "audio_sizes",
            "audio_embeds",
            "inputs_embeds",
            "audio_projection_mode",
            "position_ids",
        ],
        "outputs": ["logits", "image_features", "audio_features", "inputs_embeds"],
    }

    SPECIAL_TOKENS_MAP = {
        "eos_token_id": [
            "<eos>",
            "<end_of_turn>",
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
        ],
        "bos_token_id": ["<bos>", "<s>", "<|begin_of_text|>"],
        "pad_token_id": ["<pad>", "<|padding|>"],
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_config(
        self,
        hf_config: Dict[str, Any],
        model_filename: str,
        model_folder: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Creates the genai_config.json structure with strict precedence.

        Precedence Order (Lowest to Highest):
        1. Hugging Face Config (Baseline metadata)
        2. Generation Config (Length, Token IDs)
        3. Tokenizer Data (Vocab, Special Token IDs)
        4. ONNX Model (Structure: Inputs, Dimensions)
        5. Runtime Overrides (Explicit **kwargs)
        """
        self.logger.info(f"Starting Config Generation for {model_filename}")

        # 1. Start with HF Config (Metadata Baseline)
        final_params = hf_config.copy()

        # 1a. Early Pad Imputation (from Base EOS)
        # Fixes Llama 3 which has EOS list in base config but Tokenizer overrides it with single EOT.
        if final_params.get("pad_token_id") is None and "eos_token_id" in final_params:
            eos = final_params["eos_token_id"]
            if isinstance(eos, list) and len(eos) > 0:
                final_params["pad_token_id"] = eos[0]
            elif isinstance(eos, int):
                final_params["pad_token_id"] = eos

        # Flatten composite configs (e.g. text_config)
        if "text_config" in final_params:
            original_type = final_params.get("model_type")
            final_params.update(final_params.pop("text_config"))
            if original_type:
                # Restore top-level type if needed (e.g. gemma3)
                final_params["model_type"] = original_type

        # 2. Update from Generation Config (Length, Tokens)
        gen_params = self._load_generation_config(model_folder)
        if gen_params:
            self.logger.debug(
                f"Applying Generation Config overrides: {list(gen_params.keys())}"
            )
            final_params.update(gen_params)

        # 3. Update from Tokenizer Data (Vocab, Special Tokens)
        # Includes tokenizer.json, added_tokens.json, tokenizer_config.json
        tok_params, vocab_map = self._load_tokenizer_data(model_folder)
        if tok_params:
            self.logger.debug(
                f"Applying Tokenizer Data overrides: {list(tok_params.keys())}"
            )
            final_params.update(tok_params)

        # 3a. Vocabulary Fallback Scan (Low Priority)
        # Only guess tokens from vocab if they are still missing after HF and Tokenizer Configs
        if vocab_map:
            for param_key, candidates in self.SPECIAL_TOKENS_MAP.items():
                if param_key not in final_params:
                    for c in candidates:
                        if c in vocab_map:
                            self.logger.debug(
                                f"Scanning vocab found {param_key} -> {c} ({vocab_map[c]})"
                            )
                            final_params[param_key] = vocab_map[c]
                            break

        # 3b. Early Normalization (before ONNX inspection)
        # Mapping aliases to context_length so min() logic works in Step 4
        if "context_length" not in final_params:
            for alias in [
                "max_position_embeddings",
                "n_ctx",
                "seq_length",
                "max_sequence_length",
            ]:
                if alias in final_params:
                    final_params["context_length"] = final_params[alias]
                    break

        # 4. Update from Primary ONNX Model (Structural Source of Truth)
        onnx_path = os.path.join(model_folder, model_filename)
        onnx_info: Dict[str, Any] = {"inputs": [], "outputs": [], "dimensions": {}}

        if os.path.exists(onnx_path):
            onnx_info = self._inspect_onnx_model(onnx_path)
            # Override Dimensions
            for k, onnx_v in onnx_info.get("dimensions", {}).items():
                if k == "context_length" and k in final_params:
                    old_v = final_params[k]

                    # Ratio-based heuristic for context_length
                    smaller = min(old_v, onnx_v)
                    larger = max(old_v, onnx_v)

                    # If mismatch, use larger if difference is huge (sliding window), else smaller
                    if smaller < larger / 2:
                        final_params[k] = larger
                        self.logger.debug(
                            f"Preferring ONNX capacity: Config={old_v}, ONNX={onnx_v} -> {larger}"
                        )
                    else:
                        final_params[k] = smaller
                        self.logger.debug(
                            f"Preferring safe limit: Config={old_v}, ONNX={onnx_v} -> {smaller}"
                        )
                else:
                    self.logger.debug(
                        f"Overriding {k}: Config={final_params.get(k)} -> ONNX={onnx_v}"
                    )
                    final_params[k] = onnx_v
        else:
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

        # 5. Runtime Overrides
        if kwargs:
            final_params.update(kwargs)

        # 6. Infer Derived Params (Strict Logic)
        self._infer_derived_params(final_params)

        # 7. Multimodal Scan & Config
        self._handle_multimodal_sections(final_params, model_folder)

        # 8. Validate Critical Parameters
        self._validate_params(final_params)

        # 9. Build Final Config
        # Try to detect model_type if missing or generic
        genai_type = final_params.get("model_type", "model")
        if genai_type == "model":
            # Guess from full path (frequently contains model name in HF cache)
            path_lower = os.path.join(model_folder, model_filename).lower()
            # Simplified priority rules: (Pattern, Type)
            detect_rules = [
                ("gemma-3-270m", "gemma3_text"),
                ("gemma-3-1b", "gemma3_text"),
                ("gemma-3", "gemma3"),
                ("gemma-2", "gemma2"),
                ("gemma", "gemma"),
                ("phi-4-mm", "phi4mm"),
                ("phi-3-v", "phi3v"),
                ("phi-3", "phi3"),
                ("phi3", "phi3"),
                ("phi", "phi"),
                ("llama", "llama"),
                ("qwen3", "qwen3"),
                ("qwen2", "qwen2"),
            ]

            for kw, mtype in detect_rules:
                if kw in path_lower:
                    genai_type = mtype
                    break

            final_params["model_type"] = genai_type
            self.logger.info(f"Guessed model_type from path: {genai_type}")

        decoder_config = {
            "session_options": final_params.get(
                "session_options",
                {"log_id": "onnxruntime-genai", "provider_options": []},
            ),
            "filename": model_filename,
            "inputs": self._map_io_names(onnx_info["inputs"], "inputs"),
            "outputs": self._map_io_names(onnx_info["outputs"], "outputs"),
        }

        # Add Architecture Params to Decoder
        for k in [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
        ]:
            if k in final_params:
                # Some models use head_size alias
                target_k = "head_size" if k == "head_dim" else k
                decoder_config[target_k] = final_params[k]

        final_params["decoder"] = decoder_config
        final_params["type"] = genai_type

        return {"model": self._filter_model_params(final_params)}

    def _handle_multimodal_sections(self, params, model_folder):
        """Scans for vision, speech, and embedding models."""
        for section in ["vision", "speech", "embedding"]:
            # 1. Find the ONNX model file
            candidates = []
            for root, _, files in os.walk(model_folder):
                for f in files:
                    if section in f and f.endswith(".onnx"):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, model_folder)
                        candidates.append(rel_path)

            if not candidates:
                continue

            f_path = candidates[0]
            full_path = os.path.join(model_folder, f_path)
            f_info = self._inspect_onnx_model(full_path)

            section_config = {
                "filename": f_path,
                "inputs": self._map_io_names(f_info["inputs"], "inputs"),
                "outputs": self._map_io_names(f_info["outputs"], "outputs"),
            }

            # 2. Find Config Filename (e.g. vision_processor.json)
            config_patterns = [
                f"{section}_processor.json",
                "preprocessor_config.json",
                f"{section}_config.json",
            ]

            for root, _, files in os.walk(model_folder):
                for f in files:
                    if f in config_patterns:
                        section_config["config_filename"] = f
                        break
                if "config_filename" in section_config:
                    break

            # 3. Find Adapter Filename (e.g. model.onnx_adapter)
            # Use basename for finding files relative to their own directories or root
            base_filename = os.path.basename(f_path)
            adapter_patterns = [f"{base_filename}_adapter", f"{base_filename}.adapter"]

            for root, _, files in os.walk(model_folder):
                for f in files:
                    # Check for direct pattern match or section-based match
                    if f in adapter_patterns or (
                        section in f.lower() and "_adapter" in f.lower()
                    ):
                        # We want the path relative to model_folder
                        full_f = os.path.join(root, f)
                        section_config["adapter_filename"] = os.path.relpath(
                            full_f, model_folder
                        )
                        break
                if "adapter_filename" in section_config:
                    break

            params[section] = section_config

    HIDDEN_SIZE_CANDIDATES = [
        "model.embed_tokens.weight",
        "final_layernorm.weight",
        "model.norm.weight",
        "input_layernorm.weight",
    ]

    def _inspect_onnx_model(self, model_path: str) -> Dict[str, Any]:
        """Inspects ONNX model graph for Inputs, Outputs, and Dimensions."""
        self.logger.info(f"Inspecting ONNX model: {model_path}")
        try:
            model = onnx.load(model_path, load_external_data=False)
            graph = model.graph

            initializers = set(i.name for i in graph.initializer)
            inputs = [i.name for i in graph.input if i.name not in initializers]
            outputs = [o.name for o in graph.output]

            dims = {}
            # 1. Deduce vocab_size from logits if possible
            for out in graph.output:
                if "logits" in out.name.lower():
                    shape = self._get_tensor_shape(out)
                    if len(shape) == 3 and isinstance(shape[-1], int):
                        dims["vocab_size"] = shape[-1]

            # 2. Key Value Dimensions
            for inp in graph.input:
                if "past_key_values" in inp.name and "key" in inp.name:
                    shape = self._get_tensor_shape(inp)
                    # Expected: [Batch, NumKVHeads, PastSeqLen, HeadDim]
                    if (
                        len(shape) == 4
                        and isinstance(shape[1], int)
                        and isinstance(shape[3], int)
                    ):
                        dims["num_key_value_heads"] = shape[1]
                        dims["head_dim"] = shape[3]
                        break  # Found it

            # 3. Hidden Size
            for init in graph.initializer:
                if any(h in init.name for h in self.HIDDEN_SIZE_CANDIDATES):
                    # Layernorm weights are (Hidden,)
                    # Embedding weights are (Vocab, Hidden)
                    # Linear weights are (Out, Hidden)
                    if len(init.dims) == 1:
                        dims["hidden_size"] = init.dims[0]
                        break
                    elif len(init.dims) == 2:
                        dims["hidden_size"] = min(init.dims)
                        if "layernorm" in init.name or "norm" in init.name:
                            break

            # 4. Layer Count
            layer_indices = set()
            for inp in graph.input:
                if "past_key_values" in inp.name:
                    parts = inp.name.split(".")
                    for part in parts:
                        if part.isdigit():
                            layer_indices.add(int(part))
            if layer_indices:
                dims["num_hidden_layers"] = max(layer_indices) + 1

            # 5. Attention Heads (From q_proj output dim)
            if "head_dim" in dims:
                for init in graph.initializer:
                    if "q_proj" in init.name:
                        out_dim = init.dims[0]
                        if "Q4" in init.name:
                            if out_dim % dims["head_dim"] == 0:
                                dims["num_attention_heads"] = (
                                    out_dim // dims["head_dim"]
                                )
                            elif (out_dim * 2) % dims["head_dim"] == 0:
                                dims["num_attention_heads"] = (out_dim * 2) // dims[
                                    "head_dim"
                                ]
                        else:
                            if out_dim % dims["head_dim"] == 0:
                                dims["num_attention_heads"] = (
                                    out_dim // dims["head_dim"]
                                )

                        if "num_attention_heads" in dims:
                            break

            # 6. Context Length (Architectural Imputation)
            for init in graph.initializer:
                if any(c in init.name for c in self.POS_EMBED_CANDIDATES):
                    if len(init.dims) == 2:
                        max_dim = max(init.dims)
                        if max_dim >= 4096:
                            dims["context_length"] = max_dim
                            break

            return {
                "inputs": inputs,
                "outputs": outputs,
                "dimensions": dims,
                "initializers": list(initializers),
            }
        except Exception as e:
            self.logger.warning(f"Failed to inspect ONNX model: {e}")
            return {"inputs": [], "outputs": [], "dimensions": {}}

    def _get_tensor_shape(self, value_info):
        shape = []
        if value_info.type.tensor_type.HasField("shape"):
            for d in value_info.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(d.dim_value)
                else:
                    shape.append("dynamic")
        return shape

    def _load_tokenizer_data(
        self, model_folder
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Reads tokenizer.json, added_tokens.json, and tokenizer_config.json.
        Returns (params, vocab_map).
        """
        params = {}
        vocab_map = {}  # content -> id

        # 1. tokenizer.json (Source of Truth for IDs)
        tok_path = os.path.join(model_folder, "tokenizer.json")
        if os.path.exists(tok_path):
            try:
                with open(tok_path, "r") as f:
                    data = json.load(f)

                # Vocab size
                vocab_obj = data.get("model", {}).get("vocab", {})
                if vocab_obj:
                    params["vocab_size"] = len(vocab_obj)
                    vocab_map.update(vocab_obj)

                # Added tokens
                added = data.get("added_tokens", [])
                for t in added:
                    vocab_map[t["content"]] = t["id"]

                # NOTE: Fallback scan moved to create_config to avoid overriding explicit HF config

            except Exception as e:
                self.logger.warning(f"Tokenizer check failed: {e}")

        # 2. added_tokens.json (Supplemental Vocab)
        added_path = os.path.join(model_folder, "added_tokens.json")
        if os.path.exists(added_path):
            try:
                with open(added_path, "r") as f:
                    added_data = json.load(f)
                    if isinstance(added_data, dict):
                        vocab_map.update(added_data)
                        if vocab_map:
                            params["vocab_size"] = max(vocab_map.values()) + 1
            except Exception as e:
                self.logger.debug(f"Failed to read added_tokens.json: {e}")

        # 3. tokenizer_config.json (Map special tokens to IDs)
        tok_cfg_path = os.path.join(model_folder, "tokenizer_config.json")
        if os.path.exists(tok_cfg_path):
            try:
                with open(tok_cfg_path, "r") as f:
                    data = json.load(f)

                # Dynamic Lookup
                for type_key in ["bos_token", "eos_token", "pad_token", "unk_token"]:
                    val = data.get(type_key)
                    id_key = f"{type_key}_id"

                    # A: Explicit ID in config
                    if id_key in data and data[id_key] is not None:
                        params[id_key] = data[id_key]
                        continue

                    # B: Resolve string to ID from accumualted vocab_map
                    if isinstance(val, str):
                        # Check explicit vocab
                        if val in vocab_map:
                            params[id_key] = vocab_map[val]
                    elif isinstance(val, dict) and "content" in val:
                        c = val["content"]
                        if c in vocab_map:
                            params[id_key] = vocab_map[c]

            except Exception as e:
                self.logger.warning(f"Tokenizer config check failed: {e}")

        return params, vocab_map

    def _load_generation_config(self, model_folder) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        gen_path = os.path.join(model_folder, "generation_config.json")
        if not os.path.exists(gen_path):
            return params

        try:
            with open(gen_path, "r") as f:
                data = json.load(f)

            # Context Length candidates
            if "max_length" in data:
                params["context_length"] = data["max_length"]
            elif "max_position_embeddings" in data:
                params["context_length"] = data["max_position_embeddings"]
            # EOS/BOS/PAD IDs
            for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
                if k in data:
                    params[k] = data[k]

            return params

        except Exception as e:
            self.logger.warning(f"Generation config check failed: {e}")
            return params

    def _infer_derived_params(self, params):
        """Strict inference of parameters. Raises errors if impossible."""

        # 1. Attention Heads
        if "num_attention_heads" not in params:
            # Check aliases
            if "n_head" in params:
                params["num_attention_heads"] = params["n_head"]
            elif "hidden_size" in params and "head_dim" in params:
                # Strict derivation
                params["num_attention_heads"] = (
                    params["hidden_size"] // params["head_dim"]
                )
            else:
                raise ValueError(
                    "'num_attention_heads' missing and cannot be derived (need hidden_size + head_dim)."
                )

        # 2. Key Value Heads
        if "num_key_value_heads" not in params:
            if "n_key_value_heads" in params:
                params["num_key_value_heads"] = params["n_key_value_heads"]
            elif "num_kv_heads" in params:
                params["num_key_value_heads"] = params["num_kv_heads"]
            else:
                params["num_key_value_heads"] = params["num_attention_heads"]

        # 3. Head Dim
        if "head_dim" not in params:
            if "head_size" in params:
                params["head_dim"] = params["head_size"]
            elif "hidden_size" in params and "num_attention_heads" in params:
                params["head_dim"] = (
                    params["hidden_size"] // params["num_attention_heads"]
                )
            else:
                raise ValueError("'head_dim' missing and cannot be derived.")

        # 4. Context Length
        if "context_length" not in params:
            # Aliases
            candidates = [
                "max_position_embeddings",
                "n_ctx",
                "seq_length",
                "max_sequence_length",
            ]
            found = False
            for c in candidates:
                if c in params:
                    params["context_length"] = params[c]
                    found = True
                    break

            if not found:
                raise ValueError("context_length could not be determined.")

        # 5. Pad Token Fallback
        if "pad_token_id" not in params and "eos_token_id" in params:
            # If EOS is a single int, use it for PAD
            eos = params["eos_token_id"]
            if isinstance(eos, int):
                self.logger.debug(f"Inferring pad_token_id from eos_token_id: {eos}")
                params["pad_token_id"] = eos
            elif isinstance(eos, list) and len(eos) > 0:
                self.logger.debug(
                    f"Inferring pad_token_id from first eos_token_id: {eos[0]}"
                )
                params["pad_token_id"] = eos[0]

    def _validate_params(self, params):
        required = [
            "vocab_size",
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "eos_token_id",
        ]
        for r in required:
            if r not in params:
                raise ValueError(f"Missing required parameter '{r}'. Cannot proceed.")

    def _map_io_names(self, onnx_names: List[str], io_type: str) -> Dict[str, str]:
        mapping = {}

        # 1. Direct Pattern Match
        for candidate in self.IO_CANDIDATES.get(io_type, []):
            for name in onnx_names:
                if candidate in name:
                    mapping[candidate] = name
                    break

        # 2. KV Cache Logic
        if io_type == "inputs":
            if any("past_key_values" in n for n in onnx_names):
                mapping["past_key_names"] = "past_key_values.%d.key"
                mapping["past_value_names"] = "past_key_values.%d.value"
        elif io_type == "outputs":
            if any("present" in n for n in onnx_names):
                mapping["present_key_names"] = "present.%d.key"
                mapping["present_value_names"] = "present.%d.value"

        # 3. Fallback: Direct Mapping (Heuristic)
        # If we encounter an IO name we don't recognize, try to map it using the last segment.
        # But skip proper KV cache names.
        mapped_values = set(mapping.values())
        for name in onnx_names:
            if name in mapped_values:
                continue
            if "past_key_values" in name or "present" in name:
                continue

            short_name = name.split(".")[-1]
            if short_name not in mapping:
                mapping[short_name] = name

        return mapping

    def _filter_model_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Strict keep-list to avoid polluting config with HF-only junk."""
        return {k: v for k, v in params.items() if k in self.MODEL_PARAM_ALLOWLIST}
