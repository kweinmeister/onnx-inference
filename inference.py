import os
import logging
import json
from typing import cast, Iterator
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CONTEXT_LENGTH = 8192
DEFAULT_MODEL_ID = "onnx-community/gemma-3-270m-it-ONNX"
DEFAULT_ONNX_FILE = "onnx/model_q4f16.onnx"
DEFAULT_ALLOW_PATTERNS = ["*.json", "*.onnx*"]
PREFERRED_PROVIDERS = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


class OnnxTextGenerator:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_file: str = DEFAULT_ONNX_FILE,
        allow_patterns: list = DEFAULT_ALLOW_PATTERNS,
    ) -> None:
        """Initializes the ONNX inference session and tokenizer using Hugging Face Hub."""
        # Get model folder first (checks cache, downloads if needed)
        model_folder = self._get_model_path(model_id, onnx_file, allow_patterns)

        # Load tokenizer from cached model folder (no HTTP request)
        logger.info("Loading tokenizer from cache...")
        tokenizer_path = os.path.join(model_folder, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        model_path = os.path.join(model_folder, onnx_file)
        self.session = self._create_inference_session(model_path)

        # Log which provider is being used
        actual_provider = self.session.get_providers()[0]
        logger.info(f"Active execution provider: {actual_provider}")

        self.output_names = [output.name for output in self.session.get_outputs()]

        self._setup_eos_token(model_folder)
        self._setup_inputs(model_folder)

    def _create_inference_session(self, model_path: str) -> ort.InferenceSession:
        """Creates and configures the ONNX inference session."""
        sess_options = ort.SessionOptions()
        # Use sched_getaffinity on Linux (containers) or fallback to cpu_count
        if hasattr(os, "sched_getaffinity"):
            sess_options.intra_op_num_threads = len(os.sched_getaffinity(0))
        else:
            sess_options.intra_op_num_threads = os.cpu_count() or 1

        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        logger.info(f"Creating ONNX Inference Session from {model_path}...")

        # Use only providers that are both preferred and available
        available_providers = ort.get_available_providers()
        providers = [p for p in PREFERRED_PROVIDERS if p in available_providers]

        return ort.InferenceSession(model_path, sess_options, providers=providers)

    def _setup_eos_token(self, model_folder: str) -> None:
        """Sets up the EOS token ID from config or heuristics."""
        self.eos_token_id = None
        # Try to load generation config for EOS token
        try:
            config_path = os.path.join(model_folder, "generation_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.eos_token_id = config.get("eos_token_id")
                    if isinstance(self.eos_token_id, list):
                        self.eos_token_id = self.eos_token_id[0]
                logger.info(f"Loaded EOS token ID {self.eos_token_id} from config")
        except Exception as e:
            logger.warning(f"Could not load generation config: {e}")

        if self.eos_token_id is None:
            # Fallback to heuristics
            for token in ["</s>", "<|im_end|>", "<|endoftext|>", "<eos>"]:
                token_id = self.tokenizer.token_to_id(token)
                if token_id is not None:
                    self.eos_token_id = token_id
                    logger.info(f"Using EOS token: {token} (ID: {token_id})")
                    break

        if self.eos_token_id is None:
            logger.warning("Warning: No EOS token found. Generation might not stop.")

    def _setup_inputs(self, model_folder: str) -> None:
        """Analyzes model inputs and sets up KV cache structures."""
        inputs = self.session.get_inputs()
        self.use_position_ids = "position_ids" in [i.name for i in inputs]

        # Initialize KV cache dimensions
        num_key_value_heads = None
        head_dim = None

        # Try to load model config for dimensions
        try:
            config_path = os.path.join(model_folder, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

                    hidden_size = config.get("hidden_size")
                    num_attention_heads = config.get("num_attention_heads")

                    if hidden_size and num_attention_heads:
                        head_dim = hidden_size // num_attention_heads

                    # Handle GQA/MQA by checking num_key_value_heads
                    num_key_value_heads = (
                        config.get("num_key_value_heads", num_attention_heads)
                        or num_key_value_heads
                    )

                logger.info(
                    f"Loaded KV cache dims from config: heads={num_key_value_heads}, dim={head_dim}"
                )
        except Exception as e:
            logger.warning(f"Could not load model config for KV cache dims: {e}")

        for i in inputs:
            if i.name.startswith("past_key_values") and len(i.shape) == 4:
                try:
                    if isinstance(i.shape[1], int) and isinstance(i.shape[3], int):
                        # Use actual ONNX shapes if they are static integers,
                        # as this is safer for the engine
                        num_key_value_heads = i.shape[1]
                        head_dim = i.shape[3]
                        break
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not determine KV cache shape from {i.name}: {e}"
                    )

        if num_key_value_heads is None or head_dim is None:
            raise ValueError(
                "Could not determine KV cache dimensions (num_key_value_heads, head_dim) "
                "from config.json or ONNX model shapes. Please ensure the model has valid "
                "shapes or a valid config.json."
            )

        self.empty_past_inputs = {}

        for i in inputs:
            if i.name.startswith("past_key_values"):
                dtype = np.float16 if "float16" in i.type else np.float32
                self.empty_past_inputs[i.name] = np.zeros(
                    (1, num_key_value_heads, 0, head_dim), dtype=dtype
                )

        # Map outputs to inputs for KV cache
        self.output_to_input_map = {}
        for output_name in self.output_names:
            if "present" in output_name:
                # Typically present.0.key -> past_key_values.0.key
                # Replace 'present' with 'past_key_values'
                input_name = output_name.replace("present", "past_key_values")
                if input_name in self.empty_past_inputs:
                    self.output_to_input_map[output_name] = input_name

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Iterator[tuple[str, dict]]:
        """
        Yields generated tokens one by one along with metadata.

        Yields:
            tuple: (chunk_text, metadata_dict) where metadata is updated on each yield
        """
        (
            current_input_ids,
            current_attention_mask,
            past_key_values,
            all_token_ids,
            current_text,
        ) = self._prepare_generation_inputs(prompt, max_new_tokens)

        tokens_generated = 0

        current_seq_len = current_attention_mask.shape[1]

        for _ in range(max_new_tokens):
            current_attention_mask = np.ones((1, current_seq_len), dtype=np.int64)

            try:
                logits, past_key_values = self._run_inference_step(
                    current_input_ids, current_attention_mask, past_key_values
                )
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                break

            next_token_id = self._sample_token(logits, temperature, top_p)

            if (
                self.eos_token_id is not None
                and next_token_id.item() == self.eos_token_id
            ):
                yield ("", {
                    "tokens_generated": tokens_generated,
                    "finish_reason": "stop"
                })
                break

            all_token_ids.append(next_token_id.item())
            tokens_generated += 1

            # Prepare for next iteration
            current_input_ids = next_token_id
            current_seq_len += 1
            
            # Full-Decode Difference "Production-Grade" Strategy
            # 1. Decode the full sequence
            full_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
            
            # 2. Check if text grew and is not ending in a replacement character (UTF-8 check)
            # "\ufffd" is the replacement character used when a multi-byte char is incomplete
            if len(full_text) > len(current_text) and not full_text.endswith("\ufffd"):
                new_text_chunk = full_text[len(current_text):]
                current_text = full_text
                
                yield (new_text_chunk, {
                    "tokens_generated": tokens_generated,
                    "finish_reason": None 
                })

    def _prepare_generation_inputs(
        self, prompt: str, max_new_tokens: int
    ) -> tuple[np.ndarray, np.ndarray, dict, list[int], str]:
        """Encodes prompt and prepares initial state for generation."""
        encoding = self.tokenizer.encode(prompt)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        # Check context length
        if (
            input_ids.shape[1] + max_new_tokens > MAX_CONTEXT_LENGTH
        ):  # Soft limit warning
            logger.warning("Prompt + max_new_tokens might exceed model context")

        current_text = self.tokenizer.decode(
            input_ids[0].tolist(), skip_special_tokens=True
        )

        return (
            input_ids,
            attention_mask,
            self.empty_past_inputs,
            input_ids[0].tolist(),
            current_text,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict:
        """
        Autoregressive text generation using Temperature and Top-P (Nucleus) Sampling.

        Returns:
            dict: {
                "generated_text": str,
                "finish_reason": str,  # "stop" or "length"
                "tokens_generated": int
            }
        """
        chunks: list[str] = []
        metadata = {}
        for chunk, meta in self.stream_generate(
            prompt, max_new_tokens, temperature, top_p
        ):
            chunks.append(chunk)
            metadata = meta

        return {
            "generated_text": "".join(chunks),
            "finish_reason": metadata.get("finish_reason") or "length",
            "tokens_generated": metadata.get("tokens_generated", 0),
        }

    def _run_inference_step(
        self, input_ids: np.ndarray, attention_mask: np.ndarray, past_key_values: dict
    ) -> tuple[np.ndarray, dict]:
        """Runs a single inference step and returns logits and updated KV cache."""
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **past_key_values,
        }

        if self.use_position_ids:
            ort_inputs["position_ids"] = self._compute_position_ids(
                input_ids, attention_mask
            )

        outputs = cast(list[np.ndarray], self.session.run(None, ort_inputs))

        logits = outputs[0][0, -1, :]
        new_past_key_values = self._extract_past_from_outputs(outputs)

        return logits, new_past_key_values

    def _compute_position_ids(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Computes position IDs based on attention mask and input shape."""
        # Calculate position IDs for the current input tokens
        positions = (np.cumsum(attention_mask, axis=1) - 1).astype(np.int64)
        if input_ids.shape[1] < attention_mask.shape[1]:
            positions = positions[:, -input_ids.shape[1] :]
        return positions

    def _extract_past_from_outputs(self, outputs: list[np.ndarray]) -> dict:
        """Extracts KV cache from outputs and maps them to inputs for the next step."""
        past_key_values = {}
        # outputs is a list of numpy arrays. valid names are in self.output_names
        for name, array in zip(self.output_names, outputs):
            if name in self.output_to_input_map:
                input_name = self.output_to_input_map[name]
                past_key_values[input_name] = array
        return past_key_values

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
    ) -> np.ndarray:
        """Samples the next token from logits."""
        # Greedy generation
        if temperature <= 0:
            token = np.argmax(logits)
            return np.array([[token]], dtype=np.int64)

        # Temperature Scaling
        logits = logits / temperature
        logits -= np.max(logits)  # Numerical stability
        probs = np.exp(logits)
        probs /= np.sum(probs)
        
        # Add epsilon for numerical stability
        probs += 1e-12
        probs /= np.sum(probs)

        # Top-P (Nucleus) Sampling
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_ids = np.searchsorted(np.cumsum(sorted_probs), top_p)

            # Mask out tokens below the threshold, but keep at least one
            cutoff = sorted_indices[cumulative_ids + 1 :]
            probs[cutoff] = 0

            # Renormalize with safety check
            probability_sum = np.sum(probs)
            if probability_sum == 0:
                # Handle edge case where all probabilities are zero
                probs[sorted_indices[0]] = 1.0
            else:
                probs /= probability_sum

        # Sample
        token = np.random.choice(len(probs), p=probs)

        return np.array([[token]], dtype=np.int64)

    def _get_model_path(
        self, model_id: str, onnx_file: str, allow_patterns: list
    ) -> str:
        """
        Get the model path, checking cache first to avoid unnecessary downloads.

        Args:
            model_id: HuggingFace model ID
            onnx_file: Path to the ONNX file within the model repo
            allow_patterns: File patterns to download/cache

        Returns:
            Path to the model folder
        """
        try:
            # Try to load from cache without making network requests
            logger.info(f"Checking cache for model {model_id}...")
            model_folder = snapshot_download(
                repo_id=model_id,
                allow_patterns=allow_patterns,
                local_files_only=True,  # Don't make network requests
            )
            logger.info(f"✓ Model found in cache: {model_folder}")
            return model_folder
        except Exception:
            # Model not in cache, download it
            logger.info(f"Model not cached, downloading {model_id}...")
            model_folder = snapshot_download(
                repo_id=model_id, allow_patterns=allow_patterns
            )
            logger.info(f"✓ Model downloaded to: {model_folder}")
            return model_folder
