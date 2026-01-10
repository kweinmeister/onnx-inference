import json
import logging
import os
import shutil
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import onnx

import onnxruntime
import onnxruntime_genai as _og
from huggingface_hub import snapshot_download

from config import GenAIConfigGenerator

# Silence static analysis complaints about missing members in C-extension
og = cast(Any, _og)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("onnxruntime_genai").setLevel(logging.WARNING)

DEFAULT_PROVIDERS = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]
DEFAULT_MODEL_ID = "onnx-community/SmolLM2-135M-ONNX"


def bundle_onnx_model(
    input_path: str, output_path: Optional[str] = None
) -> Optional[str]:
    """
    Bundles an ONNX model with its external data into a single file.
    Critically required for CoreMLExecutionProvider to work on macOS with split models.
    Only attempts bundling if total size < 2GB (Protobuf limit).

    Args:
        input_path: Path to the source .onnx file
        output_path: Optional destination path. If None, appends '_bundled.onnx'

    Returns:
        Path to the bundled model file, or None if bundling is skipped/failed.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_bundled{ext}"

    # If already exists, assume valid and return
    if os.path.exists(output_path):
        return output_path

    try:
        # Check size constraints
        # 1. Base proto size
        total_size = os.path.getsize(input_path)

        # 2. External data size (without loading full model into memory)
        model_head = onnx.load(input_path, load_external_data=False)
        base_dir = os.path.dirname(input_path)

        # Robust size check: Sum unique external files + proto size
        external_files = set()
        for tensor in model_head.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                for entry in tensor.external_data:
                    if entry.key == "location":
                        external_files.add(entry.value)

        for fname in external_files:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                total_size += os.path.getsize(fpath)

        limit = 2 * 1024 * 1024 * 1024  # 2GB limit
        if total_size >= limit:
            logger.warning(
                f"Model size ({total_size / 1024**3:.2f} GB) exceeds 2GB limit. Skipping bundling."
            )
            return None

        logger.info(f"Bundling CoreML model from {input_path} to {output_path}...")
        model = onnx.load(input_path)
        onnx.save(model, output_path)
        logger.info("✓ Model bundling complete.")

        return output_path

    except Exception as e:
        logger.warning(f"Failed to bundle model: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


class OnnxTextGenerator:
    def __init__(
        self,
        model_id: Optional[str] = DEFAULT_MODEL_ID,
        onnx_file: Optional[str] = None,
        execution_providers: Optional[Union[str, List[str]]] = None,
        num_beams: int = 1,
    ) -> None:
        """
        Initializes the ONNX GenAI model and tokenizer.

        Args:
            model_id: HuggingFace model ID or local path.
            onnx_file: Optional specific ONNX file to use.
            execution_provider: Optional specific provider. If None, tries DEFAULT_PROVIDERS.
            num_beams: Default number of beams for search.
        """

        self.num_beams = num_beams

        # 1. Download/Locate Model
        if model_id is None:
            model_id = DEFAULT_MODEL_ID
        self.repo_folder = self._get_model_path(model_id, onnx_file)

        # 2. Detect proper model folder (if model is in subdir)
        self.model_folder = self.repo_folder
        try:
            # Search relative to repo folder to see if we should scope down
            rel_path = self._detect_onnx_model(self.repo_folder, onnx_file)
            subdir = os.path.dirname(rel_path)
            if subdir:
                self.model_folder = os.path.join(self.repo_folder, subdir)
                logger.debug(f"Targeting model in subdirectory: {self.model_folder}")
        except FileNotFoundError:
            # _ensure_genai_config will handle detection failure or raising later
            pass

        # 3. Detect Provider (moved up to handle CoreML specifics)
        available = onnxruntime.get_available_providers()  # type: ignore
        candidates = execution_providers or DEFAULT_PROVIDERS
        if isinstance(candidates, str):
            candidates = [candidates]

        self.execution_provider = "CPUExecutionProvider"  # Fallback
        for p in candidates:
            if p in available:
                self.execution_provider = p
                break

        # 5. Ensure genai_config.json exists (auto-generate if missing)
        self._ensure_genai_config(onnx_file, repo_root=self.repo_folder)

        # 6. Ensure tokenizer files exist at root (copy from subdirs if needed)
        self._ensure_tokenizer_files(repo_root=self.repo_folder)

        logger.info(f"Loading GenAI Model from {self.model_folder}...")

        # 7. Load Model (with Reactive CoreML Bundling)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.config = og.Config(self.model_folder)
                # Apply hardware settings (CUDA, CoreML, etc.)
                if self.execution_provider != "CPUExecutionProvider":
                    self.config.clear_providers()
                    self.config.append_provider(self.execution_provider)

                # Optimization settings based on search type
                try:
                    if self.execution_provider == "CUDAExecutionProvider":
                        if self.num_beams > 1:
                            self.config.set_provider_option(
                                self.execution_provider,
                                "past_present_share_buffer",
                                "0",
                            )
                            self.config.set_provider_option(
                                self.execution_provider, "enable_cuda_graph", "0"
                            )
                        else:
                            self.config.set_provider_option(
                                self.execution_provider,
                                "past_present_share_buffer",
                                "1",
                            )
                except Exception:
                    logger.debug(
                        f"Optional optimizations not supported for provider: {self.execution_provider}"
                    )

                self.config.overlay(
                    json.dumps({"search": {"num_beams": self.num_beams}})
                )

                # Instantiate
                self.model = og.Model(self.config)
                self.tokenizer = og.Tokenizer(self.model)
                logger.info(f"✓ Model loaded with {self.execution_provider}")
                break  # Success

            except Exception as e:
                # Checkout for specific CoreML "Not a directory" error
                if (
                    self.execution_provider == "CoreMLExecutionProvider"
                    and "Not a directory" in str(e)
                    and attempt == 0
                ):
                    logger.warning(
                        f"CoreML load failed with 'Not a directory'. Attempting bundling workaround... ({e})"
                    )
                    try:
                        detected_file = self._detect_onnx_model(
                            self.model_folder, onnx_file
                        )
                        base, ext = os.path.splitext(detected_file)
                        # Avoid double bundling if we already tried?
                        # Assuming detected_file is the one that failed.
                        fixed_filename = f"{base}_coreml_bundled{ext}"
                        fixed_path = os.path.join(self.model_folder, fixed_filename)
                        full_detected_path = os.path.join(
                            self.model_folder, detected_file
                        )

                        bundled_path = bundle_onnx_model(full_detected_path, fixed_path)
                        if bundled_path:
                            # Update config to point to new file
                            self._ensure_genai_config(
                                fixed_filename, repo_root=self.repo_folder
                            )
                            continue  # Retry loop
                    except Exception as bundle_err:
                        logger.error(f"Bundling failed: {bundle_err}")

                logger.error(
                    f"Failed to load model using {self.execution_provider}: {e}"
                )
                raise

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        min_length: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> Iterator[tuple[str, dict]]:
        """
        Yields generated tokens one by one using OGA's generator.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            min_length: Minimum total length (prompt + gen)
            temperature: Sampling temperature (0 for greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1 = penalize)
            num_beams: Optional override for beam search
            do_sample: Optional explicit sampling toggle

        Yields:
            tuple: (chunk_text, metadata_dict) where metadata contains tokens_generated and finish_reason
        """
        # 1. Validation for Multimodal models (Beam search is currently unstable in OGA for these)
        check_beams = num_beams if num_beams is not None else self.num_beams
        if check_beams > 1:
            config_path = os.path.join(self.model_folder, "genai_config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    model_cfg = cfg.get("model", {})
                    if (
                        "vision" in model_cfg
                        or "speech" in model_cfg
                        or "embedding" in model_cfg
                    ):
                        raise ValueError(
                            f"Beam search (num_beams={check_beams}) is currently not supported for "
                            "multimodal models in ONNX GenAI due to stability issues in the "
                            "underlying library's beam expansion logic. Please use num_beams=1."
                        )
            except ValueError:
                raise
            except Exception:
                pass

        # 1. Tokenize input (use encode_batch for consistency with GenAI examples)
        input_tokens = self.tokenizer.encode_batch([prompt])
        # input_tokens is a Tensor with shape (batch_size, sequence_length)
        prompt_length = input_tokens.shape()[1]  # shape() is a method

        # 2. Prepare Generation Parameters
        params = og.GeneratorParams(self.model)

        # Set search options
        # IMPORTANT: max_length is total sequence length (prompt + generation)
        search_options = {
            "max_length": prompt_length + max_new_tokens,
            "min_length": min_length if min_length > 0 else prompt_length,
            "repetition_penalty": repetition_penalty,
        }

        # Sampling logic
        if do_sample is not None:
            search_options["do_sample"] = do_sample
        else:
            search_options["do_sample"] = temperature > 0 and (
                num_beams is None or num_beams <= 1
            )

        if search_options["do_sample"]:
            search_options["temperature"] = temperature
            search_options["top_p"] = top_p

        # Beam search logic
        if num_beams is not None:
            search_options["num_beams"] = num_beams
            if num_beams > 1:
                # Beam search is mutually exclusive with sampling
                if do_sample is True:
                    raise ValueError(
                        f"Beam search (num_beams={num_beams}) is mutually exclusive with do_sample=True."
                    )
                search_options["do_sample"] = False

        params.set_search_options(**search_options)

        # 3. Create Generator and append input tokens
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        tokenizer_stream = self.tokenizer.create_stream()

        tokens_generated = 0

        try:
            while not generator.is_done():
                generator.generate_next_token()

                if generator.is_done():
                    break

                # Get the new token
                new_token = generator.get_next_tokens()[0]

                # Decode incrementally
                chunk = tokenizer_stream.decode(new_token)

                tokens_generated += 1

                # Check if we've hit max_new_tokens
                if tokens_generated >= max_new_tokens:
                    if chunk:
                        yield (
                            chunk,
                            {
                                "tokens_generated": tokens_generated,
                                "finish_reason": None,
                            },
                        )
                    yield (
                        "",
                        {
                            "tokens_generated": tokens_generated,
                            "finish_reason": "length",
                        },
                    )
                    break

                # Yield chunk if there's text
                if chunk:
                    yield (
                        chunk,
                        {"tokens_generated": tokens_generated, "finish_reason": None},
                    )

            # If loop finished naturally (EOS token)
            if generator.is_done() and tokens_generated < max_new_tokens:
                yield (
                    "",
                    {"tokens_generated": tokens_generated, "finish_reason": "stop"},
                )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        min_length: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> dict:
        """
        Full generation wrapper.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            min_length: Minimum total length
            temperature: Sampling temperature (0 for greedy)
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1 = penalize)
            num_beams: Optional override for beam search
            do_sample: Optional explicit sampling toggle

        Returns:
            dict: {
                "generated_text": str,
                "finish_reason": str,  # "stop" or "length"
                "tokens_generated": int
            }
        """
        chunks: List[str] = []
        metadata = {}
        for chunk, meta in self.stream_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            do_sample=do_sample,
        ):
            chunks.append(chunk)
            metadata = meta

        return {
            "generated_text": "".join(chunks),
            "finish_reason": metadata.get("finish_reason") or "length",
            "tokens_generated": metadata.get("tokens_generated", 0),
        }

    def _ensure_tokenizer_files(self, repo_root: Optional[str] = None) -> None:
        """Ensures that tokenizer config files are in the model folder."""
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        ]

        for filename in tokenizer_files:
            root_path = os.path.join(self.model_folder, filename)
            if os.path.exists(root_path):
                continue

            # If not in root, try to find it recursively
            try:
                # Reuse the detection logic to find the file
                rel_path = self._detect_onnx_model(
                    self.model_folder, filename, fallback=False
                )
                full_path = os.path.join(self.model_folder, rel_path)
                # Copy to root
                shutil.copy2(full_path, root_path)

                logger.info(
                    f"✓ Copied tokenizer file {filename} from {rel_path} to root"
                )

            except (FileNotFoundError, RuntimeError):
                # File might not exist for this model (e.g. merges.txt for some), which is fine
                pass

        # If files are still missing but we have a repo_root parent, try copying from there
        if repo_root and repo_root != self.model_folder:
            for filename in tokenizer_files:
                target_path = os.path.join(self.model_folder, filename)
                if os.path.exists(target_path):
                    continue

                source_path = os.path.join(repo_root, filename)
                if os.path.exists(source_path):
                    try:
                        shutil.copy2(source_path, target_path)
                        logger.info(
                            f"✓ Copied tokenizer file {filename} from repo root to subdir"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to copy {filename}: {e}")

    def _ensure_genai_config(
        self, onnx_file: Optional[str] = None, repo_root: Optional[str] = None
    ) -> None:
        """
        Ensures genai_config.json exists in the model folder.
        If it doesn't exist, generates it from HuggingFace config.json.
        """
        # GenAI requires config in the root folder
        root_config_path = os.path.join(self.model_folder, "genai_config.json")

        # 1. Detect ONNX model filename FIRST (recursive scan)
        try:
            full_model_path = self._detect_onnx_model(self.model_folder, onnx_file)
        except FileNotFoundError as e:
            logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Could not find any ONNX model file. Error: {e}")

        # 2. Check if a genai_config.json already exists in root
        if os.path.exists(root_config_path):
            logger.info(f"✓ Found existing genai_config.json at {root_config_path}")
            if self._verify_and_fix_config_filename(root_config_path, full_model_path):
                logger.info("✓ Verified/Adjusted model filename in genai_config.json")
            return

        # 3. Look for an existing genai_config.json near the model file or in onnx/
        model_dir = os.path.dirname(os.path.join(self.model_folder, full_model_path))
        candidate_configs = [
            os.path.join(model_dir, "genai_config.json"),
            os.path.join(self.model_folder, "onnx", "genai_config.json"),
        ]
        if repo_root:
            candidate_configs.append(os.path.join(repo_root, "genai_config.json"))

        for config_path in candidate_configs:
            if os.path.exists(config_path):
                logger.info(
                    f"Found existing config at {config_path}, copying to root..."
                )
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    # Update ALL 'filename' fields to the relative path within the repo
                    # (e.g. gemma-3-vision.onnx -> path/to/gemma-3-vision.onnx)

                    # Special case: ensure the primary decoder filename matches our detected one
                    # This is important if we detected 'onnx/model.onnx' but config says 'model.onnx'
                    if "model" in config and "decoder" in config["model"]:
                        config["model"]["decoder"]["filename"] = full_model_path

                    # Run the recursive fixer to ensure everything else points to valid files
                    self._fix_paths_in_config(config)

                    with open(root_config_path, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=4)

                    logger.info(
                        f"✓ Copied and adjusted genai_config.json to {root_config_path}"
                    )
                    return
                except Exception as e:
                    logger.warning(f"Failed to use config at {config_path}: {e}")

        # 4. Generate genai_config if none found
        logger.info("genai_config.json not found, generating from metadata...")
        try:
            # Find the best baseline metadata (config.json)
            # Priorities: 1. Same directory as the ONNX file, 2. Root directory
            model_dir = os.path.dirname(
                os.path.join(self.model_folder, full_model_path)
            )
            candidate_metadata = [
                os.path.join(model_dir, "config.json"),
                os.path.join(self.model_folder, "config.json"),
            ]
            if repo_root:
                candidate_metadata.append(os.path.join(repo_root, "config.json"))

            hf_config: Dict[str, Any] = {}
            for meta_path in candidate_metadata:
                if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                data = json.loads(content)
                                # Primary check: does this config actually have token information?
                                # (Some ONNX subdirs have "dummy" configs with just model_type)
                                if "eos_token_id" in data or not hf_config:
                                    hf_config = data
                                    logger.info(
                                        f"✓ Found baseline metadata at {meta_path}"
                                    )
                                    if "eos_token_id" in hf_config:
                                        break  # Found a good one
                    except Exception as e:
                        logger.debug(f"Skipping metadata at {meta_path}: {e}")

            # Generate genai_config using our generator
            genai_config = self._create_genai_config(hf_config, full_model_path)

            # Save to root directory
            with open(root_config_path, "w", encoding="utf-8") as f:
                json.dump(genai_config, f, indent=4)

            logger.info(f"✓ Generated genai_config.json at {root_config_path}")

        except Exception as e:
            msg = f"Failed to generate genai_config.json: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

        # Final verification
        if not os.path.exists(root_config_path):
            raise RuntimeError(f"Internal Error: {root_config_path} was not created.")

    def _verify_and_fix_config_filename(
        self, config_path: str, full_model_path: str
    ) -> bool:
        """
        Verifies that all 'filename' entries in the config point to existing files.
        If not, tries to detect the correct ones recursively.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            any_changed = False
            # Force the primary decoder model to match our detected best model
            if "model" in config and "decoder" in config["model"]:
                # Ensure it's using the correct path
                if config["model"]["decoder"].get("filename") != full_model_path:
                    config["model"]["decoder"]["filename"] = full_model_path
                    logger.info(f"✓ Forced primary decoder to {full_model_path}")
                    any_changed = True

            if self._fix_paths_in_config(config):
                any_changed = True

            if any_changed:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4)
                logger.info("✓ Updated genai_config.json with verified paths")
                return True

            return True

        except Exception as e:
            logger.warning(f"Failed to verify/fix config paths: {e}")
            return False

    def _fix_paths_in_config(self, config: dict) -> bool:
        """
        Recursively finds 'filename' keys and ensures they point to valid files.
        If a file is missing, searches recursively in the model folder.
        Returns True if any paths were updated.
        """
        any_changed = False

        def fix_recursive(d):
            nonlocal any_changed
            if not isinstance(d, (dict, list)):
                return

            # Use a list to iterate if it's a list
            if isinstance(d, list):
                for item in d:
                    fix_recursive(item)
                return

            # If it's a dict, check for filename-like keys and recurse values
            for k, v in d.items():
                if isinstance(v, str) and (k == "filename" or k.endswith("_filename")):
                    # 1. Check if path exists as-is (relative to root)
                    full_root = os.path.join(self.model_folder, v)
                    if os.path.exists(full_root):
                        continue

                    # 2. Not found, try to find it recursively
                    logger.warning(
                        f"File '{v}' not found at {full_root}. Searching recursively..."
                    )
                    try:
                        # Search for the same basename, NO fallback for auxiliary models
                        new_rel_path = self._detect_onnx_model(
                            self.model_folder, v, fallback=False
                        )
                        d[k] = new_rel_path
                        logger.info(f"✓ Replaced {v} with {new_rel_path}")
                        any_changed = True
                    except FileNotFoundError:
                        logger.error(f"Could not find replacement for {v}")

                # Recurse into nested structures (dict or list)
                elif isinstance(v, (dict, list)):
                    fix_recursive(v)

        fix_recursive(config)
        return any_changed

    def _detect_onnx_model(
        self,
        folder: str,
        preferred_file: Optional[str] = "model.onnx",
        fallback: bool = True,
    ) -> str:
        """Finds the primary ONNX model file.

        Prioritizes:
        1. Exact path match relative to folder.
        2. Exact filename match in the root of the folder.
        3. Recursive search (only as a fallback or if using default filename).

        Args:
            folder: Base directory to search
            preferred_file: Preferred filename (basename or partial path). If None, defaults to "model.onnx".
            fallback: Whether to return any ONNX file if preferred is not found

        Returns:
            ONNX model path relative to folder
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Directory not found: {folder}")

        if preferred_file is None:
            preferred_file = "model.onnx"

        # 1. Check exact path as provided
        direct_path = os.path.join(folder, preferred_file)
        if os.path.isfile(direct_path):
            return preferred_file

        # 2. Check root folder for the filename (if preferred_file was a path)
        target_base = os.path.basename(preferred_file)
        root_path = os.path.join(folder, target_base)
        if os.path.isfile(root_path):
            return target_base

        # 3. Recursive pass: look for exact filename match
        for root, _, files in os.walk(folder):
            if target_base in files:
                full_path = os.path.join(root, target_base)
                return os.path.relpath(full_path, folder)

        # 4. If a specific non-default file was requested, be stricter
        # If we reached here, we didn't find the exact file recursively.
        is_custom_file = preferred_file != "model.onnx"
        if is_custom_file and not fallback:
            raise FileNotFoundError(
                f"Specific ONNX model '{preferred_file}' not found in {folder}"
            )

        # 5. Fallback: find ANY .onnx file recursively (only if permitted)
        if fallback:
            all_onnx = []
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.endswith(".onnx"):
                        all_onnx.append(os.path.join(root, f))

            if all_onnx:
                # Sort to prefer "text", "decoder", or "model" hits
                keywords = ["text", "decoder", "model"]
                for kw in keywords:
                    for path in all_onnx:
                        if kw in os.path.basename(path).lower():
                            logger.info(f"Using detected {kw} model: {path}")
                            return os.path.relpath(path, folder)

                # Absolute fallback: first one
                full_path = all_onnx[0]
                logger.info(f"Using first available ONNX model: {full_path}")
                return os.path.relpath(full_path, folder)

        raise FileNotFoundError(
            f"No ONNX model files found in {folder} (searched for {preferred_file})"
        )

    def _create_genai_config(self, hf_config: dict, model_filename: str) -> dict:
        """
        Creates a genai_config.json structure from HuggingFace config.
        Delegates to the GenAIConfigGenerator class.

        Args:
            hf_config: HuggingFace config.json contents
            model_filename: ONNX model filename to use (relative to model folder)

        Returns:
            GenAI config dictionary
        """
        generator = GenAIConfigGenerator()
        return generator.create_config(hf_config, model_filename, self.model_folder)

    def _get_model_path(self, model_id: str, onnx_file: Optional[str] = None) -> str:
        """Downloads model via HF Hub or finds it in cache."""

        def check_integrity(folder):
            # 1. Do we have at least one configuration file for metadata?
            has_config = any(
                os.path.exists(os.path.join(folder, f))
                for f in ["config.json", "genai_config.json", "tokenizer_config.json"]
            )
            if not has_config:
                return False

            # 2. If a specific file is requested, ensure it (and any .data) exists
            if onnx_file:
                for root, _, files in os.walk(folder):
                    if onnx_file in files:
                        # Check for .data file as well
                        if os.path.exists(os.path.join(root, f"{onnx_file}.data")):
                            return True

                        # If the .onnx file is > 100MB, it likely doesn't have external data
                        if (
                            os.path.getsize(os.path.join(root, onnx_file))
                            > 100 * 1024 * 1024
                        ):
                            return True

                        # Missing data file for small ONNX
                        return False
                return False

            # 3. Default: Do we have at least ONE .onnx file?
            for root, _, files in os.walk(folder):
                if any(f.endswith(".onnx") for f in files):
                    return True
            return False

        # Construct allow_patterns: common root files plus specific onnx if requested
        # We use "*" patterns (e.g. *.json) which are recursive in huggingface_hub,
        # matching files in root AND subdirectories. "**/pattern" does NOT match root files.
        patterns = ["*.json", "*.txt", "*.model", "*.py", "*.md", "LICENSE*", "*.data"]

        if onnx_file:
            # Strip extension for multi-part matching
            base = onnx_file
            if base.endswith(".onnx"):
                base = base[:-5]

            # Match recursively throughout the repo (matches 'model.onnx' and 'onnx/model.onnx')
            patterns.append(f"*{base}.onnx*")
            logger.info(f"Targeting specific ONNX model: {onnx_file}")
        else:
            # If no onnx_file specified, download ALL .onnx models recursively
            patterns.append("*.onnx*")
            logger.info("No specific ONNX file requested, downloading all .onnx files.")

        try:
            logger.info(f"Checking cache for model {model_id}...")
            model_folder = snapshot_download(
                repo_id=model_id,
                allow_patterns=patterns,
                local_files_only=True,
            )

            # Check if we actually have a model to run
            if check_integrity(model_folder):
                logger.info(f"✓ Model found in cache: {model_folder}")
                return model_folder
            else:
                logger.warning("Cache exists but no ONNX model files found.")

        except Exception:
            pass  # Fall through to download

        logger.info(f"Downloading model {model_id} (patterns: {patterns})...")
        model_folder = snapshot_download(repo_id=model_id, allow_patterns=patterns)
        logger.info(f"✓ Model downloaded to: {model_folder}")
        return model_folder
