"""
vLLM backend implementation with multi-GPU support
"""

import gc
import logging
import time
from collections.abc import Generator
from typing import Any, Optional

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available. Install with: pip install vllm")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

from utils.gpu_monitor import get_gpu_monitor

from .base import BaseBackend

logger = logging.getLogger(__name__)


class VLLMBackend(BaseBackend):
    """vLLM backend with multi-GPU support via tensor parallelism"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vLLM backend

        Args:
            config: Backend configuration with vLLM settings
        """
        super().__init__(config)
        self.llm_engine: Optional[Any] = None
        self.gpu_monitor = get_gpu_monitor()

        # Detect available GPUs
        self.gpu_count = self._detect_gpus()

        # vLLM configuration with defaults
        self.vllm_config = {
            "tensor_parallel_size": config.get("tensor_parallel_size", self.gpu_count),
            "pipeline_parallel_size": config.get("pipeline_parallel_size", 1),
            "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.90),
            "max_num_seqs": config.get("max_num_seqs", 256),
            "max_num_batched_tokens": config.get("max_num_batched_tokens"),
            "max_model_len": config.get("max_model_len"),
            "kv_cache_dtype": config.get("kv_cache_dtype", "auto"),
            "dtype": config.get("dtype", "auto"),
            "enforce_eager": config.get("enforce_eager", False),
            "enable_prefix_caching": config.get("enable_prefix_caching", True),
            "enable_chunked_prefill": config.get("enable_chunked_prefill", False),
            "max_parallel_loading_workers": config.get("max_parallel_loading_workers"),
            "disable_custom_all_reduce": config.get("disable_custom_all_reduce", False),
            "quantization": config.get("quantization"),
            "trust_remote_code": config.get("trust_remote_code", True),
            "tokenizer_mode": config.get("tokenizer_mode", "auto"),
            "download_dir": config.get("download_dir"),
            "load_format": config.get("load_format", "auto"),
            "seed": config.get("seed", 0)
        }

        # Adjust tensor parallel size based on GPU availability
        if self.vllm_config["tensor_parallel_size"] > self.gpu_count:
            logger.warning(f"Requested tensor_parallel_size ({self.vllm_config['tensor_parallel_size']}) "
                           f"exceeds available GPUs ({self.gpu_count}). Adjusting to {self.gpu_count}")
            self.vllm_config["tensor_parallel_size"] = max(1, self.gpu_count)

        logger.info(f"vLLM Backend initialized with {self.gpu_count} GPU(s)")
        logger.info(f"Tensor parallel size: {self.vllm_config['tensor_parallel_size']}")

    def _detect_gpus(self) -> int:
        """Detect number of available GPUs"""
        if not TORCH_AVAILABLE:
            return 0

        try:
            count = torch.cuda.device_count()
            if count > 0:
                logger.info(f"Detected {count} CUDA device(s)")
                for i in range(count):
                    name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"  GPU {i}: {name} ({memory:.1f} GB)")
            return count
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return 0

    def _calculate_optimal_settings(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate optimal vLLM settings based on model and hardware

        Args:
            model_config: Model-specific configuration

        Returns:
            Optimized settings dictionary
        """
        settings = self.vllm_config.copy()

        # Get GPU memory info
        gpu_memory = self.gpu_monitor.get_total_gpu_memory()
        total_memory_gb = gpu_memory[0] / 1024 if gpu_memory[0] > 0 else 24  # Default 24GB

        # Estimate model size (rough approximation)
        model_size = model_config.get("size", "30B").upper()
        if "B" in model_size:
            param_count = float(model_size.replace("B", ""))
            # Estimate memory for quantized model (4-bit ≈ 0.5 bytes per param)
            model_memory_gb = param_count * 0.5
        else:
            model_memory_gb = 10  # Default estimate

        # Calculate available memory for KV cache
        kv_cache_memory_gb = (total_memory_gb - model_memory_gb) * settings["gpu_memory_utilization"]

        # Estimate maximum context length based on available memory
        # Rough estimate: 100KB per 1K tokens for KV cache
        kv_mb_per_1k_tokens = 100
        max_context_from_memory = int((kv_cache_memory_gb * 1024) / kv_mb_per_1k_tokens * 1000)

        # Use model's max context or calculated, whichever is smaller
        model_max_context = model_config.get("max_context_tokens", 131072)
        optimal_context = min(max_context_from_memory, model_max_context)

        # Set max_model_len if not specified
        if settings["max_model_len"] is None:
            settings["max_model_len"] = optimal_context
            logger.info(f"Calculated optimal context length: {optimal_context} tokens")

        # Adjust batch settings based on context
        if settings["max_num_batched_tokens"] is None:
            settings["max_num_batched_tokens"] = min(optimal_context, 32768)

        # Enable optimizations for large models
        if param_count >= 30:
            settings["enable_chunked_prefill"] = True
            logger.info("Enabled chunked prefill for large model")

        return settings

    def load_model(self, model_path: str, model_config: dict[str, Any]) -> bool:
        """
        Load a model using vLLM

        Args:
            model_path: Path to the model (HuggingFace format or model ID)
            model_config: Model configuration

        Returns:
            True if successful, False otherwise
        """
        if not VLLM_AVAILABLE:
            logger.error("vLLM is not available")
            return False

        try:
            # Unload existing model
            self.unload_model()

            logger.info(f"Loading model: {model_path}")
            logger.info(f"Model config: {model_config}")

            # Calculate optimal settings
            settings = self._calculate_optimal_settings(model_config)

            # Log memory status before loading
            self.gpu_monitor.log_memory_status()

            start_time = time.time()

            # Create vLLM engine
            self.llm_engine = LLM(
                model=model_path,
                tensor_parallel_size=settings["tensor_parallel_size"],
                pipeline_parallel_size=settings["pipeline_parallel_size"],
                gpu_memory_utilization=settings["gpu_memory_utilization"],
                max_num_seqs=settings["max_num_seqs"],
                max_model_len=settings["max_model_len"],
                kv_cache_dtype=settings["kv_cache_dtype"],
                dtype=settings["dtype"],
                enforce_eager=settings["enforce_eager"],
                enable_prefix_caching=settings["enable_prefix_caching"],
                enable_chunked_prefill=settings["enable_chunked_prefill"],
                quantization=settings["quantization"],
                trust_remote_code=settings["trust_remote_code"],
                tokenizer_mode=settings["tokenizer_mode"],
                download_dir=settings["download_dir"],
                load_format=settings["load_format"],
                seed=settings["seed"]
            )

            load_time = time.time() - start_time

            self.model = self.llm_engine  # For compatibility
            self.model_path = model_path
            self.model_config = model_config

            logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
            if self.llm_engine:
                logger.info(f"Max model length: {self.llm_engine.llm_engine.model_config.max_model_len}")

            # Log memory status after loading
            self.gpu_monitor.log_memory_status()

            # Log GPU memory distribution
            if settings["tensor_parallel_size"] > 1:
                self._log_gpu_distribution()

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.llm_engine = None
            self.model = None
            self.model_path = None
            self.model_config = None
            return False

    def _log_gpu_distribution(self):
        """Log GPU memory distribution for multi-GPU setup"""
        try:
            gpu_info = self.gpu_monitor.get_gpu_info()
            logger.info("GPU Memory Distribution:")
            for gpu in gpu_info:
                logger.info(f"  GPU {gpu['index']}: {gpu['used_memory_mb']:.0f}MB / "
                            f"{gpu['total_memory_mb']:.0f}MB "
                            f"({gpu['memory_utilization_percent']:.1f}%)")
        except Exception as e:
            logger.debug(f"Could not log GPU distribution: {e}")

    def unload_model(self):
        """Unload the current model and free GPU memory"""
        if self.llm_engine:
            logger.info("Unloading vLLM model")
            try:
                # vLLM cleanup
                del self.llm_engine
                self.llm_engine = None
                self.model = None
                self.model_path = None
                self.model_config = None

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logger.info("Model unloaded successfully")

            except Exception as e:
                logger.error(f"Error unloading model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using vLLM

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """
        if not self.llm_engine:
            raise RuntimeError("No model loaded")

        # Validate request
        is_valid, error_msg = self.validate_request(prompt, **kwargs)
        if not is_valid:
            raise ValueError(error_msg)

        try:
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", -1),
                max_tokens=kwargs.get("max_tokens", 2048),
                stop=kwargs.get("stop"),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                length_penalty=kwargs.get("length_penalty", 1.0),
                seed=kwargs.get("seed")
            )

            # Generate
            outputs = self.llm_engine.generate([prompt], sampling_params)

            # Return the generated text
            text = outputs[0].outputs[0].text
            return str(text) if text is not None else ""

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming using vLLM

        Note: vLLM doesn't support true streaming in the same way as llama.cpp,
        but we can simulate it by generating in chunks

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Yields:
            Generated text chunks
        """
        if not self.llm_engine:
            raise RuntimeError("No model loaded")

        # For now, generate the full response and yield it in chunks
        # In production, you'd want to use vLLM's async engine for true streaming
        try:
            full_response = self.generate(prompt, **kwargs)

            # Yield response in chunks to simulate streaming
            chunk_size = kwargs.get("stream_chunk_size", 20)  # characters per chunk
            for i in range(0, len(full_response), chunk_size):
                yield full_response[i:i + chunk_size]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """
        Get backend status

        Returns:
            Status dictionary
        """
        status = {
            "backend": "vllm",
            "model_loaded": self.llm_engine is not None,
            "model_path": self.model_path,
            "model_config": self.model_config,
            "vllm_available": VLLM_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "gpu_count": self.gpu_count,
            "gpu_memory": self.gpu_monitor.get_memory_usage_summary(),
            "vllm_config": self.vllm_config
        }

        if self.llm_engine:
            try:
                status["model_info"] = {
                    "max_model_len": self.llm_engine.llm_engine.model_config.max_model_len,
                    "tensor_parallel_size": self.vllm_config["tensor_parallel_size"],
                    "gpu_memory_utilization": self.vllm_config["gpu_memory_utilization"]
                }
            except Exception:
                pass

        return status

    def supports_multi_gpu(self) -> bool:
        """Check if backend supports multi-GPU operation"""
        return self.gpu_count > 1

    def get_context_window(self) -> int:
        """Get the maximum context window size"""
        if self.llm_engine:
            try:
                max_len = self.llm_engine.llm_engine.model_config.max_model_len
                return int(max_len) if max_len is not None else super().get_context_window()
            except Exception:
                pass
        return super().get_context_window()
