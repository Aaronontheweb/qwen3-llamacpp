#!/usr/bin/env python3
"""
Test suite for vLLM backend with multi-GPU support
Run these tests on your Linux server with GPUs available
"""

import logging
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from backends.factory import BackendFactory
from backends.vllm_backend import VLLMBackend
from model_manager import ModelManager

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVLLMBackend:
    """Test suite for vLLM backend"""

    @pytest.fixture
    def single_gpu_config(self):
        """Configuration for single GPU testing"""
        return {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768
        }

    @pytest.fixture
    def multi_gpu_config(self):
        """Configuration for multi-GPU testing"""
        return {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 131072
        }

    def test_gpu_detection(self):
        """Test GPU detection"""
        backend = VLLMBackend({})
        assert backend.gpu_count >= 0

        if backend.gpu_count > 0:
            logger.info(f"Detected {backend.gpu_count} GPU(s)")
            assert torch.cuda.is_available()

    def test_single_gpu_initialization(self, single_gpu_config):
        """Test initialization with single GPU"""
        backend = VLLMBackend(single_gpu_config)

        assert backend.vllm_config["tensor_parallel_size"] == 1
        assert backend.vllm_config["gpu_memory_utilization"] == 0.9
        assert backend.vllm_config["max_model_len"] == 32768

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
    def test_multi_gpu_initialization(self, multi_gpu_config):
        """Test initialization with multiple GPUs"""
        backend = VLLMBackend(multi_gpu_config)

        assert backend.vllm_config["tensor_parallel_size"] == 2
        assert backend.supports_multi_gpu()

    def test_auto_tensor_parallel(self):
        """Test automatic tensor parallel size detection"""
        config = {"tensor_parallel_size": "auto"}
        backend = VLLMBackend(config)

        expected_size = max(1, torch.cuda.device_count())
        assert backend.vllm_config["tensor_parallel_size"] == expected_size

    def test_backend_factory(self):
        """Test backend creation through factory"""
        backend = BackendFactory.create_backend("vllm", {"tensor_parallel_size": 1})

        assert backend is not None
        assert backend.supports_streaming()
        assert "vllm" in BackendFactory.get_available_backends()


class TestModelLoading:
    """Test model loading with vLLM"""

    @pytest.fixture
    def test_model_path(self):
        """Path to a small test model"""
        # Use a small model for testing
        return "microsoft/Phi-3.5-mini-instruct"

    @pytest.mark.slow
    def test_load_small_model(self, test_model_path):
        """Test loading a small model"""
        backend = VLLMBackend({"tensor_parallel_size": 1})

        model_config = {
            "size": "4B",
            "max_context_tokens": 8192
        }

        success = backend.load_model(test_model_path, model_config)
        assert success

        # Check model is loaded
        assert backend.llm_engine is not None
        assert backend.get_context_window() > 0

        # Clean up
        backend.unload_model()

    @pytest.mark.slow
    def test_generation(self, test_model_path):
        """Test text generation"""
        backend = VLLMBackend({"tensor_parallel_size": 1})
        backend.load_model(test_model_path, {"size": "4B"})

        prompt = "Hello, how are you?"
        response = backend.generate(prompt, max_tokens=50, temperature=0.7)

        assert response is not None
        assert len(response) > 0
        logger.info(f"Generated response: {response}")

        backend.unload_model()


class TestMemoryManagement:
    """Test GPU memory management"""

    def test_memory_calculation(self):
        """Test optimal memory calculation"""
        backend = VLLMBackend({})

        model_config = {
            "size": "30B",
            "max_context_tokens": 131072
        }

        settings = backend._calculate_optimal_settings(model_config)

        assert "max_model_len" in settings
        assert settings["max_model_len"] > 0
        assert settings["max_model_len"] <= 131072

    @pytest.mark.gpu
    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring during model load"""
        if torch.cuda.device_count() == 0:
            pytest.skip("No GPUs available")

        backend = VLLMBackend({"tensor_parallel_size": 1})

        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()

        # Load small model
        backend.load_model("microsoft/Phi-3.5-mini-instruct", {"size": "4B"})

        # Check memory increased
        loaded_memory = torch.cuda.memory_allocated()
        assert loaded_memory > initial_memory

        # Unload and check memory released
        backend.unload_model()
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        assert final_memory < loaded_memory


class TestContextScaling:
    """Test context window scaling"""

    @pytest.mark.slow
    @pytest.mark.parametrize("context_size", [4096, 8192, 16384])
    def test_different_context_sizes(self, context_size):
        """Test different context window sizes"""
        config = {
            "tensor_parallel_size": 1,
            "max_model_len": context_size
        }

        backend = VLLMBackend(config)
        backend.load_model("microsoft/Phi-3.5-mini-instruct", {"size": "4B"})

        # Create prompt of appropriate size
        prompt = "Test " * (context_size // 10)

        try:
            response = backend.generate(prompt, max_tokens=10)
            assert response is not None
            logger.info(f"Successfully handled {context_size} token context")
        except Exception as e:
            logger.error(f"Failed at {context_size} tokens: {e}")
            raise
        finally:
            backend.unload_model()


class TestMultiGPUDistribution:
    """Test multi-GPU KV cache distribution"""

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
    @pytest.mark.slow
    def test_tensor_parallel_distribution(self):
        """Test that model is distributed across GPUs"""
        config = {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.9
        }

        backend = VLLMBackend(config)

        # Record memory before loading
        initial_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            initial_memory.append(torch.cuda.memory_allocated())

        # Load model
        backend.load_model("Qwen/Qwen2.5-7B-Instruct", {"size": "7B"})

        # Check memory on each GPU
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            current_memory = torch.cuda.memory_allocated()
            memory_used = (current_memory - initial_memory[i]) / 1024**3

            logger.info(f"GPU {i}: {memory_used:.2f} GB used")

            # Both GPUs should have significant memory usage
            if i < 2:
                assert memory_used > 1.0, f"GPU {i} not being utilized"

        backend.unload_model()


def run_basic_tests():
    """Run basic tests that don't require GPUs"""
    print("Running basic tests...")

    # Test imports
    try:
        from backends.vllm_backend import VLLMBackend
        print("✓ vLLM backend imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import vLLM backend: {e}")
        return False

    # Test factory
    try:
        from backends.factory import BackendFactory
        backends = BackendFactory.get_available_backends()
        print(f"✓ Available backends: {backends}")
    except Exception as e:
        print(f"✗ Backend factory error: {e}")
        return False

    # Test model manager
    try:
        manager = ModelManager()
        print(f"✓ Model manager initialized with backend: {manager.current_backend_type}")
    except Exception as e:
        print(f"✗ Model manager error: {e}")
        return False

    return True


if __name__ == "__main__":
    # Run basic tests first
    if not run_basic_tests():
        print("\nBasic tests failed. Please check your installation.")
        sys.exit(1)

    print("\nBasic tests passed!")
    print("\nTo run full test suite with GPUs:")
    print("  pytest tests/test_vllm_backend.py -v")
    print("\nTo run only GPU tests:")
    print("  pytest tests/test_vllm_backend.py -v -m gpu")
    print("\nTo run with specific GPU count:")
    print("  CUDA_VISIBLE_DEVICES=0,1 pytest tests/test_vllm_backend.py -v")
