"""
Backend factory for creating different backend implementations
"""

import logging
from typing import Any, ClassVar

from .base import BaseBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory class for creating backend instances"""

    # Registry of available backends
    _backends: ClassVar[dict] = {}

    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """
        Register a new backend type

        Args:
            name: Backend name
            backend_class: Backend class
        """
        cls._backends[name] = backend_class
        logger.info(f"Registered backend: {name}")

    @classmethod
    def create_backend(cls, backend_type: str, config: dict[str, Any]) -> BaseBackend:
        """
        Create a backend instance

        Args:
            backend_type: Type of backend to create
            config: Backend configuration

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is not supported
        """
        # Lazy import backends to avoid circular dependencies
        if not cls._backends:
            cls._initialize_backends()

        backend_type = backend_type.lower()

        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(f"Unknown backend type: {backend_type}. Available: {available}")

        backend_class = cls._backends[backend_type]

        # Add backend type to config for reference
        config["backend_type"] = backend_type

        logger.info(f"Creating {backend_type} backend")
        backend_instance = backend_class(config)
        return backend_instance

    @classmethod
    def _initialize_backends(cls):
        """Initialize available backends with lazy imports"""

        # Try to import vLLM backend
        try:
            from .vllm_backend import VLLMBackend
            cls.register_backend("vllm", VLLMBackend)
        except ImportError as e:
            logger.warning(f"vLLM backend not available: {e}")

        # Try to import llama.cpp backend (keeping for compatibility)
        try:
            from .llama_cpp_backend import LlamaCppBackend
            cls.register_backend("llama_cpp", LlamaCppBackend)
            cls.register_backend("llama.cpp", LlamaCppBackend)  # Alias
        except ImportError as e:
            logger.debug(f"llama.cpp backend not available: {e}")

        # Try to import Transformers backend
        try:
            from .transformers_backend import TransformersBackend
            cls.register_backend("transformers", TransformersBackend)
            cls.register_backend("hf", TransformersBackend)  # Alias
        except ImportError as e:
            logger.debug(f"Transformers backend not available: {e}")

        # Try to import ExLlama backend
        try:
            from .exllama_backend import ExLlamaBackend
            cls.register_backend("exllama", ExLlamaBackend)
            cls.register_backend("exllamav2", ExLlamaBackend)  # Alias
        except ImportError as e:
            logger.debug(f"ExLlama backend not available: {e}")

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """
        Get list of available backend types

        Returns:
            List of backend names
        """
        if not cls._backends:
            cls._initialize_backends()

        return list(cls._backends.keys())

    @classmethod
    def get_backend_info(cls, backend_type: str) -> dict[str, Any]:
        """
        Get information about a specific backend

        Args:
            backend_type: Backend type

        Returns:
            Backend information dictionary
        """
        if not cls._backends:
            cls._initialize_backends()

        backend_type = backend_type.lower()

        if backend_type not in cls._backends:
            return {"error": f"Unknown backend: {backend_type}"}

        backend_class = cls._backends[backend_type]

        info = {
            "name": backend_type,
            "class": backend_class.__name__,
            "module": backend_class.__module__,
            "supports_streaming": True,  # Most backends support streaming
            "supports_multi_gpu": False  # Default, overridden by backend
        }

        # Try to get more info from the backend class
        try:
            temp_backend = backend_class({})
            info["supports_multi_gpu"] = temp_backend.supports_multi_gpu()
            info["supports_streaming"] = temp_backend.supports_streaming()
        except Exception:
            pass

        return info


def create_backend(backend_type: str, config: dict[str, Any]) -> BaseBackend:
    """
    Convenience function to create a backend

    Args:
        backend_type: Type of backend to create
        config: Backend configuration

    Returns:
        Backend instance
    """
    return BackendFactory.create_backend(backend_type, config)


def get_available_backends() -> list[str]:
    """
    Convenience function to get available backends

    Returns:
        List of backend names
    """
    return BackendFactory.get_available_backends()
