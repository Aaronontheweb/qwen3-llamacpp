"""
Base backend interface for model serving
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """Abstract base class for all backend implementations"""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize backend with configuration

        Args:
            config: Backend configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_path = None
        self.model_config = None

    @abstractmethod
    def load_model(self, model_path: str, model_config: dict[str, Any]) -> bool:
        """
        Load a model for inference

        Args:
            model_path: Path to the model
            model_config: Model-specific configuration

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the current model and free resources"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Yields:
            Generated text chunks
        """
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """
        Get backend status and information

        Returns:
            Status dictionary
        """
        pass

    def cleanup(self):
        """Clean up resources"""
        self.unload_model()

    def supports_streaming(self) -> bool:
        """Check if backend supports streaming generation"""
        return True

    def supports_multi_gpu(self) -> bool:
        """Check if backend supports multi-GPU operation"""
        return False

    def get_context_window(self) -> int:
        """Get the maximum context window size"""
        if self.model_config:
            return self.model_config.get("max_context_tokens", 4096)
        return 4096

    def validate_request(self, prompt: str, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate a generation request

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.model:
            return False, "No model loaded"

        if not prompt:
            return False, "Empty prompt"

        # Check token limits if available
        max_tokens = kwargs.get("max_tokens", 2048)
        context_window = self.get_context_window()

        # Basic validation - can be extended by subclasses
        if max_tokens > context_window:
            return False, f"max_tokens ({max_tokens}) exceeds context window ({context_window})"

        return True, None
