"""
Unified model manager for different backend types
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from backends.factory import BackendFactory, get_available_backends
from utils.gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for model loading and switching with multiple backend support"""
    
    def __init__(self, config_path: str = "models_config.json"):
        """
        Initialize model manager
        
        Args:
            config_path: Path to models configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.backend = None
        self.current_model_id = None
        self.current_backend_type = None
        self.gpu_monitor = get_gpu_monitor()
        
        # Initialize backend
        self._initialize_backend()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "backend": {
                "type": "vllm",
                "fallback": "llama_cpp",
                "vllm_config": {
                    "tensor_parallel_size": "auto",
                    "gpu_memory_utilization": 0.90
                }
            },
            "models": {},
            "download_path": "./models",
            "cache_dir": "./cache"
        }
    
    def _initialize_backend(self):
        """Initialize the backend based on configuration"""
        backend_config = self.config.get("backend", {})
        backend_type = backend_config.get("type", "vllm")
        fallback_type = backend_config.get("fallback", "llama_cpp")
        
        # Get available backends
        available = get_available_backends()
        logger.info(f"Available backends: {available}")
        
        # Try primary backend
        if backend_type in available:
            self._create_backend(backend_type)
        # Try fallback backend
        elif fallback_type in available:
            logger.warning(f"Primary backend '{backend_type}' not available. Using fallback: {fallback_type}")
            self._create_backend(fallback_type)
        # Use any available backend
        elif available:
            first_available = available[0]
            logger.warning(f"Using first available backend: {first_available}")
            self._create_backend(first_available)
        else:
            raise RuntimeError("No backends available. Please install vLLM or llama-cpp-python.")
    
    def _create_backend(self, backend_type: str):
        """Create backend instance"""
        backend_config = self.config.get("backend", {})
        
        # Get backend-specific config
        if backend_type == "vllm":
            specific_config = backend_config.get("vllm_config", {})
        elif backend_type in ["llama_cpp", "llama.cpp"]:
            specific_config = backend_config.get("llama_cpp_config", {})
        else:
            specific_config = {}
        
        # Handle auto tensor_parallel_size for vLLM
        if backend_type == "vllm" and specific_config.get("tensor_parallel_size") == "auto":
            gpu_count = self.gpu_monitor.device_count if hasattr(self.gpu_monitor, 'device_count') else 1
            specific_config["tensor_parallel_size"] = max(1, gpu_count)
            logger.info(f"Auto-detected tensor_parallel_size: {specific_config['tensor_parallel_size']}")
        
        # Merge with global settings
        config = {
            "download_path": self.config.get("download_path", "./models"),
            "cache_dir": self.config.get("cache_dir", "./cache"),
            **specific_config
        }
        
        try:
            self.backend = BackendFactory.create_backend(backend_type, config)
            self.current_backend_type = backend_type
            logger.info(f"Successfully created {backend_type} backend")
        except Exception as e:
            logger.error(f"Failed to create {backend_type} backend: {e}")
            raise
    
    def switch_backend(self, backend_type: str) -> bool:
        """
        Switch to a different backend type
        
        Args:
            backend_type: Backend type to switch to
            
        Returns:
            True if successful
        """
        if backend_type == self.current_backend_type:
            logger.info(f"Already using {backend_type} backend")
            return True
        
        # Unload current model if any
        if self.backend and self.current_model_id:
            self.backend.unload_model()
            self.current_model_id = None
        
        # Clean up old backend
        if self.backend:
            self.backend.cleanup()
            self.backend = None
        
        # Create new backend
        try:
            self._create_backend(backend_type)
            return True
        except Exception as e:
            logger.error(f"Failed to switch to {backend_type}: {e}")
            return False
    
    def load_model_by_id(self, model_id: str, backend_type: Optional[str] = None) -> bool:
        """
        Load a model by its ID
        
        Args:
            model_id: Model ID from configuration
            backend_type: Optional backend type to use
            
        Returns:
            True if successful
        """
        if model_id not in self.config.get("models", {}):
            logger.error(f"Unknown model ID: {model_id}")
            return False
        
        model_config = self.config["models"][model_id]
        
        # Switch backend if requested
        if backend_type and backend_type != self.current_backend_type:
            if not self.switch_backend(backend_type):
                return False
        
        # Determine model path based on backend type
        model_path = self._get_model_path(model_config)
        
        if not model_path:
            logger.error(f"Could not determine model path for {model_id}")
            return False
        
        # Load model
        logger.info(f"Loading model {model_id} with {self.current_backend_type} backend")
        success = self.backend.load_model(model_path, model_config)
        
        if success:
            self.current_model_id = model_id
            logger.info(f"Successfully loaded model: {model_id}")
        else:
            logger.error(f"Failed to load model: {model_id}")
        
        return success
    
    def _get_model_path(self, model_config: Dict[str, Any]) -> Optional[str]:
        """
        Get the appropriate model path based on backend type
        
        Args:
            model_config: Model configuration
            
        Returns:
            Model path or None
        """
        # For vLLM, use HuggingFace model name/path
        if self.current_backend_type == "vllm":
            # Check if we have a local HF model
            model_name = model_config.get("name")
            if model_name:
                # Check local directory first
                download_path = self.config.get("download_path", "./models")
                local_path = os.path.join(download_path, model_name.replace("/", "_"))
                
                if os.path.exists(local_path):
                    logger.info(f"Using local HF model: {local_path}")
                    return local_path
                else:
                    # Use HuggingFace model ID for auto-download
                    logger.info(f"Using HuggingFace model: {model_name}")
                    return model_name
        
        # For llama.cpp, use GGUF file
        elif self.current_backend_type in ["llama_cpp", "llama.cpp"]:
            gguf_name = model_config.get("gguf_name", model_config.get("name"))
            if gguf_name:
                download_path = self.config.get("download_path", "./models")
                model_dir = os.path.join(download_path, gguf_name.replace("/", "_"))
                
                # Look for GGUF file
                if os.path.exists(model_dir):
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            if file.endswith('.gguf'):
                                gguf_path = os.path.join(root, file)
                                logger.info(f"Found GGUF file: {gguf_path}")
                                return gguf_path
        
        # Fallback to name field
        return model_config.get("name")
    
    def get_current_model(self) -> Optional[str]:
        """Get current model ID"""
        return self.current_model_id
    
    def get_current_backend(self) -> Optional[str]:
        """Get current backend type"""
        return self.current_backend_type
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        for model_id, model_config in self.config.get("models", {}).items():
            model_info = {
                "id": model_id,
                "name": model_config.get("name"),
                "type": model_config.get("type"),
                "size": model_config.get("size"),
                "description": model_config.get("description"),
                "max_context_tokens": model_config.get("max_context_tokens"),
                "is_current": model_id == self.current_model_id,
                "backend_compatible": self._is_model_compatible(model_config)
            }
            models.append(model_info)
        
        return models
    
    def _is_model_compatible(self, model_config: Dict[str, Any]) -> Dict[str, bool]:
        """Check which backends are compatible with a model"""
        compatible = {}
        
        # vLLM needs HuggingFace models
        compatible["vllm"] = bool(model_config.get("name"))
        
        # llama.cpp needs GGUF files
        compatible["llama_cpp"] = bool(model_config.get("gguf_name") or 
                                      (model_config.get("name") and "GGUF" in model_config.get("name", "")))
        
        return compatible
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using current model"""
        if not self.backend or not self.current_model_id:
            raise RuntimeError("No model loaded")
        
        return self.backend.generate(prompt, **kwargs)
    
    def generate_stream(self, prompt: str, **kwargs):
        """Generate text with streaming"""
        if not self.backend or not self.current_model_id:
            raise RuntimeError("No model loaded")
        
        return self.backend.generate_stream(prompt, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        status = {
            "current_model": self.current_model_id,
            "current_backend": self.current_backend_type,
            "available_backends": get_available_backends(),
            "available_models": self.get_available_models(),
            "config": self.config
        }
        
        if self.backend:
            status["backend_status"] = self.backend.get_status()
        
        return status
    
    def cleanup(self):
        """Clean up resources"""
        if self.backend:
            self.backend.cleanup()
            self.backend = None
        self.current_model_id = None
        self.current_backend_type = None


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(config_path: str = "models_config.json") -> ModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config_path)
    return _model_manager