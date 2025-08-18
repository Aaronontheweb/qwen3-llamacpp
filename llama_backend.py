"""
llama.cpp backend integration for Qwen3 multi-GPU server
"""

import logging
import os
import time
import re
from typing import Dict, List, Optional, Any, Generator
from pathlib import Path

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not available. Install with: pip install llama-cpp-python")

from utils.gpu_monitor import get_gpu_monitor
from utils.model_utils import validate_gguf_file, get_model_info_from_gguf

logger = logging.getLogger("qwen3_server.llama_backend")


class LlamaBackend:
    """llama.cpp backend for multi-GPU model inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_path = None
        self.model_config = None
        self.gpu_monitor = get_gpu_monitor()
        
        # llama.cpp settings
        self.llama_settings = {
            "n_gpu_layers": -1,  # Use all available GPUs (default, can be overridden)
            "n_ctx": 262144,     # Context length - 256k tokens (let users decide)
            "n_batch": 512,      # Batch size
            "n_threads": os.cpu_count(),  # Use all CPU threads
            "verbose": False,    # Disable verbose output
            "use_mmap": True,    # Use memory mapping
            "use_mlock": False,  # Don't lock memory
            "offload_kqv": True, # Offload KQV to GPU when available
            # KV cache quantization - use Q8_0 for 2x memory savings with minimal quality loss
            # Options: 0 (F32), 1 (F16), 2 (Q8_0), 3 (Q4_0), 4 (Q4_1), 5 (IQ2_XXS), 6 (IQ2_XS)
            "type_k": 2,  # Q8_0 quantization for keys (8-bit)
            "type_v": 2,  # Q8_0 quantization for values (8-bit)
        }
        
        # Update settings from config
        if "llama_settings" in config:
            self.llama_settings.update(config["llama_settings"])
    
    def _extract_training_context_from_error(self, error_message: str) -> Optional[int]:
        """
        Extract the training context size from llama.cpp error messages
        
        Args:
            error_message: Error message from llama.cpp
            
        Returns:
            Training context size in tokens, or None if not found
        """
        # Look for pattern: n_ctx_train (40960)
        match = re.search(r'n_ctx_train \((\d+)\)', str(error_message))
        if match:
            return int(match.group(1))
        return None
    
    def load_model(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """
        Load a model using llama.cpp - simplified approach that trusts llama.cpp
        
        Args:
            model_path: Path to the GGUF model file
            model_config: Model configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python is not available")
            return False
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            # Unload existing model
            self.unload_model()
            
            logger.info(f"Loading model: {model_path}")
            
            # Prepare llama.cpp settings with sensible defaults
            settings = self.llama_settings.copy()
            
            # Allow override of GPU layers if specified
            if "n_gpu_layers" in model_config:
                settings["n_gpu_layers"] = model_config["n_gpu_layers"]
            
            # Start with the desired context size (or max)
            target_context = model_config.get("max_context_tokens", 131072)
            
            # List of context sizes to try, from largest to smallest
            # We'll try these in order until one works
            context_sizes = [
                target_context,
                131072,  # 128k
                98304,   # 96k
                65536,   # 64k
                49152,   # 48k
                32768,   # 32k
                16384,   # 16k
                8192,    # 8k
                4096     # 4k (minimum)
            ]
            
            # Remove duplicates and sort descending
            context_sizes = sorted(list(set([c for c in context_sizes if c <= target_context])), reverse=True)
            
            logger.info(f"Will try context sizes: {context_sizes}")
            
            # Try loading with progressively smaller context sizes
            start_time = time.time()
            successful_context = None
            
            for context_size in context_sizes:
                settings["n_ctx"] = context_size
                settings["n_batch"] = min(512, context_size // 8)  # Reasonable batch size
                
                logger.info(f"Attempting to load with context size: {context_size} tokens")
                
                try:
                    self.model = Llama(
                        model_path=model_path,
                        **settings
                    )
                    
                    # Success! Model loaded
                    successful_context = context_size
                    load_time = time.time() - start_time
                    
                    self.model_path = model_path
                    self.model_config = model_config
                    
                    logger.info(f"✓ Model loaded successfully with {context_size} token context in {load_time:.2f}s")
                    
                    # Try to get actual context size from model
                    try:
                        actual_context = self.model.n_ctx()
                        logger.info(f"Actual model context: {actual_context} tokens")
                    except:
                        pass
                    
                    # Log memory usage
                    self.gpu_monitor.log_memory_status()
                    
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.debug(f"Failed with {context_size} tokens: {error_msg}")
                    
                    # Clean up failed attempt
                    if hasattr(self, 'model') and self.model is not None:
                        try:
                            del self.model
                        except:
                            pass
                        self.model = None
                    
                    # Check if error mentions a specific context limit
                    training_context = self._extract_training_context_from_error(error_msg)
                    if training_context and training_context < context_size:
                        logger.info(f"Model has training context limit of {training_context} tokens")
                        # Add this as the next size to try
                        if training_context not in context_sizes:
                            remaining = [c for c in context_sizes if c < context_size]
                            context_sizes = [training_context] + remaining
                    
                    # Continue to next smaller size
                    continue
            
            # All attempts failed
            logger.error(f"Failed to load model with any context size. Last tried: {context_sizes[-1]}")
            self.model = None
            self.model_path = None
            self.model_config = None
            return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.model_path = None
            self.model_config = None
            return False
    
    def unload_model(self):
        """Unload the current model"""
        if self.model:
            logger.info("Unloading model")
            try:
                del self.model
                self.model = None
                self.model_path = None
                self.model_config = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("Model unloaded successfully")
                
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if not self.model:
            raise RuntimeError("No model loaded")
        
        try:
            # Set default parameters
            generation_params = {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
            
            # Update with model defaults
            if self.model_config and "default_params" in self.model_config:
                generation_params.update(self.model_config["default_params"])
            
            # Update with user parameters
            generation_params.update(kwargs)
            
            # Generate
            response = self.model(
                prompt,
                **generation_params
            )
            
            return response["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generate text with streaming
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Yields:
            Generated text chunks
        """
        if not self.model:
            raise RuntimeError("No model loaded")
        
        try:
            # Set default parameters
            generation_params = {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|endoftext|>"],
                "stream": True
            }
            
            # Update with model defaults
            if self.model_config and "default_params" in self.model_config:
                generation_params.update(self.model_config["default_params"])
            
            # Update with user parameters
            generation_params.update(kwargs)
            
            # Generate with streaming
            for chunk in self.model(
                prompt,
                **generation_params
            ):
                if chunk["choices"][0]["finish_reason"] is not None:
                    break
                
                text_chunk = chunk["choices"][0]["text"]
                if text_chunk:
                    yield text_chunk
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {}
        
        try:
            # Get basic model info
            info = {
                "model_path": self.model_path,
                "model_config": self.model_config,
                "context_length": self.model.n_ctx(),
                "vocab_size": self.model.n_vocab(),
                "embedding_size": self.model.n_embd(),
            }
            
            # Get file info
            if self.model_path:
                file_info = get_model_info_from_gguf(self.model_path)
                if file_info:
                    info.update(file_info)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get backend status
        
        Returns:
            Status dictionary
        """
        status = {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "model_config": self.model_config,
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "gpu_memory": self.gpu_monitor.get_memory_usage_summary()
        }
        
        if self.model:
            status["model_info"] = self._get_model_info()
        
        return status
    
    def cleanup(self):
        """Clean up resources"""
        self.unload_model()
        self.gpu_monitor.cleanup()


class ModelManager:
    """Manager for model loading and switching"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = LlamaBackend(config)
        self.current_model_id = None
        
    def load_model_by_id(self, model_id: str) -> bool:
        """
        Load a model by its ID
        
        Args:
            model_id: Model ID from configuration
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.config["models"]:
            logger.error(f"Unknown model ID: {model_id}")
            return False
        
        model_config = self.config["models"][model_id]
        model_name = model_config["name"]
        download_path = self.config["download_path"]
        
        # Construct model path and find the actual GGUF file
        model_dir = os.path.join(download_path, model_name.replace("/", "_"))
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            return False
        
        # Look for any .gguf file in the model directory (including subdirectories)
        model_path = None
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.gguf'):
                    test_path = os.path.join(root, file)
                    # Check if file is not empty
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        model_path = test_path
                        logger.info(f"Found model file: {model_path}")
                        break
            if model_path:
                break
        
        if not model_path:
            logger.error(f"No model file found in {model_dir}")
            return False
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        
        # Load model
        success = self.backend.load_model(model_path, model_config)
        if success:
            self.current_model_id = model_id
            logger.info(f"Switched to model: {model_id}")
        
        return success
    
    def get_current_model(self) -> Optional[str]:
        """Get current model ID"""
        return self.current_model_id
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        for model_id, model_config in self.config["models"].items():
            model_name = model_config["name"]
            download_path = self.config["download_path"]
            model_path = os.path.join(download_path, model_name.replace("/", "_"), "model.gguf")
            
            models.append({
                "id": model_id,
                "name": model_name,
                "config": model_config,
                "downloaded": os.path.exists(model_path),
                "path": model_path if os.path.exists(model_path) else None,
                "is_current": model_id == self.current_model_id
            })
        
        return models
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            "current_model": self.current_model_id,
            "backend_status": self.backend.get_status(),
            "available_models": self.get_available_models()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.backend.cleanup()


# Global model manager instance
model_manager = None


def get_model_manager(config: Dict[str, Any]) -> ModelManager:
    """Get or create the global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(config)
    return model_manager


def get_llama_backend(config: Dict[str, Any]) -> LlamaBackend:
    """Get the llama.cpp backend instance"""
    manager = get_model_manager(config)
    return manager.backend 