"""
llama.cpp backend integration for Qwen3 multi-GPU server
"""

import logging
import os
import time
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
            "n_gpu_layers": -1,  # Use all available GPUs
            "n_ctx": 131072,     # Max context length - 128k tokens
            "n_batch": 512,      # Batch size
            "n_threads": os.cpu_count(),  # Use all CPU threads
            "verbose": False,    # Disable verbose output
            "use_mmap": True,    # Use memory mapping
            "use_mlock": False,  # Don't lock memory
            "cont_batching": True,  # Enable continuous batching for dynamic growth
        }
        
        # Update settings from config
        if "llama_settings" in config:
            self.llama_settings.update(config["llama_settings"])
    
    def load_model(self, model_path: str, model_config: Dict[str, Any]) -> bool:
        """
        Load a model using llama.cpp
        
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
        
        if not validate_gguf_file(model_path):
            logger.error(f"Invalid GGUF file: {model_path}")
            return False
        
        try:
            # Unload existing model
            self.unload_model()
            
            logger.info(f"Loading model: {model_path}")
            
            # Check memory requirements
            memory_check = self._check_memory_requirements(model_config)
            if not memory_check[0]:  # memory_check returns (fits, details)
                logger.error(f"Model does not fit in available GPU memory: {memory_check[1]}")
                return False
            
            # Prepare llama.cpp settings
            settings = self.llama_settings.copy()
            
            # Dynamically estimate context window based on available GPU memory
            total_memory_mb, available_memory_mb = self.gpu_monitor.get_total_gpu_memory()
            logger.info(f"GPU Memory - Total: {total_memory_mb}MB, Available: {available_memory_mb}MB")

            # Heuristic: KV-cache memory per 1k tokens (MB)
            # Approximate KV-cache memory cost per 1k tokens (MB).  Values are deliberately high-side
            # to stay within GPU memory when multiple tensors are resident.
            kv_mb_per_1k = {
                "30B": 800,    # ~0.8 GB / 1k tokens (reduced from 1.3GB - was too conservative)
                "14B": 400,    # ~0.4 GB / 1k tokens (reduced from 0.6GB)
                "7B": 200,     # ~0.2 GB / 1k tokens (reduced from 0.27GB)
                "4B": 100      # ~0.1 GB / 1k tokens (reduced from 0.13GB)
            }
            model_size = model_config.get("size", "7B")
            kv_per_1k = kv_mb_per_1k.get(model_size, 300)  # fallback 0.3 GB per 1k tokens

            # Reserve 70% of free memory for KV-cache to leave room for weights & overhead
            kv_budget_mb = int(available_memory_mb * 0.7)
            logger.info(f"KV-cache budget: {kv_budget_mb}MB (from {available_memory_mb}MB available)")
            max_tokens_estimate = int((kv_budget_mb / kv_per_1k) * 1000)  # tokens
            # Round down to nearest 1024 for safety
            max_tokens_estimate = (max_tokens_estimate // 1024) * 1024

            # Use the full available context - let it grow as needed
            estimated_ctx = 131072

            settings["n_ctx"] = estimated_ctx
            logger.info(f"Estimated context window based on GPU memory: {estimated_ctx} tokens (model size {model_size})")
            
            # Adjust batch size crudely based on model size
            if model_size == "30B":
                settings["n_batch"] = 256
            elif model_size == "14B":
                settings["n_batch"] = 512
            else:
                settings["n_batch"] = 1024
            
            logger.info(f"Loading model with settings: n_ctx={settings['n_ctx']}, n_batch={settings['n_batch']}")
            
            # Load model with fallback to smaller context windows
            start_time = time.time()
            
            # Try loading with initial settings
            try:
                self.model = Llama(
                    model_path=model_path,
                    **settings
                )
                load_time = time.time() - start_time
                
                self.model_path = model_path
                self.model_config = model_config
                
                logger.info(f"Model loaded successfully in {load_time:.2f}s")
                logger.info(f"Model info: {self._get_model_info()}")
                
                # Log GPU memory status
                self.gpu_monitor.log_memory_status()
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load with context window {settings['n_ctx']}: {e}")
                
                # Clean up failed attempt
                if hasattr(self, 'model') and self.model is not None:
                    try:
                        del self.model
                    except:
                        pass
                    self.model = None
                
                # Try with smaller context window for 30B models
                if model_config.get("size") == "30B" and settings["n_ctx"] > 16384:
                    logger.info("Retrying with smaller context window (16k)...")
                    settings["n_ctx"] = 16384
                    settings["n_batch"] = 128
                    
                    try:
                        self.model = Llama(
                            model_path=model_path,
                            **settings
                        )
                        load_time = time.time() - start_time
                        
                        self.model_path = model_path
                        self.model_config = model_config
                        
                        logger.info(f"Model loaded successfully with 16k context in {load_time:.2f}s")
                        logger.info(f"Model info: {self._get_model_info()}")
                        
                        # Log GPU memory status
                        self.gpu_monitor.log_memory_status()
                        
                        return True
                        
                    except Exception as e2:
                        logger.error(f"Failed to load even with 16k context: {e2}")
                        # Clean up second failed attempt
                        if hasattr(self, 'model') and self.model is not None:
                            try:
                                del self.model
                            except:
                                pass
                            self.model = None
                
                # If we get here, all attempts failed
                logger.error(f"Failed to load model: {e}")
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
    
    def _check_memory_requirements(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if model fits in available GPU memory
        
        Args:
            model_config: Model configuration
            
        Returns:
            Memory check results
        """
        required_memory_gb = model_config.get("memory_estimate_gb", 0)
        return self.gpu_monitor.check_model_fits(required_memory_gb)
    
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