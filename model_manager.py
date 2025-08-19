"""
Unified model manager for different backend types
"""

import json
import logging
import os
import sys
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backends.base import BaseBackend
from backends.factory import BackendFactory, get_available_backends
from utils.gpu_monitor import get_gpu_monitor
from utils.logging_config import setup_logging

console = Console()

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
        self.backend: Optional[BaseBackend] = None
        self.current_model_id: Optional[str] = None
        self.current_backend_type: Optional[str] = None
        self.gpu_monitor = get_gpu_monitor()

        # Initialize backend
        self._initialize_backend()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()

        try:
            with open(self.config_path) as f:
                config_data = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return dict(config_data) if isinstance(config_data, dict) else self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration"""
        return {
            "backend": {
                "type": "vllm",
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
        fallback_type = backend_config.get("fallback", None)

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
            raise RuntimeError("No backends available. Please install vLLM.")

    def _create_backend(self, backend_type: str):
        """Create backend instance"""
        backend_config = self.config.get("backend", {})

        # Get backend-specific config
        specific_config = backend_config.get("vllm_config", {}) if backend_type == "vllm" else {}

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
        if backend_type and backend_type != self.current_backend_type and not self.switch_backend(backend_type):
            return False

        # Determine model path based on backend type
        model_path = self._get_model_path(model_config)

        if not model_path:
            logger.error(f"Could not determine model path for {model_id}")
            return False

        # Load model
        logger.info(f"Loading model {model_id} with {self.current_backend_type} backend")
        if not self.backend:
            logger.error("No backend available")
            return False
        success = self.backend.load_model(model_path, model_config)

        if success:
            self.current_model_id = model_id
            logger.info(f"Successfully loaded model: {model_id}")
            return True
        else:
            logger.error(f"Failed to load model: {model_id}")
            return False

    def _get_model_path(self, model_config: dict[str, Any]) -> Optional[str]:
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
            if model_name and isinstance(model_name, str):
                # Check local directory first
                download_path = self.config.get("download_path", "./models")
                local_path = os.path.join(download_path, model_name.replace("/", "_"))

                if os.path.exists(local_path):
                    logger.info(f"Using local HF model: {local_path}")
                    return str(local_path)
                else:
                    # Use HuggingFace model ID for auto-download
                    logger.info(f"Using HuggingFace model: {model_name}")
                    return str(model_name)

        # Only vLLM backend supported

        # Fallback to name field
        name = model_config.get("name")
        return str(name) if name and isinstance(name, str) else None

    def get_current_model(self) -> Optional[str]:
        """Get current model ID"""
        return self.current_model_id

    def get_current_backend(self) -> Optional[str]:
        """Get current backend type"""
        return self.current_backend_type

    def get_available_models(self) -> list[dict[str, Any]]:
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

    def _is_model_compatible(self, model_config: dict[str, Any]) -> dict[str, bool]:
        """Check which backends are compatible with a model"""
        compatible = {}

        # vLLM needs HuggingFace models
        compatible["vllm"] = bool(model_config.get("name"))

        # Only vLLM backend supported

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

    def get_status(self) -> dict[str, Any]:
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


class ModelManagerCLI:
    """Command-line interface for model management"""

    def __init__(self, config_path: str = "models_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.gpu_monitor = get_gpu_monitor()
        self.manager = get_model_manager(config_path)

        # Set up logging
        log_level = self.config.get("server", {}).get("log_level", "INFO")
        self.logger = setup_logging(log_level)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path) as f:
                data = json.load(f)
                return dict(data) if isinstance(data, dict) else {}
        except FileNotFoundError:
            console.print(f"[red]Configuration file not found: {self.config_path}[/red]")
            sys.exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in configuration file: {e}[/red]")
            sys.exit(1)

    def _save_config(self):
        """Save configuration file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving configuration: {e}[/red]")

    def list_models(self, show_details: bool = False):
        """List available models"""
        models = self.config["models"]

        table = Table(title="Available Models")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Type", style="blue")
        table.add_column("Quantization", style="white")
        table.add_column("Status", style="magenta")
        table.add_column("Backend", style="red")

        if show_details:
            table.add_column("Description", style="white")

        for model_id, model_config in models.items():
            # Check if it's the active model
            active = model_id == self.config.get("active_model")

            # Check backend compatibility
            backend_info = []
            if model_config.get("name"):
                backend_info.append("vLLM")
            # Only vLLM backend supported

            status = []
            if active:
                status.append("✓ Active")

            status_text = " | ".join(status) if status else "Available"
            backend_text = ", ".join(backend_info) if backend_info else "None"

            row = [
                model_id,
                model_config.get("name", "N/A"),
                model_config.get("size", "N/A"),
                model_config.get("type", "N/A"),
                model_config.get("quantization", "Auto"),
                status_text,
                backend_text
            ]

            if show_details:
                row.append(model_config.get("description", "N/A"))

            table.add_row(*row)

        console.print(table)

    def switch_model(self, model_id: str):
        """Switch to a different model"""
        models = self.config["models"]

        if model_id not in models:
            console.print(f"[red]Model '{model_id}' not found in configuration[/red]")
            available_models = list(models.keys())
            console.print(f"Available models: {', '.join(available_models)}")
            return

        # Update active model in config
        self.config["active_model"] = model_id
        self._save_config()

        console.print(f"[green]Switched active model to: {model_id}[/green]")
        console.print(f"Model: {models[model_id]['name']}")
        console.print("[yellow]Note: Restart the server for changes to take effect[/yellow]")

    def info_model(self, model_id: str):
        """Show detailed information about a model"""
        models = self.config["models"]

        if model_id not in models:
            console.print(f"[red]Model '{model_id}' not found[/red]")
            return

        model_config = models[model_id]

        info_text = Text()
        info_text.append(f"Model ID: {model_id}\n", style="cyan")
        info_text.append(f"Name: {model_config.get('name', 'N/A')}\n", style="white")
        info_text.append(f"GGUF Name: {model_config.get('gguf_name', 'N/A')}\n", style="white")
        info_text.append(f"Type: {model_config.get('type', 'N/A')}\n", style="white")
        info_text.append(f"Size: {model_config.get('size', 'N/A')}\n", style="white")
        info_text.append(f"Description: {model_config.get('description', 'N/A')}\n", style="white")
        info_text.append(f"Max Context: {model_config.get('max_context_tokens', 'N/A')} tokens\n", style="white")
        info_text.append(f"Quantization: {model_config.get('quantization', 'Auto')}\n", style="white")

        # Default parameters
        default_params = model_config.get("default_params", {})
        if default_params:
            info_text.append("\nDefault Parameters:\n", style="cyan")
            for key, value in default_params.items():
                info_text.append(f"  {key}: {value}\n", style="white")

        # Backend compatibility
        info_text.append("\nBackend Compatibility:\n", style="cyan")
        if model_config.get("name"):
            info_text.append("  ✓ vLLM (HuggingFace format)\n", style="green")
        if not model_config.get("name"):
            info_text.append("  ✗ No compatible formats found\n", style="red")

        panel = Panel(info_text, title=f"Model Information: {model_id}", border_style="blue")
        console.print(panel)

    def status(self):
        """Show system status"""
        # GPU status
        gpu_summary = self.gpu_monitor.get_memory_usage_summary()

        status_text = Text()
        status_text.append("GPU Status:\n", style="cyan")
        status_text.append(f"  Total Memory: {gpu_summary['total_memory_mb'] / 1024:.1f}GB\n", style="white")
        status_text.append(f"  Used Memory: {gpu_summary['used_memory_mb'] / 1024:.1f}GB\n", style="white")
        status_text.append(f"  Available Memory: {gpu_summary['available_memory_mb'] / 1024:.1f}GB\n", style="white")
        status_text.append(f"  Utilization: {gpu_summary['utilization_percent']:.1f}%\n", style="white")
        status_text.append(f"  GPU Count: {gpu_summary['gpu_count']}\n", style="white")

        # Active model
        active_model = self.config.get("active_model")
        status_text.append(f"\nActive Model: {active_model or 'None'}\n", style="cyan")

        # Backend status
        status_text.append("\nBackend Configuration:\n", style="cyan")
        backend_config = self.config.get("backend", {})
        status_text.append(f"  Primary: {backend_config.get('type', 'vllm')}\n", style="white")
        if backend_config.get('fallback'):
            status_text.append(f"  Fallback: {backend_config.get('fallback')}\n", style="white")

        # Available backends
        from backends.factory import get_available_backends
        available_backends = get_available_backends()
        status_text.append(f"  Available: {', '.join(available_backends)}\n", style="white")

        panel = Panel(status_text, title="System Status", border_style="blue")
        console.print(panel)


# CLI setup
@click.group()
@click.option('--config', default='models_config.json', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Qwen3 Model Manager CLI"""
    ctx.obj = ModelManagerCLI(config)


@cli.command("list")
@click.option('--details', is_flag=True, help='Show detailed information')
@click.pass_obj
def list_models_cmd(manager, details):
    """List available models"""
    manager.list_models(show_details=details)


@cli.command()
@click.argument('model_id')
@click.pass_obj
def switch(manager, model_id):
    """Switch to a different model"""
    manager.switch_model(model_id)


@cli.command()
@click.argument('model_id')
@click.pass_obj
def info(manager, model_id):
    """Show detailed information about a model"""
    manager.info_model(model_id)


@cli.command()
@click.pass_obj
def status(manager):
    """Show system status"""
    manager.status()


if __name__ == "__main__":
    cli()
