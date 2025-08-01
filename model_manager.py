#!/usr/bin/env python3
"""
Model management CLI for Qwen3 multi-GPU server
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm

from utils.logging_config import setup_logging
from utils.gpu_monitor import get_gpu_monitor
from utils.model_utils import (
    validate_model_config, get_model_file_size, estimate_download_time,
    format_file_size, get_model_path, is_model_downloaded,
    get_downloaded_models, cleanup_incomplete_downloads
)

console = Console()


class ModelManagerCLI:
    """Command-line interface for model management"""
    
    def __init__(self, config_path: str = "models_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.gpu_monitor = get_gpu_monitor()
        
        # Set up logging
        log_level = self.config.get("server", {}).get("log_level", "INFO")
        self.logger = setup_logging(log_level)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
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
        table.add_column("Status", style="magenta")
        table.add_column("Memory (GB)", style="red")
        
        if show_details:
            table.add_column("Description", style="white")
        
        for model_id, model_config in models.items():
            # Check if model is downloaded
            model_name = model_config["name"]
            download_path = self.config["download_path"]
            downloaded = is_model_downloaded(model_name, download_path)
            
            # Check if it's the active model
            active = model_id == self.config.get("active_model")
            
            status = []
            if downloaded:
                status.append("✓ Downloaded")
            if active:
                status.append("✓ Active")
            
            status_text = " | ".join(status) if status else "Not downloaded"
            
            row = [
                model_id,
                model_name,
                model_config["size"],
                model_config["type"],
                status_text,
                str(model_config["memory_estimate_gb"])
            ]
            
            if show_details:
                row.append(model_config["description"])
            
            table.add_row(*row)
        
        console.print(table)
    
    def download_model(self, model_id: str, force: bool = False):
        """Download a model"""
        if model_id not in self.config["models"]:
            console.print(f"[red]Unknown model ID: {model_id}[/red]")
            return False
        
        model_config = self.config["models"][model_id]
        model_name = model_config["name"]
        download_path = self.config["download_path"]
        
        # Check if already downloaded
        if is_model_downloaded(model_name, download_path) and not force:
            console.print(f"[yellow]Model {model_id} is already downloaded[/yellow]")
            return True
        
        # Create download directory
        model_dir = get_model_path(model_name, download_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get file size
        file_size = get_model_file_size(model_name)
        if file_size:
            console.print(f"File size: {format_file_size(file_size)}")
            estimated_time = estimate_download_time(file_size)
            if estimated_time > 0:
                console.print(f"Estimated download time: {estimated_time:.1f} minutes")
        
        # Check memory requirements
        memory_check = self.gpu_monitor.check_model_fits(model_config["memory_estimate_gb"])
        if not memory_check[0]:  # memory_check returns (fits, details)
            console.print(f"[red]Warning: Model requires {model_config['memory_estimate_gb']}GB but only "
                         f"{memory_check[1]['available_memory_mb']/1024:.1f}GB available[/red]")
            if not click.confirm("Continue with download anyway?"):
                return False
        
        # Download model
        try:
            from huggingface_hub import hf_hub_download
            
            console.print(f"[green]Downloading {model_name}...[/green]")
            
            # Download the model file - use Q4_K_M for good balance of size and quality
            from huggingface_hub import hf_hub_download
            
            # Find available GGUF files and select the best quality that fits in GPU memory
            from huggingface_hub import list_repo_files, hf_hub_url
            import requests
            
            files = list_repo_files(model_name)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise Exception(f"No GGUF files found in {model_name}")
            
            # Get available GPU memory
            total_memory, available_memory = self.gpu_monitor.get_total_gpu_memory()
            available_memory_gb = available_memory / 1024
            
            console.print(f"[blue]Available GPU memory: {available_memory_gb:.1f}GB[/blue]")
            
            # Get file sizes for all GGUF files
            file_sizes = {}
            for gguf_file in gguf_files:
                try:
                    # Get the download URL to check file size
                    url = hf_hub_url(repo_id=model_name, filename=gguf_file)
                    response = requests.head(url, allow_redirects=True)
                    if response.status_code == 200:
                        file_size_gb = int(response.headers.get('content-length', 0)) / (1024**3)
                        file_sizes[gguf_file] = file_size_gb
                        console.print(f"[dim]  {gguf_file}: {file_size_gb:.1f}GB[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not get size for {gguf_file}: {e}[/yellow]")
                    # Estimate size based on filename
                    if 'Q2_K' in gguf_file:
                        file_sizes[gguf_file] = 8.0  # Rough estimate
                    elif 'Q3_K' in gguf_file:
                        file_sizes[gguf_file] = 10.0
                    elif 'Q4_K' in gguf_file:
                        file_sizes[gguf_file] = 15.0
                    elif 'Q5_K' in gguf_file:
                        file_sizes[gguf_file] = 20.0
                    elif 'Q6_K' in gguf_file:
                        file_sizes[gguf_file] = 25.0
                    elif 'Q8_0' in gguf_file or 'BF16' in gguf_file:
                        file_sizes[gguf_file] = 35.0
                    else:
                        file_sizes[gguf_file] = 15.0  # Default estimate
            
            # Sort files by quality (best to worst) and find the best that fits
            # Quality order: Q4_K > Q5_K > Q6_K > Q8_0 > BF16 (avoid BF16 for most setups)
            quality_order = ['Q4_K', 'Q5_K', 'Q6_K', 'Q8_0', 'BF16']
            
            selected_file = None
            selected_quality = None
            
            for quality in quality_order:
                matching_files = [f for f in gguf_files if quality in f]
                for file in matching_files:
                    if file in file_sizes:
                        file_size_gb = file_sizes[file]
                        # Check if file fits in available memory (with 2GB buffer for safety)
                        if file_size_gb <= (available_memory_gb - 2.0):
                            selected_file = file
                            selected_quality = quality
                            break
                if selected_file:
                    break
            
            # If no file fits, select the smallest available
            if not selected_file:
                smallest_file = min(file_sizes.items(), key=lambda x: x[1])
                selected_file = smallest_file[0]
                selected_quality = "smallest"
                console.print(f"[yellow]Warning: No file fits in memory. Selecting smallest: {selected_file}[/yellow]")
            
            filename = selected_file
            file_size = file_sizes.get(filename, "unknown")
            console.print(f"[green]Selected: {filename} ({file_size:.1f}GB, {selected_quality} quality)[/green]")
            
            # Download the specific GGUF file
            model_file = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            console.print(f"[green]✓ Model {model_id} downloaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")
            return False
    
    def switch_model(self, model_id: str):
        """Switch to a different model"""
        if model_id not in self.config["models"]:
            console.print(f"[red]Unknown model ID: {model_id}[/red]")
            return False
        
        model_config = self.config["models"][model_id]
        model_name = model_config["name"]
        download_path = self.config["download_path"]
        
        # Check if model is downloaded
        if not is_model_downloaded(model_name, download_path):
            console.print(f"[red]Model {model_id} is not downloaded[/red]")
            if click.confirm("Download it now?"):
                if not self.download_model(model_id):
                    return False
            else:
                return False
        
        # Update active model
        self.config["active_model"] = model_id
        self._save_config()
        
        console.print(f"[green]✓ Switched to model: {model_id}[/green]")
        return True
    
    def info_model(self, model_id: str):
        """Show detailed information about a model"""
        if model_id not in self.config["models"]:
            console.print(f"[red]Unknown model ID: {model_id}[/red]")
            return
        
        model_config = self.config["models"][model_id]
        model_name = model_config["name"]
        download_path = self.config["download_path"]
        
        # Create info panel
        info_text = Text()
        info_text.append(f"ID: {model_id}\n", style="cyan")
        info_text.append(f"Name: {model_name}\n", style="green")
        info_text.append(f"Size: {model_config['size']}\n", style="yellow")
        info_text.append(f"Type: {model_config['type']}\n", style="blue")
        info_text.append(f"Quantization: {model_config['quantization']}\n", style="magenta")
        info_text.append(f"Memory Estimate: {model_config['memory_estimate_gb']}GB\n", style="red")
        info_text.append(f"Recommended GPUs: {model_config['recommended_gpus']}\n", style="white")
        info_text.append(f"Description: {model_config['description']}\n", style="white")
        
        # Check download status
        downloaded = is_model_downloaded(model_name, download_path)
        if downloaded:
            info_text.append("Status: ✓ Downloaded\n", style="green")
            
            # Get file info
            model_path = get_model_path(model_name, download_path)
            gguf_file = os.path.join(model_path, "model.gguf")
            if os.path.exists(gguf_file):
                file_size = os.path.getsize(gguf_file)
                info_text.append(f"File Size: {format_file_size(file_size)}\n", style="white")
        else:
            info_text.append("Status: Not downloaded\n", style="red")
        
        # Check if active
        active = model_id == self.config.get("active_model")
        if active:
            info_text.append("Active: ✓ Yes\n", style="green")
        else:
            info_text.append("Active: No\n", style="white")
        
        panel = Panel(info_text, title=f"Model Information: {model_id}", border_style="blue")
        console.print(panel)
    
    def check_model(self, model_id: str):
        """Check if a model can fit in available memory"""
        if model_id not in self.config["models"]:
            console.print(f"[red]Unknown model ID: {model_id}[/red]")
            return
        
        model_config = self.config["models"][model_id]
        required_memory_gb = model_config["memory_estimate_gb"]
        
        # Check memory
        memory_check = self.gpu_monitor.check_model_fits(required_memory_gb)
        
        # Create status panel
        status_text = Text()
        status_text.append(f"Model: {model_id}\n", style="cyan")
        status_text.append(f"Required Memory: {required_memory_gb}GB\n", style="red")
        status_text.append(f"Available Memory: {memory_check['available_memory_mb']/1024:.1f}GB\n", style="green")
        status_text.append(f"Total Memory: {memory_check['total_memory_mb']/1024:.1f}GB\n", style="yellow")
        
        if memory_check["fits"]:
            status_text.append("Status: ✓ Fits in memory\n", style="green")
        else:
            status_text.append("Status: ✗ Does not fit\n", style="red")
        
        # GPU details
        status_text.append(f"\nGPU Details:\n", style="white")
        for gpu in memory_check["gpu_details"]:
            status_text.append(f"  GPU {gpu['index']} ({gpu['name']}): "
                             f"{gpu['free_memory_mb']/1024:.1f}GB free / "
                             f"{gpu['total_memory_mb']/1024:.1f}GB total\n", style="white")
        
        panel = Panel(status_text, title="Memory Check", border_style="blue")
        console.print(panel)
    
    def cleanup(self):
        """Clean up incomplete downloads"""
        download_path = self.config["download_path"]
        
        console.print("[yellow]Cleaning up incomplete downloads...[/yellow]")
        cleaned_count = cleanup_incomplete_downloads(download_path)
        
        if cleaned_count > 0:
            console.print(f"[green]✓ Cleaned up {cleaned_count} incomplete downloads[/green]")
        else:
            console.print("[green]✓ No incomplete downloads found[/green]")


@click.group()
@click.option('--config', default='models_config.json', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """Qwen3 Model Manager CLI"""
    ctx.obj = ModelManagerCLI(config)


@cli.command()
@click.option('--details', is_flag=True, help='Show detailed information')
@click.pass_obj
def list(manager, details):
    """List available models"""
    manager.list_models(show_details=details)


@cli.command()
@click.argument('model_id')
@click.option('--force', is_flag=True, help='Force re-download')
@click.pass_obj
def download(manager, model_id, force):
    """Download a model"""
    manager.download_model(model_id, force=force)


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
@click.argument('model_id')
@click.pass_obj
def check(manager, model_id):
    """Check if a model can fit in available memory"""
    manager.check_model(model_id)


@cli.command()
@click.pass_obj
def cleanup(manager):
    """Clean up incomplete downloads"""
    manager.cleanup()


@cli.command()
@click.pass_obj
def status(manager):
    """Show system status"""
    # GPU status
    gpu_summary = manager.gpu_monitor.get_memory_usage_summary()
    
    status_text = Text()
    status_text.append("GPU Status:\n", style="cyan")
    status_text.append(f"  Total Memory: {gpu_summary['total_memory_mb']/1024:.1f}GB\n", style="white")
    status_text.append(f"  Used Memory: {gpu_summary['used_memory_mb']/1024:.1f}GB\n", style="white")
    status_text.append(f"  Available Memory: {gpu_summary['available_memory_mb']/1024:.1f}GB\n", style="white")
    status_text.append(f"  Utilization: {gpu_summary['utilization_percent']:.1f}%\n", style="white")
    status_text.append(f"  GPU Count: {gpu_summary['gpu_count']}\n", style="white")
    
    # Active model
    active_model = manager.config.get("active_model")
    status_text.append(f"\nActive Model: {active_model or 'None'}\n", style="cyan")
    
    # Downloaded models
    download_path = manager.config["download_path"]
    downloaded_models = get_downloaded_models(download_path)
    status_text.append(f"\nDownloaded Models: {len(downloaded_models)}\n", style="cyan")
    
    for model in downloaded_models:
        status_text.append(f"  {model['name']}: {model['size_formatted']}\n", style="white")
    
    panel = Panel(status_text, title="System Status", border_style="blue")
    console.print(panel)


if __name__ == "__main__":
    cli() 