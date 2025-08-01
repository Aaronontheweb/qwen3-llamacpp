"""
Model utilities for Qwen3 multi-GPU server
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from huggingface_hub import hf_hub_download, list_repo_files
import requests

logger = logging.getLogger("qwen3_server.model_utils")


def validate_model_config(model_config: Dict) -> bool:
    """
    Validate a model configuration
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["name", "type", "size", "quantization", "description"]
    
    for field in required_fields:
        if field not in model_config:
            logger.error(f"Missing required field '{field}' in model config")
            return False
    
    # Validate size format
    size = model_config["size"]
    if not isinstance(size, str) or not size.endswith("B"):
        logger.error(f"Invalid size format: {size}. Must end with 'B' (e.g., '14B')")
        return False
    
    # Validate quantization
    valid_quantizations = ["4bit", "8bit", "16bit", "32bit"]
    if model_config["quantization"] not in valid_quantizations:
        logger.error(f"Invalid quantization: {model_config['quantization']}. "
                    f"Must be one of: {valid_quantizations}")
        return False
    
    return True


def get_model_file_size(model_name: str, file_name: str = "model.gguf") -> Optional[int]:
    """
    Get the file size of a model file from HuggingFace
    
    Args:
        model_name: HuggingFace model name
        file_name: Name of the model file
        
    Returns:
        File size in bytes, or None if not found
    """
    try:
        # Try to get file info from HuggingFace
        url = f"https://huggingface.co/{model_name}/resolve/main/{file_name}"
        response = requests.head(url, allow_redirects=True)
        
        if response.status_code == 200:
            return int(response.headers.get("content-length", 0))
        else:
            logger.warning(f"Could not get file size for {model_name}/{file_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting file size for {model_name}/{file_name}: {e}")
        return None


def estimate_download_time(file_size_bytes: int, download_speed_mbps: float = 50.0) -> float:
    """
    Estimate download time for a model file
    
    Args:
        file_size_bytes: File size in bytes
        download_speed_mbps: Download speed in Mbps
        
    Returns:
        Estimated time in minutes
    """
    if download_speed_mbps <= 0:
        return 0.0
    
    # Convert to MB
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Calculate time in seconds
    time_seconds = (file_size_mb * 8) / download_speed_mbps
    
    # Convert to minutes
    return time_seconds / 60


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_model_path(model_name: str, download_path: str) -> str:
    """
    Get the local path for a model
    
    Args:
        model_name: HuggingFace model name
        download_path: Base download path
        
    Returns:
        Full path to the model directory
    """
    # Convert model name to directory name
    dir_name = model_name.replace("/", "_")
    return os.path.join(download_path, dir_name)


def is_model_downloaded(model_name: str, download_path: str) -> bool:
    """
    Check if a model is already downloaded
    
    Args:
        model_name: HuggingFace model name
        download_path: Base download path
        
    Returns:
        True if model is downloaded, False otherwise
    """
    model_path = get_model_path(model_name, download_path)
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        return False
    
    # Look for any .gguf file in the model directory (including subdirectories)
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.gguf'):
                gguf_file = os.path.join(root, file)
                # Check if file is not empty
                if os.path.exists(gguf_file) and os.path.getsize(gguf_file) > 0:
                    return True
    
    return False


def get_downloaded_models(download_path: str) -> List[Dict[str, Any]]:
    """
    Get list of downloaded models
    
    Args:
        download_path: Base download path
        
    Returns:
        List of model information dictionaries
    """
    models = []
    
    if not os.path.exists(download_path):
        return models
    
    for item in os.listdir(download_path):
        item_path = os.path.join(download_path, item)
        if os.path.isdir(item_path):
            gguf_file = os.path.join(item_path, "model.gguf")
            if os.path.exists(gguf_file):
                # Get file size
                file_size = os.path.getsize(gguf_file)
                
                models.append({
                    "name": item.replace("_", "/"),
                    "path": item_path,
                    "size_bytes": file_size,
                    "size_formatted": format_file_size(file_size),
                    "downloaded_at": os.path.getctime(gguf_file)
                })
    
    return models


def cleanup_incomplete_downloads(download_path: str) -> int:
    """
    Clean up incomplete downloads
    
    Args:
        download_path: Base download path
        
    Returns:
        Number of cleaned up directories
    """
    cleaned_count = 0
    
    if not os.path.exists(download_path):
        return cleaned_count
    
    for item in os.listdir(download_path):
        item_path = os.path.join(download_path, item)
        if os.path.isdir(item_path):
            gguf_file = os.path.join(item_path, "model.gguf")
            
            # Check if GGUF file is missing or incomplete
            if not os.path.exists(gguf_file) or os.path.getsize(gguf_file) == 0:
                try:
                    import shutil
                    shutil.rmtree(item_path)
                    logger.info(f"Cleaned up incomplete download: {item}")
                    cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning up {item}: {e}")
    
    return cleaned_count


def validate_gguf_file(file_path: str) -> bool:
    """
    Validate a GGUF file
    
    Args:
        file_path: Path to the GGUF file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False
    
    # Check file header (basic validation)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # GGUF files start with "GGUF"
            if header != b'GGUF':
                logger.warning(f"Invalid GGUF file header: {file_path}")
                return False
    except Exception as e:
        logger.error(f"Error reading GGUF file {file_path}: {e}")
        return False
    
    return True


def get_model_info_from_gguf(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract basic information from a GGUF file
    
    Args:
        file_path: Path to the GGUF file
        
    Returns:
        Model information dictionary or None
    """
    try:
        # This is a basic implementation - in practice, you might want to use
        # llama-cpp-python to get more detailed information
        if not validate_gguf_file(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        
        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "file_size_formatted": format_file_size(file_size),
            "valid": True
        }
        
    except Exception as e:
        logger.error(f"Error getting model info from {file_path}: {e}")
        return None 