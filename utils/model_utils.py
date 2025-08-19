"""
Model utilities for downloading and managing models
"""

import glob
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def validate_model_config(model_config: dict) -> bool:
    """
    Validate model configuration

    Args:
        model_config: Model configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["name", "type", "size", "description"]

    for field in required_fields:
        if field not in model_config:
            logger.error(f"Missing required field: {field}")
            return False

    # Validate model type
    valid_types = ["instruction", "chat", "coder"]
    if model_config["type"] not in valid_types:
        logger.error(f"Invalid model type: {model_config['type']}. Must be one of {valid_types}")
        return False

    # Validate size format
    size = model_config["size"]
    if not isinstance(size, str) or not (size.endswith("B") or size.endswith("M")):
        logger.error(f"Invalid size format: {size}. Must end with 'B' or 'M'")
        return False

    return True


def estimate_download_time(file_size_bytes: int, download_speed_mbps: float = 50.0) -> float:
    """
    Estimate download time for a file

    Args:
        file_size_bytes: File size in bytes
        download_speed_mbps: Download speed in Mbps (default: 50 Mbps)

    Returns:
        Estimated download time in seconds
    """
    if file_size_bytes <= 0:
        return 0.0

    # Convert Mbps to bytes per second
    # 1 Mbps = 1,000,000 bits per second = 125,000 bytes per second
    download_speed_bps = download_speed_mbps * 125_000

    return file_size_bytes / download_speed_bps


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

    size_units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0

    size = float(size_bytes)
    while size >= 1024.0 and unit_index < len(size_units) - 1:
        size /= 1024.0
        unit_index += 1

    # Format with appropriate precision
    if unit_index == 0:
        return f"{int(size)} {size_units[unit_index]}"
    else:
        return f"{size:.1f} {size_units[unit_index]}"


def get_model_path(model_name: str, download_path: str) -> str:
    """
    Get local path for a model

    Args:
        model_name: HuggingFace model name
        download_path: Base download path

    Returns:
        Local model path
    """
    # Replace / with _ to create valid directory name
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

    # Look for key HuggingFace model files
    required_files = ['config.json', 'tokenizer.json']
    model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model-00001-of-*.bin']
    
    # Check for required config files
    for required_file in required_files:
        file_path = os.path.join(model_path, required_file)
        if not os.path.exists(file_path):
            return False
    
    # Check for at least one model file
    for pattern in model_files:
        if '*' in pattern:
            # Handle glob patterns
            matches = glob.glob(os.path.join(model_path, pattern))
            if matches:
                return True
        else:
            file_path = os.path.join(model_path, pattern)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True

    return False


def get_downloaded_models(download_path: str) -> list[dict[str, Any]]:
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

    try:
        for item in os.listdir(download_path):
            model_dir = os.path.join(download_path, item)
            if os.path.isdir(model_dir):
                # Convert directory name back to model name
                model_name = item.replace("_", "/")
                
                if is_model_downloaded(model_name, download_path):
                    # Calculate total size
                    total_size = 0
                    for root, _dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                total_size += os.path.getsize(file_path)

                    models.append({
                        "name": model_name,
                        "path": model_dir,
                        "size_bytes": total_size,
                        "size_formatted": format_file_size(total_size)
                    })

    except Exception as e:
        logger.error(f"Error listing downloaded models: {e}")

    return models


def cleanup_incomplete_downloads(download_path: str) -> int:
    """
    Clean up incomplete downloads

    Args:
        download_path: Base download path

    Returns:
        Number of items cleaned up
    """
    cleaned_count = 0

    if not os.path.exists(download_path):
        return cleaned_count

    try:
        for item in os.listdir(download_path):
            item_path = os.path.join(download_path, item)
            if os.path.isdir(item_path):
                # Convert directory name back to model name
                model_name = item.replace("_", "/")
                
                if not is_model_downloaded(model_name, download_path):
                    logger.info(f"Cleaning up incomplete download: {item}")
                    import shutil
                    shutil.rmtree(item_path)
                    cleaned_count += 1

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count