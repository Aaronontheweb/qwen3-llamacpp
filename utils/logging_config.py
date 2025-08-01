"""
Logging configuration for Qwen3 multi-GPU server
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> logging.Logger:
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("qwen3_server")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "qwen3_server.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "qwen3_server_errors.log"),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    # Model management specific logger
    model_logger = logging.getLogger("qwen3_server.model_manager")
    model_logger.setLevel(logging.DEBUG)
    
    # API specific logger
    api_logger = logging.getLogger("qwen3_server.api")
    api_logger.setLevel(logging.DEBUG)
    
    # GPU monitoring logger
    gpu_logger = logging.getLogger("qwen3_server.gpu_monitor")
    gpu_logger.setLevel(logging.DEBUG)
    
    return logger


def get_logger(name: str = "qwen3_server") -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 