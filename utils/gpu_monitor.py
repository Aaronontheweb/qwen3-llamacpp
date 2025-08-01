"""
GPU monitoring utilities for Qwen3 multi-GPU server
"""

import logging
from typing import Dict, List, Optional, Tuple
import psutil
import os

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger("qwen3_server.gpu_monitor")


class GPUMonitor:
    """Monitor GPU memory and utilization"""
    
    def __init__(self):
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"NVML initialized. Found {self.device_count} GPU(s)")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_available = False
                self.device_count = 0
        else:
            logger.warning("NVML not available. GPU monitoring will be limited.")
            self.device_count = 0
    
    def get_gpu_info(self) -> List[Dict]:
        """Get information about all available GPUs"""
        gpu_info = []
        
        if not self.nvml_available:
            return gpu_info
        
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_info.append({
                    "index": i,
                    "name": name,
                    "total_memory_mb": memory_info.total // (1024 * 1024),
                    "free_memory_mb": memory_info.free // (1024 * 1024),
                    "used_memory_mb": memory_info.used // (1024 * 1024),
                    "memory_utilization_percent": (memory_info.used / memory_info.total) * 100,
                    "gpu_utilization_percent": utilization.gpu,
                    "memory_utilization_percent_nvml": utilization.memory,
                    "temperature_celsius": temperature
                })
                
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return gpu_info
    
    def get_total_gpu_memory(self) -> Tuple[int, int]:
        """Get total and available GPU memory across all GPUs"""
        total_memory = 0
        available_memory = 0
        
        gpu_info = self.get_gpu_info()
        for gpu in gpu_info:
            total_memory += gpu["total_memory_mb"]
            available_memory += gpu["free_memory_mb"]
        
        return total_memory, available_memory
    
    def check_model_fits(self, required_memory_gb: float) -> Tuple[bool, Dict]:
        """
        Check if a model with given memory requirements can fit on available GPUs
        
        Args:
            required_memory_gb: Required memory in GB
            
        Returns:
            Tuple of (fits, details)
        """
        required_memory_mb = int(required_memory_gb * 1024)
        total_memory, available_memory = self.get_total_gpu_memory()
        
        fits = available_memory >= required_memory_mb
        
        details = {
            "required_memory_mb": required_memory_mb,
            "available_memory_mb": available_memory,
            "total_memory_mb": total_memory,
            "fits": fits,
            "gpu_details": self.get_gpu_info()
        }
        
        return fits, details
    
    def get_memory_usage_summary(self) -> Dict:
        """Get a summary of current memory usage"""
        gpu_info = self.get_gpu_info()
        total_memory = sum(gpu["total_memory_mb"] for gpu in gpu_info)
        used_memory = sum(gpu["used_memory_mb"] for gpu in gpu_info)
        available_memory = sum(gpu["free_memory_mb"] for gpu in gpu_info)
        
        return {
            "total_memory_mb": total_memory,
            "used_memory_mb": used_memory,
            "available_memory_mb": available_memory,
            "utilization_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
            "gpu_count": len(gpu_info),
            "gpu_details": gpu_info
        }
    
    def log_memory_status(self):
        """Log current memory status"""
        summary = self.get_memory_usage_summary()
        logger.info(f"GPU Memory Status: {summary['used_memory_mb']}MB used / "
                   f"{summary['total_memory_mb']}MB total "
                   f"({summary['utilization_percent']:.1f}% utilization)")
        
        for gpu in summary["gpu_details"]:
            logger.debug(f"GPU {gpu['index']} ({gpu['name']}): "
                        f"{gpu['used_memory_mb']}MB used / {gpu['total_memory_mb']}MB total")
    
    def cleanup(self):
        """Clean up NVML resources"""
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
                logger.debug("NVML shutdown completed")
            except Exception as e:
                logger.warning(f"Error during NVML shutdown: {e}")


# Global GPU monitor instance
gpu_monitor = GPUMonitor()


def get_gpu_monitor() -> GPUMonitor:
    """Get the global GPU monitor instance"""
    return gpu_monitor 