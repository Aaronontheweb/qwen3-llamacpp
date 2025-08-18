"""
Process manager with proper cleanup handlers for abnormal exits
"""

import os
import signal
import atexit
import logging
import psutil
import weakref
from typing import Set, Optional
import gc

logger = logging.getLogger("qwen3_server.process_manager")


class ProcessManager:
    """Manages process lifecycle and ensures cleanup on all exit scenarios"""
    
    _instances: Set[weakref.ref] = set()
    _cleanup_registered = False
    _pid = None
    
    def __init__(self):
        self._cleanup_callbacks = []
        self._is_cleaning = False
        ProcessManager._instances.add(weakref.ref(self))
        ProcessManager._pid = os.getpid()
        
        if not ProcessManager._cleanup_registered:
            self._register_handlers()
            ProcessManager._cleanup_registered = True
    
    @classmethod
    def _register_handlers(cls):
        """Register all cleanup handlers once"""
        # Register atexit handler
        atexit.register(cls._cleanup_all)
        
        # Register signal handlers for graceful shutdown
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, cls._signal_handler)
        
        # Register handlers for abnormal termination
        if hasattr(signal, 'SIGSEGV'):
            signal.signal(signal.SIGSEGV, cls._crash_handler)
        if hasattr(signal, 'SIGABRT'):
            signal.signal(signal.SIGABRT, cls._crash_handler)
        if hasattr(signal, 'SIGBUS'):
            signal.signal(signal.SIGBUS, cls._crash_handler)
        if hasattr(signal, 'SIGFPE'):
            signal.signal(signal.SIGFPE, cls._crash_handler)
        
        logger.info(f"Process manager initialized for PID {cls._pid}")
    
    @classmethod
    def _signal_handler(cls, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, initiating cleanup...")
        cls._cleanup_all()
        os._exit(0)
    
    @classmethod
    def _crash_handler(cls, signum, frame):
        """Handle crash signals with emergency cleanup"""
        logger.error(f"CRASH: Received signal {signum}, performing emergency cleanup!")
        try:
            cls._emergency_cleanup()
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
        finally:
            # Re-raise the signal to allow core dump if configured
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
    
    @classmethod
    def _cleanup_all(cls):
        """Clean up all registered instances"""
        logger.info("Starting global cleanup...")
        
        # Clean up all alive instances
        for instance_ref in list(cls._instances):
            instance = instance_ref()
            if instance and not instance._is_cleaning:
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.error(f"Cleanup failed for instance: {e}")
        
        cls._instances.clear()
        logger.info("Global cleanup completed")
    
    @classmethod
    def _emergency_cleanup(cls):
        """Emergency cleanup for crashes - more aggressive"""
        logger.warning("Performing emergency cleanup...")
        
        # Kill any GPU processes spawned by this process
        try:
            current_process = psutil.Process(cls._pid)
            children = current_process.children(recursive=True)
            
            for child in children:
                try:
                    logger.info(f"Terminating child process {child.pid}")
                    child.terminate()
                except:
                    pass
            
            # Give processes time to terminate
            gone, alive = psutil.wait_procs(children, timeout=1)
            
            # Force kill any remaining
            for p in alive:
                try:
                    logger.warning(f"Force killing process {p.pid}")
                    p.kill()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Emergency process cleanup failed: {e}")
        
        # Force garbage collection to release GPU memory
        try:
            import gc
            gc.collect()
            
            # Try to clear GPU memory if llama_cpp is loaded
            try:
                from llama_cpp import llama_backend_free
                llama_backend_free()
                logger.info("Released llama.cpp backend memory")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Emergency memory cleanup failed: {e}")
    
    def register_cleanup(self, callback):
        """Register a cleanup callback"""
        self._cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Run all cleanup callbacks"""
        if self._is_cleaning:
            return
            
        self._is_cleaning = True
        logger.info("Running cleanup callbacks...")
        
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        self._cleanup_callbacks.clear()
        self._is_cleaning = False
    
    @classmethod
    def check_orphaned_models(cls):
        """Check for orphaned model processes from previous runs"""
        try:
            current_pid = os.getpid()
            
            # Look for llama.cpp processes not owned by current process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if it's a llama-related process
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if 'llama' in cmdline.lower() and proc.pid != current_pid:
                        # Check if parent is dead (orphaned)
                        try:
                            parent = proc.parent()
                            if parent is None or not parent.is_running():
                                logger.warning(f"Found orphaned llama process {proc.pid}, terminating...")
                                proc.terminate()
                                proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            logger.warning(f"Force killing orphaned process {proc.pid}")
                            proc.kill()
                        except:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to check for orphaned processes: {e}")


# Global process manager instance
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get or create the global process manager"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
        # Check for orphaned processes on startup
        ProcessManager.check_orphaned_models()
    return _process_manager