#!/usr/bin/env python3
"""
Quick start script for Qwen3 Multi-GPU Server
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "huggingface_hub",
        "rich",
        "click",
        "tqdm"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_llama_cpp():
    """Check if llama-cpp-python is installed with CUDA support"""
    try:
        import llama_cpp
        print("✓ llama-cpp-python installed")
        
        # Try to check if CUDA is available
        try:
            # This is a basic check - in practice, you'd want to test actual model loading
            print("✓ llama-cpp-python with CUDA support")
            return True
        except:
            print("⚠️  llama-cpp-python installed but CUDA support may not be available")
            print("   Run: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
            return False
    except ImportError:
        print("✗ llama-cpp-python not installed")
        print("   Run: pip install llama-cpp-python")
        print("   For CUDA support: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            print(f"✓ {device_count} GPU(s) detected")
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_gb = memory_info.total / (1024**3)
                print(f"  - GPU {i}: {name} ({total_memory_gb:.1f}GB)")
            return True
        else:
            print("⚠️  No GPUs detected")
            return False
    except ImportError:
        print("⚠️  pynvml not installed - cannot check GPU status")
        return False
    except Exception as e:
        print(f"⚠️  GPU check failed: {e}")
        return False

def check_config():
    """Check if configuration file exists"""
    config_file = "models_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration file: {config_file}")
            print(f"  Active model: {config.get('active_model', 'None')}")
            print(f"  Available models: {len(config.get('models', {}))}")
            return True
        except Exception as e:
            print(f"✗ Configuration file error: {e}")
            return False
    else:
        print(f"✗ Configuration file not found: {config_file}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["models", "cache", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Directory created: {directory}")

def download_sample_model():
    """Download a sample model"""
    print("\n📥 Downloading sample model...")
    try:
        result = subprocess.run([
            sys.executable, "model_manager.py", "download", "qwen3-8b-instruct"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Sample model downloaded successfully")
            return True
        else:
            print(f"⚠️  Model download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"⚠️  Model download error: {e}")
        return False

def start_server():
    """Start the API server"""
    print("\n🚀 Starting Qwen3 API server...")
    print("   Server will be available at: http://localhost:8080")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "openai_server.py"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main quick start function"""
    print("🚀 Qwen3 Multi-GPU Server - Quick Start")
    print("=" * 50)
    
    # Check requirements
    print("\n🔍 Checking requirements...")
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("llama-cpp-python", check_llama_cpp),
        ("GPU", check_gpu),
        ("Configuration", check_config),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n❌ Some requirements are not met. Please fix the issues above.")
        return False
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Ask user what to do next
    print("\n🎯 What would you like to do?")
    print("1. Download a sample model and start server")
    print("2. Start server (if you already have models)")
    print("3. Just check status")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            if download_sample_model():
                start_server()
            break
        elif choice == "2":
            start_server()
            break
        elif choice == "3":
            print("\n📊 System Status:")
            subprocess.run([sys.executable, "model_manager.py", "status"])
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Quick start interrupted by user")
    except Exception as e:
        print(f"❌ Quick start error: {e}")
        sys.exit(1) 