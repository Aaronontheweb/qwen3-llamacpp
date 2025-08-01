#!/usr/bin/env python3
"""
Local test script for Qwen3 Multi-GPU Server components
"""

import json
import sys
import os

def test_config_loading():
    """Test configuration loading"""
    print("🔍 Testing configuration loading...")
    try:
        with open("models_config.json", "r") as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
        print(f"  Active model: {config.get('active_model')}")
        print(f"  Available models: {len(config.get('models', {}))}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    modules = [
        ("utils.logging_config", "Logging configuration"),
        ("utils.gpu_monitor", "GPU monitoring"),
        ("utils.model_utils", "Model utilities"),
        ("tool_parser", "Tool parser"),
        ("llama_backend", "llama.cpp backend"),
    ]
    
    all_passed = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"✗ {description}: {e}")
            all_passed = False
    
    return all_passed

def test_gpu_monitoring():
    """Test GPU monitoring functionality"""
    print("\n🔍 Testing GPU monitoring...")
    try:
        from utils.gpu_monitor import get_gpu_monitor
        gpu_monitor = get_gpu_monitor()
        
        # Test basic functionality
        gpu_info = gpu_monitor.get_gpu_info()
        print(f"✓ GPU monitoring initialized")
        print(f"  GPUs detected: {len(gpu_info)}")
        
        if gpu_info:
            for gpu in gpu_info:
                print(f"  - GPU {gpu['index']}: {gpu['name']} ({gpu['total_memory_mb']/1024:.1f}GB)")
        
        return True
    except Exception as e:
        print(f"✗ GPU monitoring failed: {e}")
        return False

def test_tool_parser():
    """Test tool parsing functionality"""
    print("\n🔍 Testing tool parser...")
    try:
        from tool_parser import get_tool_parser
        parser = get_tool_parser()
        
        # Test XML parsing
        test_xml = """
        <tool_call>
        <function=calculate>
        <parameter=expression>
        15 * 7
        </parameter>
        </function>
        </tool_call>
        """
        
        tool_calls = parser.extract_tool_calls(test_xml)
        print(f"✓ Tool parser initialized")
        print(f"  Tool calls extracted: {len(tool_calls)}")
        
        if tool_calls:
            for tool_call in tool_calls:
                print(f"  - Function: {tool_call['function']['name']}")
                print(f"    Arguments: {tool_call['function']['arguments']}")
        
        return True
    except Exception as e:
        print(f"✗ Tool parser failed: {e}")
        return False

def test_model_utils():
    """Test model utilities"""
    print("\n🔍 Testing model utilities...")
    try:
        from utils.model_utils import validate_model_config, format_file_size
        
        # Test model config validation
        test_config = {
            "name": "test-model",
            "type": "instruction",
            "size": "7B",
            "quantization": "4bit",
            "description": "Test model"
        }
        
        is_valid = validate_model_config(test_config)
        print(f"✓ Model utilities initialized")
        print(f"  Config validation: {'✓' if is_valid else '✗'}")
        
        # Test file size formatting
        formatted_size = format_file_size(1024 * 1024 * 100)  # 100MB
        print(f"  File size formatting: {formatted_size}")
        
        return True
    except Exception as e:
        print(f"✗ Model utilities failed: {e}")
        return False

def test_llama_backend():
    """Test llama.cpp backend initialization"""
    print("\n🔍 Testing llama.cpp backend...")
    try:
        from llama_backend import get_model_manager
        
        # Load config
        with open("models_config.json", "r") as f:
            config = json.load(f)
        
        # Initialize model manager
        model_manager = get_model_manager(config)
        print(f"✓ llama.cpp backend initialized")
        print(f"  Model manager created")
        
        # Test status
        status = model_manager.get_status()
        print(f"  Current model: {status.get('current_model', 'None')}")
        
        return True
    except Exception as e:
        print(f"✗ llama.cpp backend failed: {e}")
        return False

def test_api_server():
    """Test API server initialization"""
    print("\n🔍 Testing API server...")
    try:
        from openai_server import Qwen3APIServer
        
        # Initialize server
        server = Qwen3APIServer()
        print(f"✓ API server initialized")
        print(f"  FastAPI app created")
        
        # Test routes
        routes = [route.path for route in server.app.routes]
        print(f"  Available routes: {len(routes)}")
        for route in routes[:5]:  # Show first 5 routes
            print(f"    - {route}")
        
        return True
    except Exception as e:
        print(f"✗ API server failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Qwen3 Multi-GPU Server - Local Component Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Module Imports", test_imports),
        ("GPU Monitoring", test_gpu_monitoring),
        ("Tool Parser", test_tool_parser),
        ("Model Utilities", test_model_utils),
        ("llama.cpp Backend", test_llama_backend),
        ("API Server", test_api_server),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test failed: {test_name}")
        except Exception as e:
            print(f"❌ Test error: {test_name} - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All component tests passed! Server components are working correctly.")
        print("\n💡 Next steps:")
        print("   1. Download a model: python model_manager.py download <model-id>")
        print("   2. Start the server: python openai_server.py")
        print("   3. Test the API: python test_server.py")
    else:
        print("⚠️  Some component tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 