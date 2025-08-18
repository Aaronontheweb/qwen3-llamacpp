#!/usr/bin/env python3
"""
Simple test script for Qwen3 Multi-GPU Server
"""

import json
import time
from typing import Any

import requests


def test_server_health(base_url: str = "http://localhost:8080") -> bool:
    """Test server health endpoint"""
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server health check passed: {data}")
            return True
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Server health check error: {e}")
        return False

def test_models_endpoint(base_url: str = "http://localhost:8080") -> bool:
    """Test models endpoint"""
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Models endpoint: {len(data['data'])} models available")
            for model in data['data']:
                print(f"  - {model['id']}")
            return True
        else:
            print(f"✗ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Models endpoint error: {e}")
        return False

def test_chat_completion(base_url: str = "http://localhost:8080", model: str = "qwen3-14b-instruct") -> bool:
    """Test chat completion endpoint"""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with 'Hello from Qwen3!'"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }

        response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✓ Chat completion successful")
            print(f"  Response: {data['choices'][0]['message']['content']}")
            return True
        else:
            print(f"✗ Chat completion failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Chat completion error: {e}")
        return False

def test_tool_calling(base_url: str = "http://localhost:8080", model: str = "qwen3-14b-instruct") -> bool:
    """Test tool calling functionality"""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Calculate 15 * 7"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to evaluate"
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                }
            ],
            "temperature": 0.3,
            "max_tokens": 200
        }

        response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✓ Tool calling test successful")

            # Check if tool calls were generated
            message = data['choices'][0]['message']
            if 'tool_calls' in message:
                print(f"  Tool calls generated: {len(message['tool_calls'])}")
                for tool_call in message['tool_calls']:
                    print(f"    Function: {tool_call['function']['name']}")
                    print(f"    Arguments: {tool_call['function']['arguments']}")
            else:
                print("  No tool calls generated (this is normal for some models)")

            return True
        else:
            print(f"✗ Tool calling test failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Tool calling test error: {e}")
        return False

def test_admin_endpoints(base_url: str = "http://localhost:8080") -> bool:
    """Test admin endpoints"""
    try:
        # Test model status
        response = requests.get(f"{base_url}/admin/model_status")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model status endpoint: {data.get('current_model', 'None')}")
        else:
            print(f"✗ Model status endpoint failed: {response.status_code}")
            return False

        # Test GPU usage
        response = requests.get(f"{base_url}/admin/gpu_usage")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ GPU usage endpoint: {data.get('gpu_count', 0)} GPUs")
            print(f"  Total memory: {data.get('total_memory_mb', 0)/1024:.1f}GB")
            print(f"  Available memory: {data.get('available_memory_mb', 0)/1024:.1f}GB")
        else:
            print(f"✗ GPU usage endpoint failed: {response.status_code}")
            return False

        return True
    except Exception as e:
        print(f"✗ Admin endpoints error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Qwen3 Multi-GPU Server Test Suite")
    print("=" * 50)

    base_url = "http://localhost:8080"

    tests = [
        ("Server Health", lambda: test_server_health(base_url)),
        ("Models Endpoint", lambda: test_models_endpoint(base_url)),
        ("Chat Completion", lambda: test_chat_completion(base_url)),
        ("Tool Calling", lambda: test_tool_calling(base_url)),
        ("Admin Endpoints", lambda: test_admin_endpoints(base_url)),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test failed: {test_name}")
        except Exception as e:
            print(f"❌ Test error: {test_name} - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Server is working correctly.")
    else:
        print("⚠️  Some tests failed. Check server logs for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
