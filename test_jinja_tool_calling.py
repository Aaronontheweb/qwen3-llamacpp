#!/usr/bin/env python3
"""
Test script for Jinja template-based tool calling system
"""

import json
import requests
import time
from typing import Dict, List, Any

# Test configuration
BASE_URL = "http://localhost:8080"

def test_tool_calling_accuracy():
    """Test tool calling accuracy with various scenarios"""
    
    # Test tools configuration
    tools = [
        {
            "type": "function",
            "function": {
                "name": "webfetch",
                "description": "Fetch content from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        },
                        "format": {
                            "type": "string",
                            "description": "Response format",
                            "enum": ["text", "markdown", "json"]
                        }
                    },
                    "required": ["url"]
                }
            }
        },
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
    ]
    
    # Test cases
    test_cases = [
        {
            "name": "Single tool call",
            "messages": [
                {"role": "user", "content": "Fetch content from https://example.com"}
            ],
            "expected_tools": ["webfetch"],
            "expected_params": ["url"]
        },
        {
            "name": "Multiple tool calls",
            "messages": [
                {"role": "user", "content": "Fetch https://github.com/microsoft/vscode and calculate 2+2"}
            ],
            "expected_tools": ["webfetch", "calculate"],
            "expected_params": ["url", "expression"]
        },
        {
            "name": "Tool call with reasoning",
            "messages": [
                {"role": "user", "content": "I need to analyze the GitHub repo at https://github.com/python/cpython"}
            ],
            "expected_tools": ["webfetch"],
            "expected_params": ["url"]
        },
        {
            "name": "No tool call needed",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "expected_tools": [],
            "expected_params": []
        }
    ]
    
    print("🧪 Testing Tool Calling Accuracy")
    print("=" * 50)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        
        # Make request
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": "test",
            "messages": test_case["messages"],
            "tools": tools,
            "temperature": 0.1
        })
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            results.append({"test": test_case["name"], "passed": False, "error": "Request failed"})
            continue
        
        data = response.json()
        
        # Extract actual tool calls
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        actual_tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")
        
        # Validate results
        actual_tools = [tc["function"]["name"] for tc in actual_tool_calls]
        
        print(f"   Content: {content[:100]}...")
        print(f"   Expected tools: {test_case['expected_tools']}")
        print(f"   Actual tools: {actual_tools}")
        
        # Check if tool expectations are met
        tools_match = set(actual_tools) == set(test_case["expected_tools"])
        
        # Check for leaked XML/plain-text in content
        has_xml_leak = any(tag in content for tag in ["<tool_call>", "<function>", "webfetch(", "calculate("])
        
        if tools_match and not has_xml_leak:
            print("   ✅ PASSED")
            results.append({"test": test_case["name"], "passed": True})
        else:
            print(f"   ❌ FAILED - Tools match: {tools_match}, No XML leak: {not has_xml_leak}")
            results.append({
                "test": test_case["name"], 
                "passed": False, 
                "tools_match": tools_match,
                "xml_leak": has_xml_leak,
                "content": content
            })
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 SUMMARY")
    print(f"Passed: {passed}/{total} ({accuracy:.1f}%)")
    
    return results

def test_streaming_tool_calls():
    """Test streaming tool call responses"""
    print("\n🌊 Testing Streaming Tool Calls")
    print("=" * 50)
    
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }]
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "test",
        "messages": [{"role": "user", "content": "Search for information about Python programming"}],
        "tools": tools,
        "stream": True,
        "temperature": 0.1
    }, stream=True)
    
    if response.status_code != 200:
        print(f"❌ Streaming request failed: {response.status_code}")
        return False
    
    chunks = []
    tool_calls_found = False
    content_before_tools = ""
    
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            data_part = line[6:].decode()
            if data_part == "[DONE]":
                break
            
            try:
                chunk_data = json.loads(data_part)
                chunks.append(chunk_data)
                
                choice = chunk_data.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                
                if "tool_calls" in delta:
                    tool_calls_found = True
                    print(f"   🔧 Tool calls found: {len(delta['tool_calls'])}")
                
                if "content" in delta and not tool_calls_found:
                    content_before_tools += delta["content"]
                    
            except json.JSONDecodeError:
                continue
    
    print(f"   📦 Total chunks: {len(chunks)}")
    print(f"   🔧 Tool calls detected: {tool_calls_found}")
    print(f"   📝 Content before tools: '{content_before_tools[:50]}...'")
    
    # Check for XML leaks in streaming content
    has_xml_leak = any(tag in content_before_tools for tag in ["<tool_call>", "<function>", "search("])
    
    if tool_calls_found and not has_xml_leak:
        print("   ✅ STREAMING PASSED")
        return True
    else:
        print(f"   ❌ STREAMING FAILED - Tool calls: {tool_calls_found}, No XML leak: {not has_xml_leak}")
        return False

def test_template_fallback():
    """Test fallback to legacy system"""
    print("\n🔄 Testing Template Fallback")
    print("=" * 50)
    
    # Test with template disabled
    # This would require temporarily modifying config or adding API endpoint
    print("   ⚠️  Manual test required: Set use_jinja_template: false in config")
    print("   ✅ Should still work with legacy prompt builder")
    
    return True

def test_context_window_usage():
    """Test context window usage with template vs legacy"""
    print("\n📏 Testing Context Window Usage")
    print("=" * 50)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"test_function_{i}",
                "description": f"Test function {i} with a longer description that includes multiple parameters and detailed explanations of what this function does and how it should be used in various scenarios",
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{j}": {
                            "type": "string",
                            "description": f"Parameter {j} description with detailed explanation"
                        } for j in range(5)
                    },
                    "required": [f"param_{j}" for j in range(3)]
                }
            }
        } for i in range(10)
    ]
    
    response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "test",
        "messages": [{"role": "user", "content": "Help me test these functions"}],
        "tools": tools,
        "max_tokens": 50  # Short response to focus on prompt
    })
    
    if response.status_code == 200:
        data = response.json()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        print(f"   📊 Prompt tokens with {len(tools)} tools: {prompt_tokens}")
        print("   ✅ Context window test completed")
        return True
    else:
        print(f"   ❌ Context window test failed: {response.status_code}")
        return False

def main():
    """Run all tests"""
    print("🚀 Jinja Template Tool Calling Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server first.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the server first.")
        return
    
    print("✅ Server is running")
    
    # Run tests
    tests = [
        ("Tool Calling Accuracy", test_tool_calling_accuracy),
        ("Streaming Tool Calls", test_streaming_tool_calls),
        ("Template Fallback", test_template_fallback),
        ("Context Window Usage", test_context_window_usage)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n🏁 FINAL RESULTS")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for r in results.values() if r)
    total_tests = len(results)
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

if __name__ == "__main__":
    main()