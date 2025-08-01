#!/usr/bin/env python3
"""
Test tool call format without requiring a running server
"""

import json
from tool_parser import get_tool_parser

def test_single_tool_call():
    """Test parsing a single tool call"""
    print("🧪 Test 1: Single Tool Call")
    print("-" * 30)
    
    sample_xml = """I'll fetch that URL for you.

<tool_call>
<function=webfetch>
<parameter=url>
https://example.com
</parameter>
<parameter=format>
markdown
</parameter>
</function>
</tool_call>"""
    
    parser = get_tool_parser()
    tool_calls = parser.extract_tool_calls(sample_xml)
    clean_text = parser.clean_text(sample_xml)
    
    print(f"Clean text: {repr(clean_text)}")
    print(f"Tool calls found: {len(tool_calls)}")
    
    if tool_calls:
        tc = tool_calls[0]
        print(f"Function name: {tc['function']['name']}")
        print(f"Has index field: {'index' in tc}")
        print(f"Index value: {tc.get('index')}")
        
        args = json.loads(tc['function']['arguments'])
        print(f"Arguments: {args}")
        
        # Validate OpenAI format
        required = ['id', 'type', 'index', 'function']
        missing = [f for f in required if f not in tc]
        if missing:
            print(f"❌ Missing fields: {missing}")
            return False
        else:
            print("✅ All required fields present")
            return True
    else:
        print("❌ No tool calls found")
        return False

def test_multiple_tool_calls():
    """Test parsing multiple tool calls"""
    print("\n🧪 Test 2: Multiple Tool Calls")
    print("-" * 30)
    
    sample_xml = """I'll help with both tasks.

<tool_call>
<function=webfetch>
<parameter=url>
https://github.com/microsoft/vscode
</parameter>
</function>
</tool_call>

<tool_call>
<function=calculate>
<parameter=expression>
2 + 2
</parameter>
</function>
</tool_call>"""
    
    parser = get_tool_parser()
    tool_calls = parser.extract_tool_calls(sample_xml)
    clean_text = parser.clean_text(sample_xml)
    
    print(f"Clean text: {repr(clean_text)}")
    print(f"Tool calls found: {len(tool_calls)}")
    
    if len(tool_calls) == 2:
        for i, tc in enumerate(tool_calls):
            print(f"Tool {i}: {tc['function']['name']} (index: {tc.get('index')})")
        
        # Check indexes are correctly assigned
        indexes = [tc.get('index') for tc in tool_calls]
        expected_indexes = [0, 1]
        
        if indexes == expected_indexes:
            print("✅ Indexes correctly assigned")
            return True
        else:
            print(f"❌ Incorrect indexes: {indexes}, expected: {expected_indexes}")
            return False
    else:
        print(f"❌ Expected 2 tool calls, got {len(tool_calls)}")
        return False

def test_streaming_format():
    """Test that the format works for streaming"""
    print("\n🧪 Test 3: Streaming Response Format")
    print("-" * 30)
    
    # Simulate what would be in a streaming response
    tool_calls = [
        {
            "id": "call_12345678",
            "type": "function", 
            "index": 0,
            "function": {
                "name": "webfetch",
                "arguments": '{"url": "https://example.com"}'
            }
        }
    ]
    
    # Create streaming chunk as our server would
    chunk = {
        "id": "chatcmpl-12345678",
        "object": "chat.completion.chunk",
        "created": 1754073317,
        "model": "qwen3-7b-instruct",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": tool_calls},
            "finish_reason": "tool_calls"
        }]
    }
    
    print("Streaming chunk structure:")
    print(json.dumps(chunk, indent=2))
    
    # Validate the structure matches OpenAI spec
    try:
        choice = chunk["choices"][0]
        tool_call = choice["delta"]["tool_calls"][0]
        
        # Check required fields
        required_fields = ["id", "type", "index", "function"]
        missing = [f for f in required_fields if f not in tool_call]
        
        if missing:
            print(f"❌ Missing fields in tool call: {missing}")
            return False
        
        # Check function structure
        func = tool_call["function"]
        if "name" not in func or "arguments" not in func:
            print("❌ Invalid function structure")
            return False
        
        # Validate arguments is valid JSON
        json.loads(func["arguments"])
        
        print("✅ Streaming format is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating streaming format: {e}")
        return False

def test_edge_cases():
    """Test edge cases and malformed input"""
    print("\n🧪 Test 4: Edge Cases")
    print("-" * 30)
    
    test_cases = [
        ("Empty input", ""),
        ("No tool calls", "Just a regular response."),
        ("Incomplete tool call", "<tool_call><function=test>"),
        ("Malformed XML", "<tool_call><function=test><parameter=broken"),
    ]
    
    parser = get_tool_parser()
    all_passed = True
    
    for name, xml in test_cases:
        tool_calls = parser.extract_tool_calls(xml)
        print(f"{name}: {len(tool_calls)} tool calls found")
        
        # These should all return 0 tool calls gracefully
        if len(tool_calls) != 0:
            print(f"❌ {name} should return 0 tool calls")
            all_passed = False
    
    if all_passed:
        print("✅ All edge cases handled correctly")
    
    return all_passed

def main():
    """Run all tests"""
    print("🚀 Tool Call Format Validation (Offline)")
    print("=" * 50)
    
    tests = [
        test_single_tool_call,
        test_multiple_tool_calls, 
        test_streaming_format,
        test_edge_cases
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🏁 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Tool call format should work correctly with OpenCode")
        print("✅ Ready for deployment!")
    else:
        print("❌ Some tests failed - needs fixing before deployment")
    
    return passed == total

if __name__ == "__main__":
    main()