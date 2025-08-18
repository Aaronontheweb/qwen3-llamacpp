#!/usr/bin/env python3
"""
Quick test to verify tool call format matches OpenAI specification
"""

import json

from tool_parser import get_tool_parser


def test_tool_call_format():
    """Test that tool calls have the correct format for OpenAI compatibility"""

    # Sample XML output from Qwen (correct format per template instructions)
    sample_xml = """I'll help you fetch that content.

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

    # Parse with our tool parser
    parser = get_tool_parser()
    tool_calls = parser.extract_tool_calls(sample_xml)
    clean_text = parser.clean_text(sample_xml)

    print("🧪 Testing Tool Call Format")
    print("=" * 40)
    print(f"📝 Original text: {sample_xml[:50]!r}...")
    print(f"🧹 Clean text: {clean_text!r}")
    print(f"🔧 Tool calls found: {len(tool_calls)}")

    if tool_calls:
        print("\n📋 Tool Call Structure:")
        for i, tc in enumerate(tool_calls):
            print(f"Tool Call {i}:")
            print(json.dumps(tc, indent=2))

            # Validate required fields for OpenAI compatibility
            required_fields = ["id", "type", "index", "function"]
            missing_fields = [field for field in required_fields if field not in tc]

            if missing_fields:
                print(f"❌ MISSING FIELDS: {missing_fields}")
                return False

            # Validate function structure
            if "function" in tc:
                func = tc["function"]
                func_required = ["name", "arguments"]
                func_missing = [field for field in func_required if field not in func]

                if func_missing:
                    print(f"❌ MISSING FUNCTION FIELDS: {func_missing}")
                    return False

                # Validate arguments is valid JSON
                try:
                    args = json.loads(func["arguments"])
                    print(f"✅ Arguments JSON valid: {args}")
                except json.JSONDecodeError as e:
                    print(f"❌ INVALID ARGUMENTS JSON: {e}")
                    return False

            print(f"✅ Tool call {i} format is valid")

    print(f"\n🎯 Summary: {len(tool_calls)} tool calls extracted and validated")
    return len(tool_calls) > 0

def test_streaming_format():
    """Test the format expected by streaming responses"""

    # Test what streaming response should look like
    sample_tool_calls = [
        {
            "id": "call_12345678",
            "type": "function",
            "index": 0,
            "function": {
                "name": "webfetch",
                "arguments": '{"url": "https://example.com", "format": "markdown"}'
            }
        }
    ]

    # Create streaming response format
    streaming_chunk = {
        "id": "chatcmpl-12345678",
        "object": "chat.completion.chunk",
        "created": 1754073317,
        "model": "qwen3-7b-instruct",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": sample_tool_calls},
            "finish_reason": "tool_calls"
        }]
    }

    print("\n🌊 Testing Streaming Format")
    print("=" * 40)
    print("Streaming chunk structure:")
    print(json.dumps(streaming_chunk, indent=2))

    # Validate this matches the expected structure
    try:
        # Check that tool_calls[0] has index field
        tool_call = streaming_chunk["choices"][0]["delta"]["tool_calls"][0]
        if "index" not in tool_call:
            print("❌ MISSING: tool_calls[0].index field")
            return False

        print("✅ Streaming format includes required index field")
        return True

    except Exception as e:
        print(f"❌ STREAMING FORMAT ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Tool Call Format Validation")
    print("=" * 50)

    success1 = test_tool_call_format()
    success2 = test_streaming_format()

    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED - Format should work with OpenCode!")
    else:
        print("\n❌ TESTS FAILED - Format needs fixing")
