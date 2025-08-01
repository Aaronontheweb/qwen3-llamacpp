#!/usr/bin/env python3
"""
Debug the 'mem' function call issue
"""

from tool_parser import get_tool_parser

def test_mem_function():
    """Test parsing a mem function call"""
    
    sample_xml = """I'll help you with that memory task.

<tool_call>
<function=mem>
<parameter=query>
test search
</parameter>
</function>
</tool_call>"""
    
    print("🔍 Testing 'mem' function call parsing")
    print("=" * 40)
    print("Input XML:")
    print(sample_xml)
    print("\n" + "=" * 40)
    
    parser = get_tool_parser()
    tool_calls = parser.extract_tool_calls(sample_xml)
    clean_text = parser.clean_text(sample_xml)
    
    print(f"Tool calls found: {len(tool_calls)}")
    print(f"Clean text: {repr(clean_text)}")
    
    if tool_calls:
        tc = tool_calls[0]
        print(f"Function name: '{tc['function']['name']}'")
        print(f"Arguments: {tc['function']['arguments']}")
        print("✅ Parsing successful")
    else:
        print("❌ No tool calls parsed")
    
    return len(tool_calls) > 0

def test_empty_function_content():
    """Test what causes 'Empty function content in XML' warning"""
    
    test_cases = [
        ("Normal", "<function=test>content</function>"),
        ("Empty content", "<function=>content</function>"),
        ("No equals", "<function>content</function>"), 
        ("Just function tag", "<function=test>"),
        ("Malformed", "<function=test><parameter="),
    ]
    
    print("\n🔍 Testing function content extraction")
    print("=" * 40)
    
    parser = get_tool_parser()
    
    for name, xml_fragment in test_cases:
        full_xml = f"<tool_call>{xml_fragment}</tool_call>"
        tool_calls = parser.extract_tool_calls(full_xml)
        print(f"{name}: {len(tool_calls)} tool calls found")

def test_actual_server_output():
    """Test with what the server might actually be producing"""
    
    # Based on the error, let's see if it's a formatting issue
    possible_outputs = [
        # Standard format
        """<tool_call>
<function=mem>
<parameter=query>
search term
</parameter>
</function>
</tool_call>""",
        
        # Missing closing tags
        """<tool_call>
<function=mem>
<parameter=query>
search term
</parameter>""",
        
        # Different whitespace
        """<tool_call><function=mem><parameter=query>search term</parameter></function></tool_call>""",
        
        # Extra content
        """I need to search for that.

<tool_call>
<function=mem>
<parameter=query>search term</parameter>
</function>
</tool_call>

Let me find that information."""
    ]
    
    print("\n🔍 Testing possible server outputs")
    print("=" * 40)
    
    parser = get_tool_parser()
    
    for i, xml in enumerate(possible_outputs):
        print(f"\nTest {i+1}:")
        tool_calls = parser.extract_tool_calls(xml)
        print(f"  Tool calls: {len(tool_calls)}")
        if tool_calls:
            print(f"  Function: {tool_calls[0]['function']['name']}")

if __name__ == "__main__":
    test_mem_function()
    test_empty_function_content() 
    test_actual_server_output()