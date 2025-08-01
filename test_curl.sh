#!/bin/bash

# Test script for OpenCode streaming fix using curl

echo "Testing OpenCode streaming fix with curl..."
echo "=========================================="

# Test streaming with tool calls
echo "Making streaming request with tool calls..."
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-instruct",
    "stream": true,
    "messages": [{"role":"user","content":"call the calculator tool to add 2+2"}],
    "tools": [{
      "type":"function",
      "function":{
        "name":"calculate",
        "description":"Perform mathematical calculations",
        "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}
      }
    }]
  }'

echo ""
echo "=========================================="
echo "Test completed. Look for 'delta.tool_calls' in the output above."
echo "If you see tool_calls with proper function.name, the fix is working!" 