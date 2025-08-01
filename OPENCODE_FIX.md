# OpenCode Streaming Fix

## Problem Description

OpenCode was throwing `TypeError: Y is not an Object (evaluating '"name" in Y')` when using this server as an OpenAI-compatible provider with streaming enabled and tools configured.

## Root Cause

1. **Wrong media type**: The server was returning `text/plain` instead of `text/event-stream` for Server-Sent Events (SSE)
2. **Missing tool call streaming**: The streaming generator only emitted `delta.content` text chunks, never `delta.tool_calls` objects
3. **Client expectation**: OpenCode's SDK expects `choices[0].delta.tool_calls[*].function.name` to be a proper object

## Solution Implemented

### 1. Fixed Media Type
Changed the streaming response from `text/plain` to `text/event-stream`:

```python
return StreamingResponse(
    self._generate_stream(prompt, generation_params),
    media_type="text/event-stream"  # ✅ SSE for OpenAI-style streaming
)
```

### 2. Implemented Tool Call Streaming
Modified `_generate_stream()` to:
- Buffer incoming chunks to detect tool calls that span multiple tokens
- Use `tool_parser.extract_tool_calls(buffer)` to detect complete tool calls
- Emit proper OpenAI 1.2 streaming frames with `delta.tool_calls` when tool calls are detected
- Use `tool_parser.clean_text(buffer)` to get visible text before tool calls

### 3. Proper SSE Format
All streaming frames now follow the correct format:
```
data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", ...}\n\n
```

## Testing

### Python Test
```bash
python test_streaming_fix.py
```

### Curl Test
```bash
chmod +x test_curl.sh
./test_curl.sh
```

### Manual Test
```bash
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
```

## Expected Output

You should see streaming frames like:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1754065089,
  "model": "qwen3-30b-instruct",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": { "name": "calculate", "arguments": "{\"expression\":\"2+2\"}" }
        }
      ]
    },
    "finish_reason": null
  }]
}
```

## Verification

- ✅ SSE frames use `text/event-stream` and include `data:` lines
- ✅ Plain text is emitted as `choices[0].delta.content`
- ✅ Tool calls are emitted as objects under `choices[0].delta.tool_calls[*].function` with valid `name` and `arguments` strings
- ✅ Final frame: `choices[0].delta` is empty and `finish_reason` is `"stop"`
- ✅ OpenCode no longer throws `TypeError: Y is not an Object` and tool execution is triggered

## Files Modified

- `openai_server.py`: Fixed streaming response media type and implemented tool call streaming
- `test_streaming_fix.py`: Created comprehensive test script
- `test_curl.sh`: Created curl-based test script

## Compatibility

This fix maintains full compatibility with:
- Non-streaming requests (unchanged)
- Streaming requests without tools (unchanged)
- All existing API endpoints and functionality 