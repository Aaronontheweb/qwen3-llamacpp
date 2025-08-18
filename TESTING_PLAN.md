# vLLM Multi-GPU Testing Plan for Linux/CUDA Environment

## Prerequisites
- Linux machine with CUDA 12.1+ installed
- NVIDIA drivers 525.60+ 
- Docker (optional but recommended)
- Python 3.8+
- Both RTX 3060 (12GB) and RTX 3080 Ti (12GB) GPUs available

## Environment Setup Tests

### 1. GPU Detection and CUDA Verification
```bash
# Verify both GPUs are detected
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# List GPU details
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### 2. vLLM Installation Test
```bash
# Install vLLM
pip install vllm

# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Test vLLM multi-GPU support
python -c "from vllm import LLM; print('vLLM imported successfully')"
```

## Functional Tests

### 3. Single GPU Operation Test
```python
# test_single_gpu.py
import torch
from backends.vllm_backend import VLLMBackend

config = {
    "tensor_parallel_size": 1,  # Single GPU
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768
}

backend = VLLMBackend(config)
model_path = "./models/Qwen3-30B-Instruct"
success = backend.load_model(model_path, {"max_context_tokens": 32768})
assert success, "Failed to load model on single GPU"

# Test generation
response = backend.generate("Hello, how are you?", max_tokens=100)
print(f"Single GPU response: {response}")
```

### 4. Multi-GPU Distribution Test
```python
# test_multi_gpu.py
import torch
import pynvml
from backends.vllm_backend import VLLMBackend

def get_gpu_memory_usage():
    pynvml.nvmlInit()
    usage = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage.append({
            "gpu": i,
            "used_gb": info.used / 1024**3,
            "total_gb": info.total / 1024**3,
            "percent": (info.used / info.total) * 100
        })
    return usage

# Test with 2 GPUs
config = {
    "tensor_parallel_size": 2,  # Use both GPUs
    "gpu_memory_utilization": 0.9,
    "max_model_len": 131072  # 128k context
}

print("Memory before loading:")
print(get_gpu_memory_usage())

backend = VLLMBackend(config)
model_path = "./models/Qwen3-30B-Instruct"
success = backend.load_model(model_path, {"max_context_tokens": 131072})
assert success, "Failed to load model on multiple GPUs"

print("\nMemory after loading:")
memory_usage = get_gpu_memory_usage()
print(memory_usage)

# Verify both GPUs are being used
assert all(gpu["percent"] > 30 for gpu in memory_usage), "GPUs not evenly utilized"
```

### 5. Context Window Scaling Test
```python
# test_context_scaling.py
from backends.vllm_backend import VLLMBackend
import time

def test_context_size(backend, context_tokens, test_name):
    # Create a long prompt
    prompt = "Hello " * (context_tokens // 2)  # Rough approximation
    
    start_time = time.time()
    try:
        response = backend.generate(
            prompt, 
            max_tokens=100,
            temperature=0.7
        )
        elapsed = time.time() - start_time
        print(f"✓ {test_name}: {context_tokens} tokens in {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"✗ {test_name}: Failed with {e}")
        return False

config = {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.9
}

backend = VLLMBackend(config)
backend.load_model("./models/Qwen3-30B-Instruct", {"max_context_tokens": 131072})

# Test increasing context sizes
test_sizes = [
    (4096, "4k context"),
    (8192, "8k context"),
    (16384, "16k context"),
    (32768, "32k context"),
    (65536, "64k context"),
    (98304, "96k context"),
    (131072, "128k context")
]

for size, name in test_sizes:
    test_context_size(backend, size, name)
```

### 6. KV Cache Distribution Test
```python
# test_kv_cache_distribution.py
import torch
from vllm import LLM, SamplingParams

def test_kv_cache_distribution():
    # Initialize with tensor parallelism
    llm = LLM(
        model="./models/Qwen3-30B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=131072
    )
    
    # Create a batch of requests with varying lengths
    prompts = [
        "Short prompt",
        "Medium " * 100,
        "Long " * 1000,
        "Very long " * 5000
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100
    )
    
    # Generate and monitor memory
    outputs = llm.generate(prompts, sampling_params)
    
    # Check KV cache is distributed
    # This would need internal vLLM metrics access
    print(f"Processed {len(outputs)} requests")
    for output in outputs:
        print(f"Request length: {len(output.prompt)}, Generated: {len(output.outputs[0].text)}")

test_kv_cache_distribution()
```

## Performance Benchmarks

### 7. Throughput Test
```python
# test_throughput.py
import time
import asyncio
from backends.vllm_backend import VLLMBackend

async def benchmark_throughput(backend, num_requests=100):
    prompts = [f"Question {i}: What is the meaning of life?" for i in range(num_requests)]
    
    start_time = time.time()
    tasks = []
    
    for prompt in prompts:
        task = asyncio.create_task(
            backend.generate_async(prompt, max_tokens=50)
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    total_tokens = sum(len(r.split()) for r in responses)
    throughput = total_tokens / elapsed
    
    print(f"Processed {num_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/second")
    
    return throughput

# Compare single vs multi-GPU
configs = [
    {"name": "Single GPU", "tensor_parallel_size": 1},
    {"name": "Dual GPU", "tensor_parallel_size": 2}
]

for config in configs:
    backend = VLLMBackend(config)
    backend.load_model("./models/Qwen3-30B-Instruct", {})
    throughput = asyncio.run(benchmark_throughput(backend))
    print(f"{config['name']}: {throughput:.2f} tokens/s\n")
```

### 8. Latency Test
```python
# test_latency.py
import time
from backends.vllm_backend import VLLMBackend

def test_latency(backend, prompt_length, max_tokens):
    prompt = "Tell me " * prompt_length
    
    # Test first token latency
    start_time = time.time()
    first_token = None
    
    for token in backend.generate_stream(prompt, max_tokens=max_tokens):
        if first_token is None:
            first_token = time.time() - start_time
        # Continue to get all tokens
    
    total_time = time.time() - start_time
    
    print(f"Prompt length: {prompt_length * 2} tokens")
    print(f"First token latency: {first_token * 1000:.2f}ms")
    print(f"Total generation time: {total_time:.2f}s")
    print(f"Tokens per second: {max_tokens / total_time:.2f}")
    
    return first_token, total_time

# Test various prompt lengths
backend = VLLMBackend({"tensor_parallel_size": 2})
backend.load_model("./models/Qwen3-30B-Instruct", {})

for prompt_len in [10, 100, 1000, 5000]:
    test_latency(backend, prompt_len, max_tokens=100)
    print("-" * 40)
```

## Stress Tests

### 9. Memory Stress Test
```python
# test_memory_stress.py
def stress_test_memory():
    config = {
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.95,  # Push to 95%
        "max_model_len": 131072
    }
    
    backend = VLLMBackend(config)
    backend.load_model("./models/Qwen3-30B-Instruct", {})
    
    # Generate with maximum context
    max_prompt = "Test " * 30000  # ~120k tokens
    
    try:
        response = backend.generate(max_prompt, max_tokens=1000)
        print("✓ Successfully handled maximum context")
    except torch.cuda.OutOfMemoryError:
        print("✗ OOM at maximum context")
    except Exception as e:
        print(f"✗ Error: {e}")
```

### 10. Concurrent Request Test
```python
# test_concurrent.py
import concurrent.futures
from backends.vllm_backend import VLLMBackend

def process_request(backend, request_id):
    prompt = f"Request {request_id}: Explain quantum computing"
    response = backend.generate(prompt, max_tokens=100)
    return f"Request {request_id} completed: {len(response)} chars"

backend = VLLMBackend({"tensor_parallel_size": 2})
backend.load_model("./models/Qwen3-30B-Instruct", {})

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_request, backend, i) for i in range(50)]
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
```

## Integration Tests

### 11. API Server Test
```bash
# Start the server
python openai_server.py --backend vllm --tensor-parallel-size 2

# Test endpoints
curl http://localhost:8080/v1/models

# Test completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Test streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### 12. Model Switching Test
```python
# test_model_switching.py
from backends.factory import BackendFactory

# Test switching between models
models = [
    "Qwen3-30B-Instruct",
    "Qwen3-8B-Instruct",
    "Phi-4-mini"
]

for model in models:
    backend = BackendFactory.create_backend("vllm", {"tensor_parallel_size": 2})
    success = backend.load_model(f"./models/{model}", {})
    assert success, f"Failed to load {model}"
    
    response = backend.generate("Test prompt", max_tokens=50)
    print(f"{model}: {len(response)} chars generated")
    
    backend.unload_model()
```

## Validation Checklist

### Essential Tests (Must Pass)
- [ ] Both GPUs detected and accessible
- [ ] vLLM loads successfully
- [ ] Model loads on single GPU
- [ ] Model loads on dual GPUs
- [ ] Basic generation works
- [ ] Streaming generation works
- [ ] 32k context works (baseline)
- [ ] 64k context works
- [ ] 96k context works
- [ ] Memory distributed across GPUs

### Performance Tests (Should Meet Targets)
- [ ] 128k context achieved
- [ ] Throughput 2x better than llama.cpp
- [ ] First token latency < 500ms
- [ ] Both GPUs show >40% utilization
- [ ] No memory leaks after 100 requests

### Stress Tests (Nice to Have)
- [ ] Handles 50 concurrent requests
- [ ] Survives 1000 sequential requests
- [ ] Gracefully handles OOM scenarios
- [ ] Model switching works smoothly

## Automated Test Script
```bash
#!/bin/bash
# run_all_tests.sh

echo "Starting vLLM Multi-GPU Test Suite"
echo "=================================="

# Environment checks
python test_gpu_detection.py || exit 1

# Functional tests
python test_single_gpu.py || exit 1
python test_multi_gpu.py || exit 1
python test_context_scaling.py || exit 1

# Performance tests
python test_throughput.py
python test_latency.py

# Stress tests
python test_memory_stress.py
python test_concurrent.py

# Integration tests
python test_api_server.py
python test_model_switching.py

echo "=================================="
echo "Test Suite Complete"
```

## Expected Results

### Memory Distribution
- GPU 0: ~45% memory usage (5.4GB / 12GB)
- GPU 1: ~45% memory usage (5.4GB / 12GB)
- Remaining memory for KV cache: ~13GB total

### Context Calculation
- KV cache budget: 13GB
- Per-token KV cache: ~100KB
- Maximum context: 130,000 tokens

### Performance Targets
- Single GPU: 32k context, 50 tokens/s
- Dual GPU: 128k context, 100+ tokens/s
- First token latency: <500ms
- Concurrent requests: 50+ without degradation