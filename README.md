# vLLM Multi-GPU Server for Qwen3 Models

A high-performance OpenAI-compatible API server with true multi-GPU support using vLLM, designed to maximize context window and throughput on multi-GPU systems.

## Key Features

- **True Multi-GPU Support**: Distributes KV cache across all GPUs using tensor parallelism
- **128k+ Token Context**: Achieve 100k-128k token contexts with dual 12GB GPUs
- **OpenAI Compatible API**: Drop-in replacement for OpenAI API
- **Flexible Backend System**: Support for vLLM, llama.cpp, and future backends
- **Auto-Configuration**: Automatically detects and configures for available GPUs
- **Production Ready**: Docker support, health checks, and monitoring

## System Requirements

### Hardware
- **GPUs**: NVIDIA GPUs with CUDA 12.1+ support
  - Minimum: 1x GPU with 12GB VRAM
  - Recommended: 2x GPUs with 12GB+ VRAM each
- **RAM**: 32GB minimum, 128GB recommended
- **Storage**: 100GB+ for models

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 12.1 or higher
- **Docker**: 20.10+ with NVIDIA Container Toolkit (optional)
- **Python**: 3.8+

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/qwen3-vllm.git
cd qwen3-vllm
git checkout feature/vllm-multi-gpu
```

### 2. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Verify GPU detection
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 3. Download Models

```bash
# Generate download script
python convert_models.py

# Run download script
chmod +x download_models.sh
./download_models.sh
```

### 4. Start Server

#### Option A: Direct Python
```bash
python openai_server.py --backend vllm --tensor-parallel-size 2
```

#### Option B: Docker
```bash
docker-compose up -d
```

### 5. Test API

```bash
# Test endpoint
curl http://localhost:8080/v1/models

# Test generation
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Configuration

### models_config.json

```json
{
  "backend": {
    "type": "vllm",
    "vllm_config": {
      "tensor_parallel_size": "auto",  // Auto-detect GPU count
      "gpu_memory_utilization": 0.90,   // Use 90% of VRAM
      "max_num_seqs": 256,              // Max concurrent sequences
      "kv_cache_dtype": "auto",         // Auto or "fp8" for efficiency
      "enable_prefix_caching": true     // Cache common prefixes
    }
  },
  "models": {
    "qwen3-30b-instruct": {
      "name": "Qwen/Qwen2.5-32B-Instruct",
      "max_context_tokens": 131072,
      "quantization": "AWQ"  // AWQ, GPTQ, or null
    }
  }
}
```

### Environment Variables

```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1        # Specify GPUs to use
export VLLM_TENSOR_PARALLEL_SIZE=2     # Number of GPUs for tensor parallelism

# Memory Settings
export VLLM_GPU_MEMORY_UTILIZATION=0.90  # GPU memory utilization

# Model Settings
export HUGGING_FACE_HUB_TOKEN=your_token  # For private models
```

## Multi-GPU Performance

### Expected Performance with Dual RTX 3060 + RTX 3080 Ti (24GB total)

| Metric | llama.cpp (Before) | vLLM (After) | Improvement |
|--------|-------------------|--------------|-------------|
| Max Context | 32k tokens | 128k tokens | 4x |
| Throughput | ~50 tokens/s | ~150 tokens/s | 3x |
| GPU Utilization | 30% | 90% | 3x |
| KV Cache Distribution | Single GPU | Both GPUs | ✓ |

### Memory Distribution

With a 30B parameter model (4-bit quantized):
- **Model Weights**: ~10GB split across both GPUs
- **KV Cache**: ~14GB available, distributed across GPUs
- **Max Context**: 128k+ tokens

## Testing

### Run Basic Tests
```bash
python tests/test_vllm_backend.py
```

### Run Full Test Suite
```bash
pytest tests/test_vllm_backend.py -v
```

### Test Multi-GPU Distribution
```bash
CUDA_VISIBLE_DEVICES=0,1 pytest tests/test_vllm_backend.py::TestMultiGPUDistribution -v
```

### Benchmark Performance
```bash
python tests/benchmark_context.py --context-sizes 32k,64k,96k,128k
```

## Docker Deployment

### Build Image
```bash
docker build -t vllm-multi-gpu:latest .
```

### Run with Docker Compose
```bash
# Start server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop server
docker-compose down
```

### Run with Docker (manual)
```bash
docker run --gpus all \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  vllm-multi-gpu:latest
```

## Monitoring

### GPU Usage
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Or use nvtop for better visualization
nvtop
```

### API Health Check
```bash
curl http://localhost:8080/health
```

### Server Status
```bash
curl http://localhost:8080/status
```

## Troubleshooting

### Issue: vLLM not detecting multiple GPUs
```bash
# Check CUDA installation
nvcc --version

# Check PyTorch GPU detection
python -c "import torch; print(torch.cuda.device_count())"

# Set GPUs explicitly
export CUDA_VISIBLE_DEVICES=0,1
```

### Issue: Out of Memory (OOM)
```bash
# Reduce GPU memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# Reduce max context length in models_config.json
"max_context_tokens": 65536
```

### Issue: Slow performance
```bash
# Enable optimizations
pip install flash-attn xformers

# Use AWQ quantization for better performance
"quantization": "AWQ"
```

## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models

### Custom Endpoints

- `GET /status` - Server and backend status
- `GET /health` - Health check
- `POST /switch_model` - Switch active model
- `POST /switch_backend` - Switch backend type

## Advanced Configuration

### Tensor Parallelism Tuning
```python
# For 4 GPUs
"tensor_parallel_size": 4

# For pipeline parallelism (advanced)
"pipeline_parallel_size": 2
"tensor_parallel_size": 2
```

### KV Cache Optimization
```python
# Use FP8 for 2x larger context
"kv_cache_dtype": "fp8"

# Enable prefix caching for repeated prompts
"enable_prefix_caching": true
```

### Quantization Options
- **AWQ**: Best performance, 4-bit quantization
- **GPTQ**: Good compatibility, 4-bit quantization
- **SqueezeLLM**: Experimental, better accuracy
- **None**: Full precision (FP16/BF16)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- vLLM team for the excellent inference engine
- Qwen team for the powerful models
- NVIDIA for CUDA and GPU support

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/qwen3-vllm/issues)
- Documentation: See TESTING_PLAN.md for detailed testing
- Conversion: See CONVERSION_NOTES.md for model conversion