# Qwen3 Multi-GPU Server

A configurable, standalone server for running Unsloth Qwen3 instruction-following models across multiple NVIDIA GPUs using llama.cpp backend with OpenAI-compatible API and tool calling support.

## 🚀 Features

- **Multi-GPU Support**: Automatic distribution across RTX 3060 (12GB) + RTX 3080 Ti (12GB) = 24GB total VRAM
- **Model Management**: Download and hot-swap between different instruction-following models
- **OpenAI Compatibility**: Full OpenAI API compatibility for seamless integration
- **Tool Calling**: Convert Qwen3 XML tool calls to OpenAI JSON format
- **Immediate Model Switching**: Hot-swap models without server restart
- **Standalone Application**: Independent of existing codebase
- **Comprehensive Logging**: Detailed logging with rotation and error tracking
- **GPU Monitoring**: Real-time GPU memory and utilization tracking

## 🏗️ Architecture

### Core Components

1. **Model Manager** (`model_manager.py`): CLI for model management and downloads
2. **llama.cpp Backend** (`llama_backend.py`): Multi-GPU model loading and inference
3. **OpenAI API Server** (`openai_server.py`): FastAPI-based OpenAI-compatible API
4. **Tool Parser** (`tool_parser.py`): XML to JSON tool calling conversion
5. **GPU Monitor** (`utils/gpu_monitor.py`): GPU memory and utilization tracking
6. **Configuration System** (`models_config.json`): Centralized model and server configuration

### File Structure

```
qwen3-server/
├── models_config.json          # Main configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── model_manager.py            # Model management CLI
├── llama_backend.py            # llama.cpp integration
├── openai_server.py            # OpenAI-compatible API server
├── tool_parser.py              # Tool calling parser
├── utils/
│   ├── __init__.py
│   ├── logging_config.py       # Logging configuration
│   ├── gpu_monitor.py          # GPU memory monitoring
│   └── model_utils.py          # Model utilities
├── models/                     # Downloaded models directory
├── cache/                      # HuggingFace cache
└── logs/                       # Application logs
```

## 📋 Requirements

### Hardware
- **GPUs**: RTX 3060 (12GB) + RTX 3080 Ti (12GB) = 24GB total VRAM
- **Target Models**: 4-bit quantized instruction-following models up to 30B parameters
- **Distribution**: llama.cpp handles multi-GPU model distribution automatically

### Software
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- NVIDIA drivers compatible with your CUDA version

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qwen3-llamacpp
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install llama-cpp-python with CUDA support**:
   ```bash
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
   ```

## ⚙️ Configuration

The server is configured through `models_config.json`. Key configuration options:

### Model Configuration
```json
{
  "models": {
    "qwen3-14b-instruct": {
      "name": "unsloth/Qwen3-14B-Instruct",
      "type": "instruction",
      "size": "14B",
      "quantization": "4bit",
      "description": "Medium instruction-following model, good balance",
      "memory_estimate_gb": 8,
      "recommended_gpus": 1,
      "chat_template": "qwen3",
      "trust_remote_code": true,
      "default_params": {
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9
      }
    }
  },
  "active_model": "qwen3-14b-instruct",
  "download_path": "./models",
  "cache_dir": "./cache",
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "log_level": "INFO"
  }
}
```

### Available Models

| Model ID | Size | Type | Memory (GB) | GPUs | Description |
|----------|------|------|-------------|------|-------------|
| `qwen3-30b-instruct` | 30B | instruction | 18 | 2 | Large instruction-following model for complex tasks |
| `qwen3-14b-instruct` | 14B | instruction | 8 | 1 | Medium instruction-following model, good balance |
| `qwen3-8b-instruct` | 8B | instruction | 5 | 1 | Fast instruction-following model for simple tasks |
| `qwen3-coder-14b` | 14B | coder | 8 | 1 | Specialized for code generation and programming |

## 🚀 Usage

### Model Management CLI

#### List Available Models
```bash
python model_manager.py list
python model_manager.py list --details
```

#### Download a Model
```bash
python model_manager.py download qwen3-14b-instruct
python model_manager.py download qwen3-30b-instruct --force
```

#### Switch Active Model
```bash
python model_manager.py switch qwen3-14b-instruct
```

#### Get Model Information
```bash
python model_manager.py info qwen3-14b-instruct
```

#### Check Memory Requirements
```bash
python model_manager.py check qwen3-30b-instruct
```

#### System Status
```bash
python model_manager.py status
```

#### Clean Up Incomplete Downloads
```bash
python model_manager.py cleanup
```

### API Server

#### Start Server with Default Configuration
```bash
python openai_server.py
```

#### Start with Custom Configuration
```bash
python openai_server.py --config models_config.json
```

#### Start with Specific Model
```bash
python openai_server.py --model qwen3-14b-instruct
```

#### Start with Custom Host/Port
```bash
python openai_server.py --host 0.0.0.0 --port 8000
```

## 🔌 API Endpoints

### Standard OpenAI Endpoints

#### List Models
```bash
curl http://localhost:8080/v1/models
```

#### Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

#### Streaming Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

#### Tool Calling
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-instruct",
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
    ]
  }'
```

### Admin Endpoints

#### Switch Model
```bash
curl -X POST http://localhost:8080/admin/switch_model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "qwen3-30b-instruct"}'
```

#### Get Model Status
```bash
curl http://localhost:8080/admin/model_status
```

#### Get GPU Usage
```bash
curl http://localhost:8080/admin/gpu_usage
```

## 🔧 Tool Calling

The server supports Qwen3's XML tool calling format and converts it to OpenAI's JSON format.

### Qwen3 XML Format
```xml
<tool_call>
<function=calculate>
<parameter=expression>
15 * 7
</parameter>
</function>
</tool_call>
```

### OpenAI JSON Conversion
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "calculate",
        "arguments": "{\"expression\": \"15 * 7\"}"
      }
    }
  ]
}
```

## 📊 Monitoring and Logging

### Log Files
- `logs/qwen3_server.log`: General application logs
- `logs/qwen3_server_errors.log`: Error logs only

### Log Levels
- **ERROR**: Model loading failures, GPU errors, API errors
- **WARNING**: Memory usage high, model switching, download progress
- **INFO**: Server startup, model loaded, request processing
- **DEBUG**: Detailed inference steps, tool parsing

### GPU Monitoring
The server provides real-time GPU monitoring through:
- Memory usage tracking
- Utilization monitoring
- Temperature monitoring
- Automatic memory validation before model loading

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Check GPU memory
python model_manager.py status

# Check specific model requirements
python model_manager.py check qwen3-30b-instruct
```

#### CUDA/GPU Issues
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU devices
nvidia-smi
```

#### Download Issues
```bash
# Clean up incomplete downloads
python model_manager.py cleanup

# Force re-download
python model_manager.py download qwen3-14b-instruct --force
```

#### Memory Issues
```bash
# Check available memory
python model_manager.py status

# Switch to smaller model
python model_manager.py switch qwen3-8b-instruct
```

### Performance Optimization

#### llama.cpp Settings
The server automatically adjusts llama.cpp settings based on model size:

- **30B models**: Reduced context (2048), smaller batch size (256)
- **14B models**: Standard context (4096), medium batch size (512)
- **8B models**: Extended context (8192), larger batch size (1024)

#### Multi-GPU Distribution
- llama.cpp automatically distributes models across available GPUs
- No manual configuration required
- Optimal memory utilization across all GPUs

## 🔄 Model Switching

The server supports hot-swapping between models without restart:

```bash
# Switch model via CLI
python model_manager.py switch qwen3-30b-instruct

# Switch model via API
curl -X POST http://localhost:8080/admin/switch_model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "qwen3-30b-instruct"}'
```

## 🚀 Performance

### Expected Performance (RTX 3060 + RTX 3080 Ti)

| Model | Tokens/sec | Memory Usage | Load Time |
|-------|------------|--------------|-----------|
| 8B | ~50-80 | ~5GB | ~10s |
| 14B | ~30-50 | ~8GB | ~15s |
| 30B | ~15-25 | ~18GB | ~30s |

*Performance may vary based on system configuration and model quantization.*

## 🔒 Security

### API Security
- CORS enabled for cross-origin requests
- Input validation with Pydantic models
- Error handling with proper HTTP status codes
- Request ID tracking for debugging

### Model Security
- Local model storage (no cloud dependencies)
- Secure model validation before loading
- Memory isolation between models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient model inference
- [Unsloth](https://github.com/unslothai/unsloth) for optimized Qwen3 models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for the API specification

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Open an issue on GitHub with detailed information

---

**Happy coding with Qwen3! 🚀**
