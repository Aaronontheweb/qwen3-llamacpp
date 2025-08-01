#!/usr/bin/env python3
"""
OpenAI-compatible API server for Qwen3 multi-GPU server
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Generator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from utils.logging_config import setup_logging
from utils.gpu_monitor import get_gpu_monitor
from llama_backend import get_model_manager, get_llama_backend
from tool_parser import get_tool_parser, get_tool_validator

# Set up logging
logger = setup_logging()

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the message sender")

class FunctionParameter(BaseModel):
    type: str = Field(..., description="Parameter type")
    description: Optional[str] = Field(None, description="Parameter description")
    enum: Optional[List[str]] = Field(None, description="Enum values")

class FunctionDefinition(BaseModel):
    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    parameters: Dict[str, Any] = Field(..., description="Function parameters")

class Tool(BaseModel):
    type: str = Field("function", description="Tool type")
    function: FunctionDefinition = Field(..., description="Function definition")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    tools: Optional[List[Tool]] = Field(None, description="Available tools")
    tool_choice: Optional[str] = Field(None, description="Tool choice")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(False, description="Stream response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class ModelSwitchRequest(BaseModel):
    model_id: str = Field(..., description="Model ID to switch to")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Response ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage")

class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field("qwen3-server", description="Model owner")

class Qwen3APIServer:
    """OpenAI-compatible API server for Qwen3 models"""
    
    def __init__(self, config_path: str = "models_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.model_manager = get_model_manager(self.config)
        self.backend = get_llama_backend(self.config)
        self.tool_parser = get_tool_parser()
        self.tool_validator = get_tool_validator()
        self.gpu_monitor = get_gpu_monitor()
        
        # Download tracking
        self.download_status = {}
        
        # Load active model
        active_model = self.config.get("active_model")
        if active_model:
            self.model_manager.load_model_by_id(active_model)
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Qwen3 Multi-GPU Server",
            description="OpenAI-compatible API server for Qwen3 models with multi-GPU support",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Set up routes
        self._setup_routes()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _setup_routes(self):
        """Set up API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "Qwen3 Multi-GPU Server",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models"""
            models = []
            for model_id, model_config in self.config["models"].items():
                models.append(ModelInfo(
                    id=model_id,
                    created=int(time.time()),
                    owned_by="qwen3-server"
                ).dict())
            
            return {
                "object": "list",
                "data": models
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Chat completions endpoint"""
            try:
                # Check if model is loaded
                if not self.backend.model:
                    raise HTTPException(status_code=503, detail="No model loaded")
                
                # Prepare messages
                messages = [msg.dict() for msg in request.messages]
                
                # Prepare tools
                tools = None
                if request.tools:
                    tools = [tool.dict() for tool in request.tools]
                
                # Generate prompt
                prompt = self._create_prompt(messages, tools)
                
                # Prepare generation parameters
                generation_params = {
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": request.top_p,
                }
                
                if request.stop:
                    generation_params["stop"] = request.stop
                
                # Generate response
                if request.stream:
                    return StreamingResponse(
                        self._generate_stream(prompt, generation_params),
                        media_type="text/plain"
                    )
                else:
                    return await self._generate_completion(prompt, generation_params, tools)
                    
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/admin/switch_model")
        async def switch_model(request: ModelSwitchRequest):
            """Switch to a different model"""
            try:
                # Check if model is downloaded first
                model_config = self.config["models"].get(request.model_id)
                if not model_config:
                    raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found in config")
                
                from utils.model_utils import is_model_downloaded
                if not is_model_downloaded(model_config["name"], self.config["download_path"]):
                    # Model not downloaded - start download in background
                    background_tasks.add_task(self._download_model_background, request.model_id)
                    return {
                        "status": "downloading", 
                        "message": f"Model {request.model_id} is being downloaded. Use /admin/download_status to check progress.",
                        "model_id": request.model_id
                    }
                
                # Model is downloaded, try to load it
                success = self.model_manager.load_model_by_id(request.model_id)
                if success:
                    self.config["active_model"] = request.model_id
                    # Save config
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    
                    return {"status": "success", "model": request.model_id}
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to load model: {request.model_id}")
                    
            except Exception as e:
                logger.error(f"Model switch error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/admin/model_status")
        async def model_status():
            """Get current model and system status"""
            try:
                status = self.model_manager.get_status()
                return status
            except Exception as e:
                logger.error(f"Status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/admin/gpu_usage")
        async def gpu_usage():
            """Get GPU memory and utilization"""
            try:
                return self.gpu_monitor.get_memory_usage_summary()
            except Exception as e:
                logger.error(f"GPU usage error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/admin/download_model")
        async def download_model(request: ModelSwitchRequest):
            """Download a model on-demand"""
            try:
                # Check if model is already downloaded
                model_config = self.config["models"].get(request.model_id)
                if not model_config:
                    raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found in config")
                
                from utils.model_utils import is_model_downloaded
                if is_model_downloaded(model_config["name"], self.config["download_path"]):
                    return {"status": "success", "message": f"Model {request.model_id} is already downloaded"}
                
                # Start download in background
                background_tasks.add_task(self._download_model_background, request.model_id)
                
                return {
                    "status": "started", 
                    "message": f"Download started for {request.model_id}",
                    "model_id": request.model_id
                }
                    
            except Exception as e:
                logger.error(f"Download error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/admin/download_status")
        async def download_status():
            """Get download status for all models"""
            try:
                return self.download_status
            except Exception as e:
                logger.error(f"Download status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _download_model_background(self, model_id: str):
        """Download a model in the background"""
        try:
            self.download_status[model_id] = {"status": "downloading", "progress": 0}
            
            # Create a temporary model manager for downloading
            temp_manager = get_model_manager(self.config)
            
            # Download the model
            success = temp_manager.download_model(model_id)
            
            if success:
                self.download_status[model_id] = {"status": "completed", "progress": 100}
                logger.info(f"Model {model_id} downloaded successfully")
            else:
                self.download_status[model_id] = {"status": "failed", "progress": 0}
                logger.error(f"Failed to download model {model_id}")
                
        except Exception as e:
            self.download_status[model_id] = {"status": "failed", "progress": 0, "error": str(e)}
            logger.error(f"Download error for {model_id}: {e}")
    
    def _create_prompt(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
        """Create prompt from messages and tools"""
        prompt_parts = []
        
        # Add system message if present
        if messages and messages[0]["role"] == "system":
            prompt_parts.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>")
            messages = messages[1:]
        
        # Add conversation messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Add tools if provided
        if tools:
            prompt_parts.append("\n\nYou have access to the following functions:\n\n")
            prompt_parts.append("<tools>")
            
            for tool in tools:
                function = tool["function"]
                prompt_parts.append(f"\n<function>")
                prompt_parts.append(f"<name>{function['name']}</name>")
                prompt_parts.append(f"<description>{function['description']}</description>")
                prompt_parts.append("<parameters>")
                
                for param_name, param_info in function["parameters"]["properties"].items():
                    prompt_parts.append(f"<parameter>")
                    prompt_parts.append(f"<name>{param_name}</name>")
                    prompt_parts.append(f"<type>{param_info['type']}</type>")
                    if "description" in param_info:
                        prompt_parts.append(f"<description>{param_info['description']}</description>")
                    prompt_parts.append("</parameter>")
                
                prompt_parts.append("</parameters>")
                prompt_parts.append("</function>")
            
            prompt_parts.append("\n</tools>")
        
        # Add assistant prefix
        prompt_parts.append("\n<|im_start|>assistant\n")
        
        return "".join(prompt_parts)
    
    async def _generate_completion(self, prompt: str, generation_params: Dict, tools: Optional[List[Dict]] = None) -> ChatCompletionResponse:
        """Generate a single completion"""
        try:
            # Generate response
            response_text = self.backend.generate(prompt, **generation_params)
            
            # Parse tool calls
            tool_calls = self.tool_parser.extract_tool_calls(response_text)
            clean_text = self.tool_parser.clean_text(response_text)
            
            # Prepare choice
            choice = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": clean_text
                },
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }
            
            if tool_calls:
                choice["message"]["tool_calls"] = tool_calls
            
            # Create response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=self.config.get("active_model", "unknown"),
                choices=[choice],
                usage={
                    "prompt_tokens": len(prompt.split()),  # Approximate
                    "completion_tokens": len(clean_text.split()),  # Approximate
                    "total_tokens": len(prompt.split()) + len(clean_text.split())
                }
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _generate_stream(self, prompt: str, generation_params: Dict) -> Generator[str, None, None]:
        """Generate streaming response"""
        try:
            # Add streaming parameter
            generation_params["stream"] = True
            
            # Generate with streaming
            for chunk in self.backend.generate_stream(prompt, **generation_params):
                # Create streaming response chunk
                chunk_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.config.get("active_model", "unknown"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None
                    }]
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send final chunk
            final_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.config.get("active_model", "unknown"),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "generation_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    def run(self, host: str = None, port: int = None):
        """Run the server"""
        server_config = self.config.get("server", {})
        host = host or server_config.get("host", "127.0.0.1")
        port = port or server_config.get("port", 8080)
        
        logger.info(f"Starting Qwen3 API server on {host}:{port}")
        logger.info(f"Active model: {self.config.get('active_model', 'None')}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 Multi-GPU API Server")
    parser.add_argument("--config", default="models_config.json", help="Configuration file path")
    parser.add_argument("--host", help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--model", help="Model ID to load")
    
    args = parser.parse_args()
    
    try:
        # Create server
        server = Qwen3APIServer(args.config)
        
        # Load specific model if requested
        if args.model:
            success = server.model_manager.load_model_by_id(args.model)
            if not success:
                logger.error(f"Failed to load model: {args.model}")
                return 1
        
        # Run server
        server.run(args.host, args.port)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 