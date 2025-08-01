#!/usr/bin/env python3
"""
OpenAI-compatible API server for Qwen3 multi-GPU server
"""

import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Generator, Union
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from jinja2 import Environment, FileSystemLoader, select_autoescape

from utils.logging_config import setup_logging
from utils.gpu_monitor import get_gpu_monitor
from llama_backend import get_model_manager, get_llama_backend
from tool_parser import get_tool_parser, get_tool_validator

# Set up logging
logger = setup_logging()

# Pydantic models for API
class ChatMessage(BaseModel):
    class Config:
        extra = "allow"
    role: str = Field(..., description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Content of the message (string or array of parts)")
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
    class Config:
        extra = "allow"
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    tools: Optional[List[Tool]] = Field(None, description="Available tools")
    
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description='Tool choice ("auto", "none", or specific function)')
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(4096, description="Maximum tokens to generate")
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


class TemplateManager:
    """Manages Jinja2 templates for prompt generation"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        try:
            self.tool_template = self.env.get_template('qwen_tool_calling.j2')
            logger.info(f"Loaded Jinja template from {template_dir}/qwen_tool_calling.j2")
        except Exception as e:
            logger.error(f"Failed to load Jinja template: {e}")
            self.tool_template = None
    
    def render_prompt(self, messages: List[Dict], tools: Optional[List[Dict]] = None, add_generation_prompt: bool = True) -> str:
        """Render prompt using Jinja template"""
        if not self.tool_template:
            raise RuntimeError("Jinja template not loaded")
        
        try:
            return self.tool_template.render(
                messages=messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt
            )
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise RuntimeError(f"Template rendering failed: {e}")
    
    def is_available(self) -> bool:
        """Check if template system is available"""
        return self.tool_template is not None

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
        
        # Initialize template manager
        template_dir = self.config.get("server", {}).get("template_dir", "templates")
        self.template_manager = TemplateManager(template_dir)
        
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
        
        # Add validation error handler
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            try:
                body_bytes = await request.body()
                body_text = body_bytes.decode("utf-8") if body_bytes else "<empty>"
            except Exception:
                body_text = "<unavailable>"
            logger.error("422 Validation Error: %s\nRequest body: %s", exc.errors(), body_text)
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

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
                model_info = {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen3-server",
                    "description": model_config.get("description", ""),
                    "context_window": {
                        "effective_tokens": model_config.get("effective_context_tokens", 32768),
                        "max_tokens": model_config.get("max_context_tokens", 262144),
                        "note": "Effectiveness may decrease beyond effective_tokens, but model supports up to max_tokens"
                    }
                }
                models.append(model_info)
            
            return {
                "object": "list",
                "data": models
            }
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """Chat completions endpoint"""
            try:
                # Debug: Log full request
                request_dump = {
                    "model": request.model,
                    "messages": [msg.model_dump() for msg in request.messages],
                    "tools": [tool.model_dump() for tool in request.tools] if request.tools else None,
                    "tool_choice": request.tool_choice,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": request.stream
                }
                logger.info(f"=== CHAT COMPLETION REQUEST ===")
                logger.info(f"REQUEST: {json.dumps(request_dump, indent=2)}")
                
                # Check if model is loaded
                if not self.backend.model:
                    raise HTTPException(status_code=503, detail="No model loaded")
                
                # Prepare messages
                messages = [msg.model_dump() for msg in request.messages]
                
                # Prepare tools
                tools = None
                if request.tools:
                    tools = [tool.model_dump() for tool in request.tools]
                    # Debug: Log tool names
                    tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
                    logger.info(f"Tools provided by client: {tool_names}")
                
                # Generate prompt
                prompt = self._create_prompt(messages, tools)
                
                # Debug: Log generated prompt (truncated for readability)
                logger.info(f"=== GENERATED PROMPT ===")
                logger.info(f"PROMPT (first 500 chars): {prompt[:500]}...")
                logger.info(f"PROMPT (last 500 chars): ...{prompt[-500:]}")
                logger.info(f"PROMPT LENGTH: {len(prompt)} characters")
                
                # Check context window usage and warn if approaching limits
                prompt_tokens = len(prompt.split())  # Approximate token count
                current_model_config = self.config["models"].get(self.config.get("active_model", ""), {})
                effective_context = current_model_config.get("effective_context_tokens", 32768)
                max_context = current_model_config.get("max_context_tokens", 262144)
                
                if prompt_tokens > effective_context:
                    logger.warning(f"Prompt length ({prompt_tokens} tokens) exceeds effective context window ({effective_context} tokens). Model performance may degrade.")
                
                if prompt_tokens > max_context:
                    raise HTTPException(status_code=400, detail=f"Prompt length ({prompt_tokens} tokens) exceeds maximum supported context window ({max_context} tokens)")
                
                # Prepare generation parameters
                generation_params = {
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": request.top_p,
                }
                
                # Debug: Log generation parameters
                logger.info(f"=== GENERATION PARAMETERS ===")
                logger.info(f"max_tokens: {request.max_tokens}")
                logger.info(f"temperature: {request.temperature}")
                logger.info(f"top_p: {request.top_p}")
                
                if request.stop:
                    generation_params["stop"] = request.stop
                
                # Generate response
                if request.stream:
                    logger.info(f"=== USING STREAMING RESPONSE ===")
                    return StreamingResponse(
                        self._generate_stream(prompt, generation_params),
                        media_type="text/event-stream"
                    )
                else:
                    logger.info(f"=== USING NON-STREAMING RESPONSE ===")
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
        """Create prompt from messages and tools using Jinja template or legacy method"""
        use_jinja = self.config.get("server", {}).get("use_jinja_template", True)
        
        if use_jinja and self.template_manager.is_available():
            try:
                return self.template_manager.render_prompt(messages, tools, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Jinja template failed, falling back to legacy: {e}")
                return self._create_prompt_legacy(messages, tools)
        else:
            return self._create_prompt_legacy(messages, tools)
    
    def _create_prompt_legacy(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
        """Legacy prompt creation method (fallback)"""
        prompt_parts = []
        
        # Add system message if present
        if messages and messages[0]["role"] == "system":
            prompt_parts.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>")
            messages = messages[1:]
        
        # Add conversation messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                # Concatenate string parts if message content is array of objects per OpenAI v1.2
                combined_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        combined_parts.append(part.get("text", ""))
                content = "".join(combined_parts)
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
            # Debug: Log entry into completion function
            logger.info(f"=== ENTERING _generate_completion ===")
            logger.info(f"Generation params: {generation_params}")
            
            # Generate response
            logger.info(f"=== CALLING backend.generate ===")
            response_text = self.backend.generate(prompt, **generation_params)
            logger.info(f"=== backend.generate COMPLETED ===")
            
            # Debug: Log raw model response
            logger.info(f"=== RAW MODEL RESPONSE ===")
            logger.info(f"RESPONSE: {repr(response_text)}")
            
            # Parse tool calls
            tool_calls = self.tool_parser.extract_tool_calls(response_text)
            clean_text = self.tool_parser.clean_text(response_text)
            
            # Debug: Log parsing results
            logger.info(f"=== PARSING RESULTS ===")
            logger.info(f"TOOL CALLS FOUND: {len(tool_calls)}")
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    logger.info(f"TOOL {i}: {json.dumps(tc, indent=2)}")
            logger.info(f"CLEAN TEXT: {repr(clean_text)}")
            
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
            
            # Debug: Log final response
            final_response = response.model_dump()
            logger.info(f"=== FINAL RESPONSE ===")
            logger.info(f"RESPONSE: {json.dumps(final_response, indent=2)}")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _generate_stream(self, prompt: str, generation_params: Dict) -> Generator[str, None, None]:
        """Generate streaming response with optimized tool call detection"""
        try:
            # Add streaming parameter
            generation_params["stream"] = True
            
            # Simplified buffer strategy - Jinja template ensures predictable format
            buffer = ""
            tool_emitted = False
            
            # Generate with streaming
            for chunk in self.backend.generate_stream(prompt, **generation_params):
                # Accumulate into buffer for tool call detection
                buffer += chunk
                
                # Check for complete tool calls in buffer
                tool_calls = self.tool_parser.extract_tool_calls(buffer)
                
                if tool_calls and not tool_emitted:
                    # Extract clean visible text before tool calls
                    visible = self.tool_parser.clean_text(buffer)
                    
                    # Emit any preceding content
                    if visible.strip():
                        content_frame = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self.config.get("active_model", "unknown"),
                            "choices": [{
                                "index": 0,
                                "delta": {"content": visible},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(content_frame)}\n\n"
                    
                    # Emit tool calls (Jinja template should ensure only one call set per response)
                    tc_frame = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.config.get("active_model", "unknown"),
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": tool_calls},
                            "finish_reason": "tool_calls"
                        }]
                    }
                    yield f"data: {json.dumps(tc_frame)}\n\n"
                    
                    tool_emitted = True
                    buffer = ""  # Clear buffer after tool emission
                    continue
                
                # Stream text chunks normally when no tool calls detected
                if not tool_emitted:
                    text_frame = {
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
                    yield f"data: {json.dumps(text_frame)}\n\n"
            
            # Send final chunk only if no tool calls were emitted
            if not tool_emitted:
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
    parser.add_argument("--context-window", type=int, help="Override context window size (for debugging)")
    
    args = parser.parse_args()
    
    try:
        # Create server
        server = Qwen3APIServer(args.config)
        
        # Override context window if specified
        if args.context_window:
            logger.info(f"Overriding context window to {args.context_window} tokens")
            server.backend.llama_settings["n_ctx"] = args.context_window
        
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