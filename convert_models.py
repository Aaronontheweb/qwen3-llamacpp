#!/usr/bin/env python3
"""
Model conversion utilities for converting between different formats
Note: Most conversions will need to be done on the Linux server with proper tools
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
    """Utility class for model format conversions"""
    
    def __init__(self, config_path: str = "models_config.json"):
        """Initialize converter with model configuration"""
        self.config = self._load_config(config_path)
        self.models_dir = self.config.get("download_path", "./models")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load model configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def check_model_formats(self):
        """Check available model formats"""
        logger.info("Checking available model formats...")
        models = self.config.get("models", {})
        
        for model_id, model_config in models.items():
            logger.info(f"\nModel: {model_id}")
            
            # Check for HuggingFace model
            hf_name = model_config.get("name")
            if hf_name:
                hf_path = Path(self.models_dir) / hf_name.replace("/", "_")
                if hf_path.exists():
                    logger.info(f"  ✓ HuggingFace format available: {hf_path}")
                else:
                    logger.info(f"  ✗ HuggingFace format not found locally (will download): {hf_name}")
            
            # Check for GGUF model
            gguf_name = model_config.get("gguf_name")
            if gguf_name:
                gguf_path = Path(self.models_dir) / gguf_name.replace("/", "_")
                if gguf_path.exists():
                    # Look for actual GGUF files
                    gguf_files = list(gguf_path.rglob("*.gguf"))
                    if gguf_files:
                        logger.info(f"  ✓ GGUF format available: {gguf_files[0]}")
                    else:
                        logger.info(f"  ✗ GGUF directory exists but no .gguf files found: {gguf_path}")
                else:
                    logger.info(f"  ✗ GGUF format not found: {gguf_path}")
    
    def prepare_download_commands(self):
        """Generate commands to download models from HuggingFace"""
        logger.info("\nGenerating download commands for missing models...")
        models = self.config.get("models", {})
        commands = []
        
        for model_id, model_config in models.items():
            hf_name = model_config.get("name")
            if hf_name:
                hf_path = Path(self.models_dir) / hf_name.replace("/", "_")
                if not hf_path.exists():
                    # Generate huggingface-cli download command
                    cmd = f"huggingface-cli download {hf_name} --local-dir {hf_path}"
                    
                    # Add quantization-specific parameters
                    quantization = model_config.get("quantization")
                    if quantization:
                        cmd += f" --revision {quantization.lower()}"
                    
                    commands.append(f"# Download {model_id}")
                    commands.append(cmd)
                    commands.append("")
        
        if commands:
            script_path = "download_models.sh"
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# Model download script for Linux server\n\n")
                f.write("# Ensure huggingface-cli is installed\n")
                f.write("pip install huggingface-hub\n\n")
                f.write("# Create models directory\n")
                f.write(f"mkdir -p {self.models_dir}\n\n")
                f.write("\n".join(commands))
            
            logger.info(f"Download script created: {script_path}")
            logger.info("Run this script on your Linux server to download models")
        else:
            logger.info("All models appear to be available locally")
    
    def generate_conversion_notes(self):
        """Generate notes for manual model conversion if needed"""
        notes = """
# Model Conversion Notes

## GGUF to HuggingFace Conversion

If you need to convert GGUF models to HuggingFace format, you'll need to:

1. On the Linux server, install conversion tools:
   ```bash
   pip install transformers accelerate safetensors
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make
   ```

2. Convert GGUF to PyTorch (if needed):
   ```bash
   python convert-gguf-to-torch.py model.gguf --outfile model.pt
   ```

3. Convert to HuggingFace format:
   ```bash
   python convert_to_hf.py --model-path model.pt --output-dir ./hf_model
   ```

## Using Quantized Models with vLLM

vLLM supports several quantization formats:
- AWQ: Best performance, requires AWQ-quantized models
- GPTQ: Good compatibility, requires GPTQ-quantized models
- SqueezeLLM: Experimental

To use quantized models:
1. Download pre-quantized models from HuggingFace
2. Specify quantization in vLLM config:
   ```python
   "quantization": "awq"  # or "gptq"
   ```

## Direct Usage Without Conversion

For most cases, you don't need to convert:
- vLLM can download and use HuggingFace models directly
- Models are cached after first download
- Quantized versions are available on HuggingFace

## Recommended Models for vLLM

Best performance with vLLM:
1. Qwen/Qwen2.5-32B-Instruct-AWQ (4-bit AWQ)
2. Qwen/Qwen2.5-32B-Instruct-GPTQ (4-bit GPTQ)
3. Qwen/Qwen2.5-7B-Instruct (FP16, smaller model)
"""
        
        notes_path = "CONVERSION_NOTES.md"
        with open(notes_path, 'w') as f:
            f.write(notes)
        
        logger.info(f"Conversion notes saved to: {notes_path}")
    
    def validate_config(self):
        """Validate model configuration for vLLM compatibility"""
        logger.info("\nValidating model configuration...")
        models = self.config.get("models", {})
        
        for model_id, model_config in models.items():
            issues = []
            
            # Check for HF model name
            if not model_config.get("name"):
                issues.append("Missing HuggingFace model name")
            
            # Check context size
            max_context = model_config.get("max_context_tokens", 0)
            if max_context > 131072:
                issues.append(f"Context size {max_context} may be too large for available memory")
            
            # Check quantization
            quantization = model_config.get("quantization")
            if quantization and quantization not in ["AWQ", "GPTQ", "SqueezeLLM", None]:
                issues.append(f"Unknown quantization format: {quantization}")
            
            if issues:
                logger.warning(f"{model_id}: {', '.join(issues)}")
            else:
                logger.info(f"{model_id}: ✓ Configuration valid")


def main():
    """Main conversion utility"""
    converter = ModelConverter()
    
    print("=" * 60)
    print("Model Format Checker and Converter")
    print("=" * 60)
    
    # Check current model formats
    converter.check_model_formats()
    
    print("\n" + "=" * 60)
    
    # Validate configuration
    converter.validate_config()
    
    print("\n" + "=" * 60)
    
    # Generate download commands
    converter.prepare_download_commands()
    
    # Generate conversion notes
    converter.generate_conversion_notes()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Copy download_models.sh to your Linux server")
    print("2. Run: chmod +x download_models.sh && ./download_models.sh")
    print("3. Models will be downloaded in HuggingFace format")
    print("4. vLLM will use them directly without conversion")
    print("=" * 60)


if __name__ == "__main__":
    main()