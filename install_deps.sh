#!/bin/bash
set -e

echo "Installing core dependencies..."
pip install -r requirements.txt

echo "Installing FlashAttention with version constraint for vLLM compatibility..."
pip install "flash-attn>=2.7.1,<=2.8.0" --no-build-isolation

echo "All dependencies installed successfully!"