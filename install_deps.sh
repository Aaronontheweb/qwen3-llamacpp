#!/bin/bash
set -e

echo "Installing core dependencies..."
pip install -r requirements.txt

echo "Installing FlashAttention with --no-build-isolation..."
pip install flash-attn --no-build-isolation

echo "All dependencies installed successfully!"