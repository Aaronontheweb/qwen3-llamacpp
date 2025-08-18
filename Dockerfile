# Multi-stage Dockerfile for vLLM multi-GPU server
# Optimized for NVIDIA GPUs with CUDA 12.1

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Build vLLM and dependencies
FROM base AS builder

WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install optional optimizations
RUN pip install flash-attn --no-build-isolation || true
RUN pip install xformers || true

# Stage 3: Runtime image
FROM base AS runtime

# Create app user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . /app

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python3", "openai_server.py", "--host", "0.0.0.0", "--port", "8080"]