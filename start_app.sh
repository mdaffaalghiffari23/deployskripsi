#!/bin/bash
# Startup script for Retinopati DiabeTEST

echo "🩺 Starting Retinopati DiabeTEST..."

# Set environment variables to reduce PyTorch warnings
export TORCH_HOME=/tmp/torch_cache
export PYTHONWARNINGS="ignore::UserWarning"

# Create torch cache directory
mkdir -p /tmp/torch_cache

# Check if using uv or regular python
if command -v uv &> /dev/null; then
    echo "✅ Using uv environment"
    uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none
else
    echo "✅ Using standard Python"
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none
fi
