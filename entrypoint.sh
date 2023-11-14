#!/bin/bash
set -e

# Path to where the model will be stored
MODEL_PATH="/app/models/llama-2-7b-chat.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Check if the model file exists and download it if not
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found, downloading..."
    curl -o "$MODEL_PATH" "$MODEL_URL"
fi

# Execute the main command (e.g., running a Python script)
exec "$@"
