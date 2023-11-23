#!/bin/bash
set -e

# Path to where the model will be stored
MODEL_PATH="/app/models/zephyr-7b-beta.Q6_K.gguf"
MODEL_URL="https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q6_K.gguf"

# Check if the model file exists and download it if not
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found, downloading..."
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
fi

# Execute the main command (e.g., running a Python script)
exec "$@"

