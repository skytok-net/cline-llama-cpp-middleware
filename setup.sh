#!/bin/bash

set -e

echo "Setting up environment for Kokab middleware..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it first:"
    echo "pip install uv"
    exit 1
fi

# Create virtual environment
ENV_NAME="kokab-env"
echo "Creating virtual environment: $ENV_NAME"
uv venv "$ENV_NAME"

# Determine activate script path
ACTIVATE_SCRIPT="$ENV_NAME/bin/activate"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    ACTIVATE_SCRIPT="$ENV_NAME/Scripts/activate"
fi

echo "Virtual environment created at: $(pwd)/$ENV_NAME"
echo "Activate with: source $ACTIVATE_SCRIPT"

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt --no-cache

echo ""
echo "Environment setup complete!"
echo ""
echo "To start the middleware server:"
echo "1. Activate the environment: source $ACTIVATE_SCRIPT"
echo "2. Run the server: python middleware.py"
echo ""
echo "Make sure your llama.cpp server is running at http://localhost:8083/completion"