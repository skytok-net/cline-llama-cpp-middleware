# Kokab Middleware

A FastAPI middleware that provides a Claude-compatible API interface for llama.cpp server.

## Overview

The Kokab middleware service acts as a bridge between your application and the llama.cpp server, providing a Claude-compatible API interface. This allows you to use llama.cpp with applications designed for Claude's API without modification.

## Features

- Claude-compatible API interface
- FastAPI-based middleware
- Easy integration with llama.cpp server
- Automatic dependency management

## Setup

### Prerequisites

- Python 3.8+
- llama.cpp server running at http://localhost:8083/completion

### Manual Setup

```bash
# Install uv if not already installed
pip install uv

# Create a virtual environment
uv venv kokab-env

# Activate the virtual environment
source kokab-env/bin/activate  # On Windows: kokab-env\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Automated Setup

For convenience, setup scripts are provided:

- **Linux/MacOS**: Run `./setup.sh`
- **Windows**: Run `setup.bat`

## Usage

1. Ensure your llama.cpp server is running at http://localhost:8083/completion
2. Start the middleware server:

```bash
python middleware.py
```

3. Your application can now make Claude-compatible API calls to the middleware server.

## Service Management

To run the middleware as a service, use the provided `kokab.service` file.

## Contributing

Contributions are welcome! Please ensure all changes are properly tested before submitting a pull request.

## Client Configuration

To configure a client to use the Kokab middleware, follow these steps:

1. Set the base URL to `http://localhost:8000` (or the appropriate host/port if different)
2. Use the following endpoints:
   - POST `/v1/complete` - For text completion requests
   - POST `/v1/chat` - For chat-style interactions
3. The API accepts and returns JSON data in the same format as Claude's API

Example request:

```bash
curl -X POST http://localhost:8000/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

No authentication is required for local development. For production use, consider adding API key authentication.

## License

[MIT License](LICENSE)