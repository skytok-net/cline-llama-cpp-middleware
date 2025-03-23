@echo off
echo Setting up environment for Kokab middleware...

:: Check if uv is installed
python -m pip show uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo uv is not installed. Please install it first:
    echo pip install uv
    exit /b 1
)

:: Create virtual environment
set ENV_NAME=kokab-env
echo Creating virtual environment: %ENV_NAME%
uv venv %ENV_NAME%

echo Virtual environment created at: %CD%\%ENV_NAME%
echo Activate with: %ENV_NAME%\Scripts\activate

:: Install dependencies
echo Installing dependencies...
uv pip install -r requirements.txt --no-cache

echo.
echo Environment setup complete!
echo.
echo To start the middleware server:
echo 1. Activate the environment: %ENV_NAME%\Scripts\activate
echo 2. Run the server: python middleware.py
echo.
echo Make sure your llama.cpp server is running at http://localhost:8083/completion