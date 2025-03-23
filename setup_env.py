#!/usr/bin/env python3

import os
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=True, 
            text=True, 
            capture_output=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)

def main():
    print("Setting up environment for Kokab middleware...")
    
    # Create a virtual environment using uv
    env_name = "kokab-env"
    print(f"Creating virtual environment: {env_name}")
    
    try:
        # Check if uv is installed
        subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv is not installed. Please install it first:")
        print("pip install uv")
        sys.exit(1)
    
    # Create virtual environment
    run_command(["uv", "venv", env_name])
    
    # Determine the activate script path based on OS
    if sys.platform == "win32":
        activate_script = os.path.join(env_name, "Scripts", "activate")
    else:
        activate_script = os.path.join(env_name, "bin", "activate")
    
    print(f"Virtual environment created at: {os.path.abspath(env_name)}")
    print(f"Activate with: source {activate_script}")
    
    # Create requirements.txt file with dependencies
    requirements = [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "httpx>=0.24.1",
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("Created requirements.txt with necessary dependencies")
    
    # Install dependencies using uv
    print("Installing dependencies...")
    run_command(["uv", "pip", "install", "-r", "requirements.txt", "--no-cache"])
    
    print("\nEnvironment setup complete!")
    print("\nTo start the middleware server:")
    print(f"1. Activate the environment: source {activate_script}")
    print("2. Run the server: python middleware.py")
    print("\nMake sure your llama.cpp server is running at http://localhost:8083/completion")

if __name__ == "__main__":
    main()