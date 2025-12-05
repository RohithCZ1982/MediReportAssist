#!/bin/bash
# Ollama Setup Script for Linux/Mac

echo "========================================"
echo "Ollama Setup for Patient Discharge Assistant"
echo "========================================"
echo

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] Ollama is not installed"
    echo
    echo "Please install Ollama from: https://ollama.ai"
    echo "Or run: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "[OK] Ollama is installed"
echo

# Check Ollama version
echo "Checking Ollama version..."
ollama --version
echo

# Check if Ollama server is running
echo "Checking if Ollama server is running..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[OK] Ollama server is running"
else
    echo "[WARNING] Ollama server is not running"
    echo
    echo "Starting Ollama server in background..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
    
    # Check again
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama server started"
    else
        echo "[ERROR] Could not start Ollama server"
        echo
        echo "Please start Ollama manually:"
        echo "  1. Run: ollama serve"
        echo "  2. Keep that terminal open"
        echo "  3. Then run this script again"
        exit 1
    fi
fi

echo

# List installed models
echo "Checking installed models..."
ollama list
echo

# Check if llama3.2 is installed
if ollama list | grep -q "llama3.2"; then
    echo "[OK] llama3.2 model is installed"
else
    echo "[INFO] llama3.2 model is not installed"
    echo
    read -p "Do you want to install llama3.2 now? (y/n): " install
    if [[ "$install" == "y" || "$install" == "Y" ]]; then
        echo
        echo "Downloading llama3.2 model (this may take several minutes)..."
        ollama pull llama3.2
        if [ $? -eq 0 ]; then
            echo "[OK] llama3.2 installed successfully"
        else
            echo "[ERROR] Failed to install llama3.2"
            exit 1
        fi
    else
        echo
        echo "You can install it later with: ollama pull llama3.2"
    fi
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "  1. Run: python check_setup.py"
echo "  2. Start the server: python start.py"
echo "  3. Open browser: http://localhost:8000"
echo


