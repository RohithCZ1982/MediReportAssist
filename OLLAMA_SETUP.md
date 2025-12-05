# Ollama Setup Guide

Ollama allows you to run large language models (LLMs) locally on your computer, ensuring complete data privacy and offline operation.

## What is Ollama?

Ollama is a tool that makes it easy to run LLMs locally. It handles model downloads, inference, and provides a simple API interface.

## Installation Steps

### Step 1: Download Ollama

1. **Visit the Ollama website**: https://ollama.ai
2. **Click "Download"** - it will detect your operating system automatically
3. **For Windows**: Download the `.exe` installer
4. **Run the installer** and follow the installation wizard

### Step 2: Verify Installation

Open a new terminal/command prompt and verify Ollama is installed:

```bash
ollama --version
```

You should see the version number (e.g., `ollama version is 0.1.x`)

### Step 3: Start Ollama Server

Ollama runs as a background service. To start it:

**Option A: Automatic (Recommended)**
- On Windows, Ollama usually starts automatically after installation
- It runs as a background service

**Option B: Manual Start**
```bash
ollama serve
```

This will start the server on `http://localhost:11434`

**Note**: Keep the terminal window open if you started it manually. If it's running as a service, you can close the terminal.

### Step 4: Download a Language Model

You need to download at least one LLM model. For this project, we recommend:

```bash
ollama pull llama3.2
```

This downloads the Llama 3.2 model (about 2GB). Other options:

- **Smaller/faster**: `ollama pull llama3.2:1b` (1 billion parameters, ~1.3GB)
- **Better quality**: `ollama pull llama3.2:3b` (3 billion parameters, ~2GB)
- **Larger/better**: `ollama pull llama3.1:8b` (8 billion parameters, ~4.7GB)

**First-time download**: This may take several minutes depending on your internet speed.

### Step 5: Verify Model is Installed

List all installed models:

```bash
ollama list
```

You should see your downloaded model(s) listed.

### Step 6: Test Ollama (Optional)

Test that Ollama is working correctly:

```bash
ollama run llama3.2
```

This opens an interactive chat. Type a message and press Enter. Type `/bye` to exit.

Or test via API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

## Integration with Patient Discharge Assistant

Once Ollama is set up, the Patient Discharge Instructions Assistant will automatically use it.

### Configuration

The system uses these default settings:
- **Model**: `llama3.2`
- **API URL**: `http://localhost:11434`

To use a different model, set an environment variable:

**Windows (PowerShell)**:
```powershell
$env:LLM_MODEL="llama3.1:8b"
```

**Windows (Command Prompt)**:
```cmd
set LLM_MODEL=llama3.1:8b
```

**Linux/Mac**:
```bash
export LLM_MODEL=llama3.1:8b
```

### Verify Integration

1. Start your Patient Discharge Assistant:
   ```bash
   python start.py
   ```

2. The console should show:
   ```
   ✓ Ollama is running
   ✓ Default model 'llama3.2' is available
   ```

   If you see warnings, check that:
   - Ollama is running (`ollama serve` or check Windows services)
   - The model is installed (`ollama list`)

## Troubleshooting

### Problem: "Could not connect to Ollama"

**Solution 1**: Start Ollama server
```bash
ollama serve
```

**Solution 2**: Check if Ollama is running
- Windows: Check Task Manager for "ollama" process
- Or visit: http://localhost:11434/api/tags

**Solution 3**: Restart Ollama
- Windows: Restart the Ollama service from Services (services.msc)
- Or restart your computer

### Problem: "Model not found"

**Solution**: Download the model
```bash
ollama pull llama3.2
```

Verify it's installed:
```bash
ollama list
```

### Problem: Slow responses

**Possible causes and solutions**:

1. **Model is too large for your system**
   - Use a smaller model: `ollama pull llama3.2:1b`

2. **Not enough RAM**
   - Close other applications
   - Use a smaller model
   - Check system requirements (usually 8GB+ RAM recommended)

3. **CPU-only inference (no GPU)**
   - This is normal - responses will be slower but still functional
   - For faster inference, consider a GPU-enabled setup

### Problem: Ollama uses too much disk space

**Solution**: Remove unused models
```bash
# List models
ollama list

# Remove a model
ollama rm model-name
```

### Problem: Port 11434 is already in use

**Solution**: Change Ollama port or stop conflicting service
```bash
# Set custom port (requires Ollama restart)
set OLLAMA_HOST=localhost:11435
```

Then update the assistant's config or environment variable:
```bash
set OLLAMA_BASE_URL=http://localhost:11435
```

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5-10GB free space (for models)
- **CPU**: Modern multi-core processor

### Recommended Requirements
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **GPU**: Optional but significantly faster (NVIDIA with CUDA support)

## Available Models

Popular models you can use:

| Model | Size | Best For |
|-------|------|----------|
| `llama3.2:1b` | ~1.3GB | Fast responses, limited RAM |
| `llama3.2` | ~2GB | Balanced (recommended) |
| `llama3.2:3b` | ~2GB | Better quality |
| `llama3.1:8b` | ~4.7GB | Higher quality, more RAM needed |
| `mistral` | ~4.1GB | Alternative to Llama |
| `phi3` | ~2.3GB | Microsoft's efficient model |

Browse more models: https://ollama.ai/library

## Advanced Usage

### Run Multiple Models

You can have multiple models installed and switch between them:

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull phi3
```

Then specify which one to use via the `LLM_MODEL` environment variable.

### Custom Model Configuration

Create a `Modelfile` for custom configurations:

```bash
ollama create my-custom-model -f Modelfile
```

### GPU Acceleration

If you have an NVIDIA GPU:

1. Install CUDA drivers
2. Ollama will automatically use GPU if available
3. Check GPU usage: `nvidia-smi` (if NVIDIA GPU)

## Security & Privacy

✅ **All processing happens locally** - no data sent to external servers
✅ **No internet required** after model download
✅ **Complete data privacy** - your documents never leave your computer
✅ **HIPAA-friendly** - suitable for healthcare applications

## Next Steps

After setting up Ollama:

1. ✅ Verify Ollama is running: `ollama list`
2. ✅ Download a model: `ollama pull llama3.2`
3. ✅ Test the assistant: `python check_setup.py`
4. ✅ Start the server: `python start.py`

## Additional Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **Model Library**: https://ollama.ai/library
- **Community**: https://github.com/ollama/ollama/discussions


