# Download llama3.2 Model

If you don't see `llama3.2` in your `ollama list`, you need to download it.

## Quick Steps

1. **Make sure Ollama is installed and running**
   - If you haven't installed Ollama yet: https://ollama.ai
   - Start Ollama: `ollama serve` (or it may run automatically as a service)

2. **Download the model**
   ```bash
   ollama pull llama3.2
   ```

3. **Verify it's installed**
   ```bash
   ollama list
   ```
   You should now see `llama3.2` in the list.

## Troubleshooting

### "ollama is not recognized"
**Solution**: Ollama is not in your PATH or not installed.

**Option 1**: Install Ollama
- Download from: https://ollama.ai
- Run the installer
- Restart your terminal

**Option 2**: Find Ollama installation path
- Default Windows location: `C:\Users\YourUsername\AppData\Local\Programs\Ollama\`
- Or check: `C:\Program Files\Ollama\`
- Add it to PATH or use full path: `C:\Users\YourUsername\AppData\Local\Programs\Ollama\ollama.exe pull llama3.2`

### "Connection refused" or "Could not connect"
**Solution**: Ollama server is not running.

1. Start Ollama server:
   ```bash
   ollama serve
   ```
   Keep this terminal window open.

2. In a NEW terminal, run:
   ```bash
   ollama pull llama3.2
   ```

### Download is slow
- This is normal - the model is ~2GB
- First download takes time depending on your internet speed
- Subsequent uses are instant (model is cached locally)

## Alternative Models

If you want a smaller/faster model:

```bash
# Smaller model (1.3GB, faster)
ollama pull llama3.2:1b

# Or use a different model
ollama pull mistral
ollama pull phi3
```

Then update your environment variable:
```bash
set LLM_MODEL=llama3.2:1b
```

## Verify Installation

After downloading, verify:

```bash
# List all models
ollama list

# Test the model
ollama run llama3.2
```

Type a message and press Enter. Type `/bye` to exit.

## Check from Python

You can also verify from your Python setup:

```bash
python check_setup.py
```

This will check if Ollama is running and if llama3.2 is available.


