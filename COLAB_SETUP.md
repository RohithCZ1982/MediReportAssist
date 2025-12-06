# Running MediReportAssist in Google Colab

Yes, you can run this project in Google Colab! However, some modifications are needed since Colab has different constraints than a local environment.

## Challenges & Solutions

### Challenge 1: Ollama (Local LLM Server)
**Problem**: Ollama runs as a local server, which is complex to set up in Colab.

**Solutions**:
1. **Use Hugging Face Transformers** (Recommended for Colab)
   - Runs entirely in Python
   - No separate server needed
   - Works well with Colab's GPU

2. **Use OpenAI API** (Alternative)
   - Requires API key
   - Simple to implement
   - Costs money per request

3. **Install Ollama in Colab** (Advanced)
   - Possible but complex
   - Requires running a server in background

### Challenge 2: Exposing FastAPI
**Solution**: Use Colab's built-in public URL or ngrok

### Challenge 3: Persistent Storage
**Solution**: Use Google Drive for ChromaDB and uploads

## Quick Start: Colab Notebook

I'll create a Colab-compatible version that uses Hugging Face Transformers. Here's how to set it up:

### Option 1: Using Hugging Face Transformers (Recommended)

1. **Open a new Colab notebook**
2. **Install dependencies**:
```python
!pip install fastapi uvicorn python-multipart
!pip install chromadb sentence-transformers
!pip install pypdf2 python-docx
!pip install transformers torch accelerate
!pip install pyngrok  # For exposing the API
```

3. **Mount Google Drive** (for persistent storage):
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. **Upload project files** or clone from GitHub:
```python
!git clone https://github.com/yourusername/MediReportAssist.git
%cd MediReportAssist
```

5. **Use the Colab-compatible RAG system** (see `rag_system_colab.py`)

6. **Start the server with public URL**:
```python
from pyngrok import ngrok
import uvicorn
from threading import Thread

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Start FastAPI server
def run_server():
    uvicorn.run("app_colab:app", host="0.0.0.0", port=8000)

server_thread = Thread(target=run_server, daemon=True)
server_thread.start()
```

### Option 2: Using OpenAI API

If you prefer OpenAI (requires API key):

1. Set your API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

2. Use the OpenAI-compatible version (modify `rag_system.py`)

## Colab-Specific Considerations

### 1. GPU Acceleration
Colab provides free GPU access. Enable it:
- Runtime → Change runtime type → GPU

This speeds up:
- Embedding generation (Sentence Transformers)
- LLM inference (Hugging Face models)

### 2. Memory Limits
- Free Colab: ~12GB RAM
- Pro Colab: ~32GB RAM

**Recommendations**:
- Use smaller models (e.g., `llama-2-7b` instead of `llama-2-70b`)
- Use 8-bit quantization for models
- Clear variables when done: `del variable_name`

### 3. Session Timeout
- Free Colab: Sessions timeout after ~90 minutes of inactivity
- **Solution**: Keep the notebook active or use Colab Pro

### 4. File Persistence
- Colab files are deleted when session ends
- **Solution**: Save to Google Drive

## Modified Files for Colab

I'll create these Colab-compatible versions:
1. `rag_system_colab.py` - Uses Hugging Face instead of Ollama
2. `app_colab.py` - Modified FastAPI app for Colab
3. `colab_notebook.ipynb` - Complete Colab notebook

## Step-by-Step Setup

### Step 1: Create New Colab Notebook

1. Go to https://colab.research.google.com
2. File → New notebook

### Step 2: Install Dependencies

Run this in the first cell:
```python
!pip install -q fastapi uvicorn[standard] python-multipart
!pip install -q chromadb sentence-transformers
!pip install -q pypdf2 python-docx
!pip install -q transformers torch accelerate bitsandbytes
!pip install -q pyngrok
```

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
import os
os.makedirs('/content/drive/MyDrive/MediReportAssist', exist_ok=True)
%cd /content/drive/MyDrive/MediReportAssist
```

### Step 4: Upload Project Files

Option A: Clone from GitHub
```python
!git clone https://github.com/yourusername/MediReportAssist.git .
```

Option B: Upload manually
- Use Colab's file upload feature
- Upload all `.py` files and `static/` folder

### Step 5: Use Colab-Compatible Code

The Colab version will:
- Use Hugging Face Transformers for LLM
- Store data in Google Drive
- Expose API via ngrok

### Step 6: Start the Server

```python
from pyngrok import ngrok
import uvicorn
from threading import Thread
import time

# Authenticate ngrok (get token from https://dashboard.ngrok.com/get-started/your-authtoken)
# ngrok.set_auth_token("your-ngrok-token")  # Optional for free tier

# Start tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")
print(f"📱 Access your app at: {public_url}/static/index.html")

# Start server in background
def run_server():
    uvicorn.run("app_colab:app", host="0.0.0.0", port=8000, log_level="info")

server = Thread(target=run_server, daemon=True)
server.start()

# Wait for server to start
time.sleep(3)
print("✅ Server is running!")
print(f"🔗 API Docs: {public_url}/docs")
```

## Model Recommendations for Colab

### For Embeddings (Always use):
- `all-MiniLM-L6-v2` (default) - Small, fast, good quality

### For LLM (Choose based on available RAM):

**Free Colab (12GB RAM)**:
- `microsoft/DialoGPT-small` - Very small, fast
- `gpt2` - Small, widely used
- `facebook/opt-1.3b` - Better quality, still small

**Colab Pro (32GB RAM)**:
- `meta-llama/Llama-2-7b-chat-hf` - Good quality (requires Hugging Face access)
- `mistralai/Mistral-7B-Instruct-v0.1` - Excellent quality
- `google/flan-t5-large` - Good for instruction following

**With GPU (Recommended)**:
- Use 8-bit quantization to fit larger models
- `meta-llama/Llama-2-7b-chat-hf` with 8-bit
- `mistralai/Mistral-7B-Instruct-v0.1` with 8-bit

## Limitations in Colab

1. **Session Timeout**: Free Colab sessions timeout after inactivity
2. **Resource Limits**: Free tier has limited RAM and compute
3. **No Background Processes**: Server stops when notebook is closed
4. **File Persistence**: Files deleted when session ends (use Drive)

## Troubleshooting

### Problem: Out of Memory
**Solution**:
- Use smaller models
- Enable 8-bit quantization
- Clear variables: `del large_variable`
- Restart runtime: Runtime → Restart runtime

### Problem: ngrok URL not working
**Solution**:
- Check if server is running: `!curl http://localhost:8000`
- Verify ngrok tunnel: Check ngrok dashboard
- Try different port: `ngrok.connect(8001)`

### Problem: Model loading fails
**Solution**:
- Use smaller models
- Load with 8-bit: `load_in_8bit=True`
- Use CPU if GPU fails: `device_map="cpu"`

## Next Steps

1. I'll create `rag_system_colab.py` - Colab-compatible RAG system
2. I'll create `app_colab.py` - Modified FastAPI app
3. I'll create a complete Colab notebook template

Would you like me to create these files now?

