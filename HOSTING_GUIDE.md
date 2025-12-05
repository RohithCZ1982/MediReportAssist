# Free Hosting Guide for Patient Discharge Instructions Assistant

This guide covers free hosting options for your application. Note that some modifications may be needed since the app uses local components (Ollama, ChromaDB).

## ⚠️ Important Considerations

Your current app uses:
- **Ollama** (local LLM) - Won't work on most free hosting platforms
- **ChromaDB** (local vector database) - Needs persistent storage
- **Sentence Transformers** - Large models need to be downloaded

**Solutions:**
1. Use cloud LLM APIs instead of Ollama
2. Use cloud vector databases
3. Simplify the stack for hosting

---

## 🆓 Free Hosting Options

### Option 1: Render (Recommended for FastAPI)

**Free Tier:**
- 750 hours/month (enough for 24/7)
- 512MB RAM
- Persistent disk storage
- Automatic SSL
- Sleeps after 15 minutes of inactivity (free tier)

**Setup Steps:**

1. **Create account**: https://render.com

2. **Prepare for deployment:**
   - Create `render.yaml` (see below)
   - Modify app to use cloud services instead of Ollama
   - Use environment variables for configuration

3. **Deploy:**
   - Connect GitHub repository
   - Render auto-detects FastAPI apps
   - Set environment variables

**Limitations:**
- App sleeps after inactivity (takes ~30 seconds to wake)
- Limited RAM (may need smaller models)
- No GPU access

**Cost:** Free (with limitations)

---

### Option 2: Railway

**Free Tier:**
- $5 credit/month (enough for small apps)
- 512MB RAM
- Persistent storage
- No sleep (always on)

**Setup:**
1. Sign up: https://railway.app
2. Connect GitHub repo
3. Railway auto-detects and deploys
4. Set environment variables

**Cost:** Free (with $5 monthly credit)

---

### Option 3: Fly.io

**Free Tier:**
- 3 shared-cpu VMs
- 256MB RAM per VM
- Persistent volumes available
- Global edge network

**Setup:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Sign up: https://fly.io
3. Run: `fly launch` in your project directory
4. Follow prompts

**Cost:** Free (with usage limits)

---

### Option 4: PythonAnywhere

**Free Tier:**
- 1 web app
- 512MB disk space
- Limited CPU time
- Always-on option available

**Setup:**
1. Sign up: https://www.pythonanywhere.com
2. Upload files via web interface or Git
3. Configure web app
4. Set up scheduled tasks if needed

**Cost:** Free (with limitations)

---

### Option 5: Hugging Face Spaces (Best for ML Apps)

**Free Tier:**
- Unlimited public spaces
- 16GB RAM
- Persistent storage
- Free GPU (limited hours)
- Auto-deploy from GitHub

**Setup:**
1. Create account: https://huggingface.co
2. Create new Space
3. Connect GitHub repo
4. Select "Docker" or "Gradio" template
5. Hugging Face provides free GPU for inference

**Cost:** Free (excellent for ML/AI apps)

---

## 🔧 Modifications Needed for Hosting

### Replace Ollama with Cloud LLM

Instead of local Ollama, use:

**Option A: OpenAI API** (not free, but cheap)
```python
# In rag_system.py
import openai

def generate_answer(self, query, contexts):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical assistant..."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

**Option B: Hugging Face Inference API** (Free tier available)
```python
import requests

def generate_answer(self, query, contexts):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()[0]["generated_text"]
```

**Option C: Groq API** (Very fast, free tier)
```python
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": prompt}]
)
```

### Replace ChromaDB with Cloud Vector DB

**Option A: Pinecone** (Free tier: 1 index)
```python
import pinecone

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone.Index("discharge-instructions")
```

**Option B: Weaviate Cloud** (Free sandbox)
```python
import weaviate

client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL"),
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
)
```

**Option C: Qdrant Cloud** (Free tier available)
```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
```

---

## 📝 Deployment Files Needed

### 1. `render.yaml` (for Render)

```yaml
services:
  - type: web
    name: discharge-assistant
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: LLM_API_KEY
        sync: false
      - key: VECTOR_DB_URL
        sync: false
```

### 2. `Dockerfile` (for Docker-based hosting)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. `.dockerignore`

```
__pycache__
*.pyc
venv/
env/
chroma_db/
uploads/
.git
.gitignore
*.md
```

### 4. `requirements-cloud.txt` (simplified for cloud)

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
aiofiles>=23.2.0
pydantic>=2.5.0
requests>=2.31.0
pypdf2>=3.0.0
python-docx>=1.1.0
openai>=1.0.0  # or groq, or huggingface-hub
pinecone-client>=2.2.0  # or qdrant-client, or weaviate-client
sentence-transformers>=2.2.0
```

---

## 🚀 Quick Start: Deploy to Render

### Step 1: Modify App for Cloud

Create `rag_system_cloud.py`:

```python
import os
import requests
from sentence_transformers import SentenceTransformer

class RAGSystemCloud:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Use cloud vector DB (Pinecone, Qdrant, etc.)
        # Use cloud LLM (OpenAI, Groq, Hugging Face)
    
    def generate_answer(self, query, contexts):
        # Use OpenAI or Groq instead of Ollama
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        prompt = f"""You are a medical assistant...
        Context: {contexts}
        Question: {query}
        Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
```

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/medical-report.git
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will auto-detect FastAPI
5. Set environment variables:
   - `OPENAI_API_KEY` (or `GROQ_API_KEY`)
   - `PINECONE_API_KEY` (or other vector DB)
6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)

---

## 💡 Recommended Free Stack

**Best Free Combination:**

1. **Hosting**: Hugging Face Spaces (free GPU) or Render
2. **LLM**: Groq API (free tier, very fast) or Hugging Face Inference
3. **Vector DB**: Pinecone (free tier) or Qdrant Cloud
4. **Embeddings**: Sentence Transformers (downloads on first run)

**Cost:** $0/month (within free tier limits)

---

## 🔐 Environment Variables

Set these in your hosting platform:

```bash
# LLM API
OPENAI_API_KEY=sk-...  # or GROQ_API_KEY, or HF_TOKEN

# Vector Database
PINECONE_API_KEY=...   # or QDRANT_API_KEY, or WEAVIATE_API_KEY
PINECONE_ENVIRONMENT=us-east-1

# App Settings
LLM_MODEL=gpt-3.5-turbo  # or llama3-8b-8192 for Groq
VECTOR_DB_TYPE=pinecone  # or qdrant, weaviate
```

---

## 📊 Comparison Table

| Platform | Free Tier | RAM | Storage | Always On | Best For |
|----------|-----------|-----|---------|-----------|----------|
| Render | 750 hrs/mo | 512MB | 1GB | No (sleeps) | FastAPI apps |
| Railway | $5 credit | 512MB | 1GB | Yes | Always-on apps |
| Fly.io | 3 VMs | 256MB | 3GB | Yes | Global apps |
| PythonAnywhere | Limited | 512MB | 512MB | Optional | Python apps |
| Hugging Face | Unlimited | 16GB | 50GB | Yes | ML/AI apps |

---

## 🎯 Quick Recommendation

**For this medical app, I recommend:**

1. **Hugging Face Spaces** - Best for ML apps, free GPU, easy deployment
2. **Render** - Easy FastAPI deployment, good free tier
3. **Railway** - Always-on, simple setup

**Modify the app to:**
- Use Groq API (free, fast) or Hugging Face Inference instead of Ollama
- Use Pinecone or Qdrant Cloud instead of local ChromaDB
- Keep sentence-transformers (downloads on first deploy)

Would you like me to create a cloud-ready version of your app?


