# Installation Guide

## Python 3.13 Compatibility Issues

If you're using Python 3.13 and encountering installation errors, some packages may not have pre-built wheels yet. Here are solutions:

### Option 1: Use Python 3.11 or 3.12 (Recommended)

The easiest solution is to use Python 3.11 or 3.12, which have full package support:

1. **Install Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/)
2. **Create a virtual environment** with the compatible Python version:
   ```bash
   # Windows
   py -3.11 -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3.11 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Option 2: Install Packages Individually (Python 3.13)

If you must use Python 3.13, try installing packages in this order:

```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Core packages (usually work)
pip install fastapi uvicorn[standard] python-multipart aiofiles pydantic requests

# Document processing
pip install pypdf2 python-docx

# ChromaDB (may need special handling)
pip install chromadb --no-build-isolation

# Sentence transformers (will install torch automatically)
pip install sentence-transformers

# If sentence-transformers fails, install torch separately:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers safetensors huggingface-hub
pip install sentence-transformers --no-deps
```

### Option 3: Use Alternative Packages

If ChromaDB or sentence-transformers fail, you can use alternatives:

#### Alternative to ChromaDB: FAISS
```bash
pip install faiss-cpu
```
Then modify `rag_system.py` to use FAISS instead of ChromaDB.

#### Alternative Embeddings: OpenAI Embeddings (requires API key)
```bash
pip install openai
```
Modify `rag_system.py` to use OpenAI embeddings instead of sentence-transformers.

### Troubleshooting Specific Errors

#### Error: "No matching distribution found"
- **Solution**: The package may not support Python 3.13 yet. Use Python 3.11 or 3.12.

#### Error: "Failed building wheel"
- **Solution**: Install build dependencies:
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install --upgrade build
  ```

#### Error with ChromaDB
- **Solution**: Try installing with no build isolation:
  ```bash
  pip install chromadb --no-build-isolation
  ```
  Or install from source:
  ```bash
  pip install git+https://github.com/chroma-core/chroma.git
  ```

#### Error with sentence-transformers or torch
- **Solution**: Install CPU-only torch first:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers
  ```

#### Error: "Microsoft Visual C++ 14.0 or greater is required"
- **Solution**: Install Microsoft C++ Build Tools from:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Verify Installation

After installation, run:
```bash
python check_setup.py
```

This will verify that all packages are installed correctly.

### Minimal Installation (Without LLM)

If you want to test the system without Ollama/LLM (it will use fallback responses):

```bash
pip install fastapi uvicorn[standard] python-multipart aiofiles pydantic requests pypdf2 python-docx
```

The system will work but will use simpler fallback responses instead of LLM-generated answers.


