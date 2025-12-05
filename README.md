# Patient Discharge Instructions Assistant

A lightweight, domain-focused system that allows patients to upload their discharge summaries and query them using natural language. The system leverages LLM with RAG (Retrieval Augmented Generation) to fetch the most relevant instructions from documents and present them in a simplified format.

## Features

- 📄 **Document Upload**: Support for PDF, TXT, and DOCX formats
- 🔍 **Natural Language Queries**: Ask questions in plain English
- 🤖 **RAG System**: Retrieves relevant context from discharge summaries
- 💬 **Query Manager**: Clarifies vague queries for better accuracy
- 🏠 **Local Deployment**: Works offline with local LLM (Ollama) for data confidentiality
- 🎨 **Modern UI**: Clean, patient-friendly interface

## Architecture

- **Backend**: FastAPI for REST API
- **RAG**: ChromaDB for vector storage, Sentence Transformers for embeddings
- **LLM**: Ollama for local LLM inference (default: llama3.2)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Prerequisites

1. **Python 3.11 or 3.12** (recommended)
   - ⚠️ **Note**: Python 3.13 may have compatibility issues with some packages
   - If using Python 3.13, see [INSTALL.md](INSTALL.md) for troubleshooting
2. **Ollama** (for local LLM)
   - Download from: https://ollama.ai
   - Install and start: `ollama serve`
   - Pull a model: `ollama pull llama3.2` (or another model of your choice)
   - 📖 **Detailed setup guide**: See [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for step-by-step instructions

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd MedicalReport
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   ⚠️ **If you encounter installation errors** (especially with Python 3.13), see [INSTALL.md](INSTALL.md) for detailed troubleshooting steps.

4. **Set up Ollama** (if not already done):
   ```bash
   # Start Ollama server (in a separate terminal)
   ollama serve
   
   # Pull the LLM model
   ollama pull llama3.2
   ```

5. **Configure environment variables** (optional):
   ```bash
   # Create .env file or set environment variables
   export LLM_MODEL=llama3.2  # Default
   export OLLAMA_BASE_URL=http://localhost:11434  # Default
   ```

## Usage

1. **Start the FastAPI server**:
   ```bash
   python app.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000/static/index.html
   ```
   Or access the API directly at:
   ```
   http://localhost:8000
   ```

3. **Upload a discharge summary**:
   - Click "Choose File" or drag and drop a PDF, TXT, or DOCX file
   - Wait for the upload confirmation

4. **Ask questions**:
   - Type your question in natural language
   - Examples:
     - "When should I take my next antibiotic?"
     - "What food should I avoid after surgery?"
     - "When is my follow-up appointment?"
     - "What activities should I avoid?"

## API Endpoints

### `POST /upload`
Upload a discharge summary document.

**Request**: Multipart form data with `file` field
**Response**: 
```json
{
  "document_id": "uuid",
  "message": "Document uploaded and processed successfully",
  "document_name": "filename.pdf"
}
```

### `POST /query`
Query the discharge summary.

**Request**:
```json
{
  "query": "When should I take my next antibiotic?",
  "document_id": "optional-uuid"
}
```

**Response**:
```json
{
  "answer": "Based on your discharge summary...",
  "clarification_needed": false,
  "clarification_questions": null,
  "sources": ["Document uuid, Section 1"]
}
```

### `GET /documents`
List all uploaded documents.

### `DELETE /documents/{document_id}`
Delete a document and its associated data.

## How It Works

1. **Document Processing**: 
   - Documents are parsed and text is extracted
   - Text is chunked into smaller segments with overlap

2. **Vector Storage**:
   - Chunks are embedded using sentence transformers
   - Stored in ChromaDB vector database

3. **Query Processing**:
   - Query Manager analyzes if clarification is needed
   - If query is vague, clarification questions are returned

4. **Retrieval**:
   - Query is embedded and matched against document chunks
   - Top-k most relevant chunks are retrieved

5. **Generation**:
   - Retrieved context is passed to LLM with a medical assistant prompt
   - LLM generates a patient-friendly answer

## Customization

### Using a Different LLM Model

Set the `LLM_MODEL` environment variable:
```bash
export LLM_MODEL=llama2  # or any other Ollama model
```

### Using OpenAI Instead of Ollama

Modify `rag_system.py` to use OpenAI API:
```python
import openai

# Replace Ollama call with OpenAI
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
```

### Adjusting Chunk Size

In `rag_system.py`, modify the `_chunk_text` method:
```python
def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
    # Adjust chunk_size and overlap as needed
```

## Security & Privacy

- **Local Deployment**: All processing happens locally
- **No External API Calls**: When using Ollama, no data leaves your machine
- **Data Storage**: Documents and vectors stored locally in `uploads/` and `chroma_db/`
- **No Logging**: No query or document content is logged externally

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Pull the model if missing: `ollama pull llama3.2`

### Document Processing Errors
- Ensure file is not corrupted
- Check file format is supported (PDF, TXT, DOCX)
- Verify file is not password-protected (PDFs)

### Empty Responses
- Check if document was processed correctly
- Verify document contains readable text
- Try rephrasing your query

## License

This project is provided as-is for educational and healthcare assistance purposes.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Disclaimer

This tool is designed to assist patients in understanding their discharge instructions. It should not replace professional medical advice. Patients should always consult with their healthcare providers for medical decisions.

