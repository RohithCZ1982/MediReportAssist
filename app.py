"""
Patient Discharge Instructions Assistant
Main FastAPI application with RAG and Query Manager
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid
from pathlib import Path

from rag_system import RAGSystem
from query_manager import QueryManager
from document_processor import DocumentProcessor

app = FastAPI(title="Patient Discharge Instructions Assistant")

# Serve static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_system = RAGSystem()
query_manager = QueryManager()
doc_processor = DocumentProcessor()

# Store for active sessions (document IDs)
active_documents = {}

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    clarification_needed: bool = False
    clarification_questions: Optional[List[str]] = None
    sources: Optional[List[str]] = None

class UploadResponse(BaseModel):
    document_id: str
    message: str
    document_name: str

@app.get("/")
async def root():
    """Serve the main HTML page"""
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Patient Discharge Instructions Assistant API", "docs": "/docs"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a discharge summary document
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.docx'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{document_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        text_content = doc_processor.process_document(str(file_path))
        
        if not text_content or len(text_content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Document appears to be empty or could not be processed"
            )
        
        # Add to RAG system
        rag_system.add_document(document_id, text_content)
        active_documents[document_id] = {
            "filename": file.filename,
            "path": str(file_path)
        }
        
        return UploadResponse(
            document_id=document_id,
            message="Document uploaded and processed successfully",
            document_name=file.filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the discharge summary using natural language
    """
    try:
        if not request.document_id:
            # Try to use the most recent document if available
            if not active_documents:
                raise HTTPException(
                    status_code=400,
                    detail="No document uploaded. Please upload a document first."
                )
            request.document_id = list(active_documents.keys())[-1]
        
        if request.document_id not in active_documents:
            raise HTTPException(
                status_code=404,
                detail="Document not found. Please upload a document first."
            )
        
        # Check if query needs clarification
        clarification_result = query_manager.analyze_query(request.query)
        
        if clarification_result["needs_clarification"]:
            return QueryResponse(
                answer="",
                clarification_needed=True,
                clarification_questions=clarification_result["questions"],
                sources=None
            )
        
        # Retrieve relevant context using RAG
        relevant_context = rag_system.retrieve_context(
            request.document_id,
            request.query,
            top_k=3
        )
        
        # Generate answer using LLM
        answer = rag_system.generate_answer(
            request.query,
            relevant_context
        )
        
        # Extract source references
        sources = [ctx.get("source", "") for ctx in relevant_context[:2]]
        
        return QueryResponse(
            answer=answer,
            clarification_needed=False,
            clarification_questions=None,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    """
    return {
        "documents": [
            {
                "document_id": doc_id,
                "filename": info["filename"]
            }
            for doc_id, info in active_documents.items()
        ]
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated data
    """
    if document_id not in active_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from RAG system
    rag_system.remove_document(document_id)
    
    # Delete file
    file_path = active_documents[document_id]["path"]
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove from active documents
    del active_documents[document_id]
    
    return {"message": "Document deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

