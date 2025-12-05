"""
RAG System
Retrieval Augmented Generation for discharge instructions
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from pathlib import Path
import requests

class RAGSystem:
    """RAG system for retrieving and generating answers from discharge summaries"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.db_path = Path("chroma_db")
        self.db_path.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection for storing document chunks
        self.collection = self.client.get_or_create_collection(
            name="discharge_instructions",
            metadata={"hnsw:space": "cosine"}
        )
        
        # LLM configuration (using Ollama for local deployment)
        self.llm_model = os.getenv("LLM_MODEL", "llama3.2")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Check if Ollama is available
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            import requests
            # Try to list models via API
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                if self.llm_model not in models:
                    print(f"Warning: Model '{self.llm_model}' not found in Ollama.")
                    print(f"Available models: {', '.join(models) if models else 'none'}")
                    print(f"Please run: ollama pull {self.llm_model}")
                    print("Falling back to a simpler response generation.")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
            print("Or set OLLAMA_BASE_URL environment variable")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def add_document(self, document_id: str, text: str):
        """
        Add a document to the vector store
        
        Args:
            document_id: Unique identifier for the document
            text: Text content of the document
        """
        # Remove existing chunks for this document
        self.remove_document(document_id)
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        if not chunks:
            raise ValueError("No text chunks generated from document")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create IDs for chunks
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{
                "document_id": document_id,
                "chunk_index": i,
                "source": f"Document {document_id}, Section {i+1}"
            } for i in range(len(chunks))]
        )
    
    def remove_document(self, document_id: str):
        """
        Remove a document and its chunks from the vector store
        
        Args:
            document_id: Unique identifier for the document
        """
        try:
            # Get all chunks for this document
            # ChromaDB uses metadata filtering
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            # If document doesn't exist or error occurs, continue silently
            print(f"Note: Could not remove document {document_id}: {e}")
    
    def retrieve_context(self, document_id: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context for a query
        
        Args:
            document_id: Document to search in
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of relevant context chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": document_id}
            )
        except Exception as e:
            # Fallback: query without filter if metadata filter fails
            print(f"Warning: Metadata filter failed, trying without filter: {e}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2  # Get more results to filter manually
            )
            
            # Filter results by document_id manually
            if results.get('metadatas') and len(results['metadatas']) > 0:
                filtered_indices = [
                    i for i, meta in enumerate(results['metadatas'][0])
                    if meta.get('document_id') == document_id
                ][:top_k]
                
                # Reconstruct filtered results
                if filtered_indices:
                    results = {
                        'documents': [[results['documents'][0][i] for i in filtered_indices]],
                        'metadatas': [[results['metadatas'][0][i] for i in filtered_indices]],
                        'distances': [[results['distances'][0][i] for i in filtered_indices]] if 'distances' in results else []
                    }
                else:
                    results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Format results
        contexts = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                contexts.append({
                    "text": doc,
                    "source": results['metadatas'][0][i].get("source", ""),
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return contexts
    
    def generate_answer(self, query: str, contexts: List[Dict]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        if not contexts:
            return "I couldn't find relevant information in the discharge summary to answer your question. Please try rephrasing your query or contact your healthcare provider."
        
        # Prepare context
        context_text = "\n\n".join([
            f"[Source: {ctx['source']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Create prompt for medical assistant
        prompt = f"""You are a specialized medical assistant helping patients understand their medical documents, including discharge instructions and test reports.

CONTEXT FROM MEDICAL DOCUMENT:
{context_text}

PATIENT QUESTION: {query}

INSTRUCTIONS:
1. Answer ONLY based on the provided medical document context above
2. Use simple, clear language that patients without medical training can understand
3. For different question types, provide:
   - Medications: Include name, dosage, frequency, timing, duration, and special instructions
   - Diet: List specific foods to eat/avoid, meal timing, and duration of restrictions
   - Activities: Specify what's allowed/prohibited, intensity levels, and duration
   - Follow-up care: Provide appointment details, dates, times, and contact information
   - Symptoms/Warnings: Clearly state what to watch for, when to seek help, and emergency signs
   - Recovery timeline: Provide expected recovery duration and milestones
   - Lab Results: Explain test names, values, units, reference ranges, and what normal/abnormal means in simple terms. Highlight any abnormal results clearly.
   - Imaging Results: Explain findings in simple language, what they mean, and any recommendations. Use patient-friendly terms instead of medical jargon.
   - Test Reports: Summarize key findings, explain what tests were done, and what the results mean for the patient
4. Format your response with:
   - Clear headings or bullet points for readability
   - Specific numbers, dates, dosages, frequencies, and test values
   - Action items in a numbered or bulleted list
   - Visual indicators (✅ for normal, ⚠️ for abnormal/attention needed)
5. For test reports specifically:
   - Explain what each test measures in simple terms
   - Compare values to reference ranges and explain if they're normal, high, or low
   - Explain what abnormal results might mean (but don't diagnose)
   - Recommend when to follow up with healthcare provider
6. If information is not in the context, clearly state: "This information is not mentioned in your document. Please contact your healthcare provider for clarification."
7. Be professional, warm, and reassuring
8. End by encouraging the patient to contact their healthcare provider if they have additional questions or concerns

ANSWER:"""
        
        try:
            import requests
            # Use Ollama API for local LLM
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more factual responses
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('response', '').strip()
                
                if not answer:
                    # Fallback if LLM doesn't respond
                    answer = self._generate_fallback_answer(query, contexts)
                
                return answer
            else:
                raise Exception(f"Ollama API returned status {response.status_code}")
        
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
            # Fallback to simple extraction
            return self._generate_fallback_answer(query, contexts)
    
    def _generate_fallback_answer(self, query: str, contexts: List[Dict]) -> str:
        """
        Generate a simple answer when LLM is not available
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            
        Returns:
            Simple extracted answer
        """
        if not contexts:
            return "I couldn't find relevant information to answer your question."
        
        # Simple extraction: return the most relevant context
        most_relevant = contexts[0]['text']
        
        # Try to extract a relevant sentence or paragraph
        query_lower = query.lower()
        sentences = most_relevant.split('.')
        
        relevant_sentences = [
            s.strip() for s in sentences
            if any(word in s.lower() for word in query_lower.split())
        ]
        
        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:3])
            if answer:
                return f"Based on your discharge summary: {answer}"
        
        return f"Here's the relevant information from your discharge summary:\n\n{most_relevant[:500]}"

