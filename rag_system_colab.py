"""
RAG System for Google Colab
Uses Hugging Face Transformers instead of Ollama for LLM inference
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings("ignore")

class RAGSystemColab:
    """RAG system for Colab using Hugging Face Transformers"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt2",
                 use_gpu: bool = True,
                 use_8bit: bool = False,
                 db_path: str = None):
        """
        Initialize RAG system for Colab
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
            llm_model: Hugging Face model name for LLM
            use_gpu: Whether to use GPU if available
            use_8bit: Whether to use 8-bit quantization (saves memory)
            db_path: Path to ChromaDB storage (use Drive path for persistence)
        """
        # Set device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        if self.device == "cuda":
            self.embedding_model = self.embedding_model.to(self.device)
        
        # Initialize ChromaDB
        if db_path is None:
            # Default to current directory, but recommend using Drive path
            db_path = Path("chroma_db")
        else:
            db_path = Path(db_path)
        
        db_path.mkdir(exist_ok=True, parents=True)
        self.db_path = db_path
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection for storing document chunks
        self.collection = self.client.get_or_create_collection(
            name="discharge_instructions",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM
        print(f"Loading LLM model: {llm_model}...")
        self.llm_model_name = llm_model
        self._load_llm(llm_model, use_8bit)
        
        print("✅ RAG System initialized successfully!")
    
    def _load_llm(self, model_name: str, use_8bit: bool = False):
        """Load Hugging Face LLM model"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optional 8-bit quantization
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if use_8bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.llm_model = self.llm_model.to(self.device)
            
            # Create pipeline for easier text generation
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print(f"✅ LLM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Error loading LLM model: {e}")
            print("Falling back to simple text extraction")
            self.llm_pipeline = None
    
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
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        ).tolist()
        
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
        print(f"✅ Document {document_id} added with {len(chunks)} chunks")
    
    def remove_document(self, document_id: str):
        """
        Remove a document and its chunks from the vector store
        
        Args:
            document_id: Unique identifier for the document
        """
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
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
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).tolist()[0]
        
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
                n_results=top_k * 2
            )
            
            # Filter results by document_id manually
            if results.get('metadatas') and len(results['metadatas']) > 0:
                filtered_indices = [
                    i for i, meta in enumerate(results['metadatas'][0])
                    if meta.get('document_id') == document_id
                ][:top_k]
                
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
5. If information is not in the context, clearly state: "This information is not mentioned in your document. Please contact your healthcare provider for clarification."
6. Be professional, warm, and reassuring
7. End by encouraging the patient to contact their healthcare provider if they have additional questions or concerns

ANSWER:"""
        
        try:
            if self.llm_pipeline is None:
                # Fallback to simple extraction
                return self._generate_fallback_answer(query, contexts)
            
            # Generate answer using Hugging Face pipeline
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part (after "ANSWER:")
            if "ANSWER:" in generated_text:
                answer = generated_text.split("ANSWER:")[-1].strip()
            else:
                # If model didn't follow format, use the continuation
                answer = generated_text[len(prompt):].strip()
            
            # Clean up answer
            answer = answer.split("\n\n")[0]  # Take first paragraph
            answer = answer.strip()
            
            if not answer:
                return self._generate_fallback_answer(query, contexts)
            
            return answer
        
        except Exception as e:
            print(f"Error generating answer with LLM: {e}")
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

