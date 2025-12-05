"""
Document Processor
Handles parsing and text extraction from various document formats
"""

from pathlib import Path
import PyPDF2
from docx import Document
import re

class DocumentProcessor:
    """Processes documents and extracts text content"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx'}
    
    def process_document(self, file_path: str) -> str:
        """
        Process document and extract text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path)
        elif file_ext == '.txt':
            return self._process_txt(file_path)
        elif file_ext == '.docx':
            return self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
        
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        
        return '\n\n'.join(text_content)
    
    def _process_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    def _process_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n\n'.join(paragraphs)
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\/]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


