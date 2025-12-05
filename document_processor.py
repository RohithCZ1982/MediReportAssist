"""
Document Processor
Handles parsing and text extraction from various document formats
"""

from pathlib import Path
import PyPDF2
from docx import Document
import re
from test_report_processor import TestReportProcessor

class DocumentProcessor:
    """Processes documents and extracts text content"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx'}
        self.test_report_processor = TestReportProcessor()
    
    def process_document(self, file_path: str, detect_test_reports: bool = True) -> str:
        """
        Process document and extract text content
        
        Args:
            file_path: Path to the document file
            detect_test_reports: If True, enhance test reports with structured data
            
        Returns:
            Extracted text content (enhanced if test report detected)
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            text = self._process_pdf(file_path)
        elif file_ext == '.txt':
            text = self._process_txt(file_path)
        elif file_ext == '.docx':
            text = self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Enhance test reports with structured information
        if detect_test_reports:
            text = self._enhance_test_report(text)
        
        return text
    
    def _enhance_test_report(self, text: str) -> str:
        """
        Enhance test report text with structured data extraction
        
        Args:
            text: Raw extracted text
            
        Returns:
            Enhanced text with structured test report information
        """
        # Detect if this is a test report
        report_type = self.test_report_processor.detect_report_type(text)
        
        if report_type == 'unknown':
            return text
        
        # Process the test report
        processed = self.test_report_processor.process_test_report(text)
        
        # Enhance text with structured summaries
        enhanced_text = text
        
        if processed['lab_summary']:
            enhanced_text = f"{processed['lab_summary']}\n\n--- ORIGINAL REPORT ---\n\n{enhanced_text}"
        
        if processed['imaging_summary']:
            enhanced_text = f"{processed['imaging_summary']}\n\n--- ORIGINAL REPORT ---\n\n{enhanced_text}"
        
        # Add metadata
        if processed['has_abnormal_results']:
            enhanced_text = "⚠️ NOTE: This report contains abnormal results that may require attention.\n\n" + enhanced_text
        
        return enhanced_text
    
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


