"""
PDF Parser module for extracting text from PDF documents.
"""
import io
import os
import logging
from typing import Dict, Any, List, Optional, Union

import PyPDF2
import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PDFParser:
    """
    Parser for extracting text and metadata from PDF files.
    Uses PyPDF2 for metadata and pdfplumber for text extraction.
    """

    def __init__(self, preserve_structure: bool = True):
        """
        Initialize the PDF parser.
        
        Args:
            preserve_structure (bool): Whether to preserve document structure 
                                      (paragraphs, sections, etc.)
        """
        self.preserve_structure = preserve_structure
        logger.info("Initialized PDF parser with preserve_structure=%s", preserve_structure)

    def parse(self, file_path: Union[str, io.BytesIO]) -> Dict[str, Any]:
        """
        Parse a PDF file and extract text and metadata.
        
        Args:
            file_path: Path to the PDF file or BytesIO object
            
        Returns:
            Dict containing extracted text and metadata
        """
        logger.info("Parsing PDF: %s", file_path if isinstance(file_path, str) else "BytesIO")
        
        # Check if file_path is a string (path) or BytesIO
        if isinstance(file_path, str):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            metadata = self._extract_metadata(file_path)
            text_content = self._extract_text(file_path)
            
            return {
                "metadata": metadata,
                "text": text_content,
                "pages": len(text_content),
                "success": True
            }
            
        except Exception as e:
            logger.error("Error parsing PDF: %s", str(e), exc_info=True)
            return {
                "metadata": {},
                "text": [],
                "pages": 0,
                "success": False,
                "error": str(e)
            }

    def _extract_metadata(self, file_path: Union[str, io.BytesIO]) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file using PyPDF2.
        
        Args:
            file_path: Path to the PDF file or BytesIO object
            
        Returns:
            Dict containing PDF metadata
        """
        try:
            if isinstance(file_path, str):
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    info = reader.metadata
            else:
                # Reset BytesIO position
                file_path.seek(0)
                reader = PyPDF2.PdfReader(file_path)
                info = reader.metadata
                
            # Convert to standard dictionary
            metadata = {}
            if info:
                for key in info:
                    metadata[key] = info[key]
                    
            # Add additional metadata
            metadata["pages"] = len(reader.pages)
            
            return metadata
            
        except Exception as e:
            logger.warning("Error extracting PDF metadata: %s", str(e))
            return {"error": str(e)}

    def _extract_text(self, file_path: Union[str, io.BytesIO]) -> List[str]:
        """
        Extract text from a PDF file using pdfplumber with structure preservation.
        
        Args:
            file_path: Path to the PDF file or BytesIO object
            
        Returns:
            List of strings, one per page
        """
        text_content = []
        
        try:
            # Open the PDF with pdfplumber
            if isinstance(file_path, str):
                pdf = pdfplumber.open(file_path)
            else:
                # Reset BytesIO position
                file_path.seek(0)
                pdf = pdfplumber.open(file_path)
            
            # Extract text from each page
            for i, page in enumerate(tqdm(pdf.pages, desc="Extracting PDF pages")):
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                
                if self.preserve_structure:
                    # Simple structure preservation - keep paragraphs
                    # For more advanced structure, consider using page.extract_words() 
                    # with clustering by position
                    text_content.append(page_text)
                else:
                    # Just extract raw text without structure
                    text_content.append(page_text.replace('\n', ' '))
                    
            pdf.close()
            return text_content
            
        except Exception as e:
            logger.error("Error extracting text from PDF: %s", str(e))
            return [f"Error extracting text: {str(e)}"]

    @staticmethod
    def is_valid_pdf(file_path: Union[str, io.BytesIO]) -> bool:
        """
        Check if a file is a valid PDF.
        
        Args:
            file_path: Path to the file or BytesIO object
            
        Returns:
            Boolean indicating if the file is a valid PDF
        """
        try:
            if isinstance(file_path, str):
                with open(file_path, 'rb') as file:
                    PyPDF2.PdfReader(file)
            else:
                # Reset BytesIO position
                file_path.seek(0)
                PyPDF2.PdfReader(file_path)
                # Reset again after checking
                file_path.seek(0)
            return True
        except Exception:
            return False