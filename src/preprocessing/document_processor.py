"""
Document Processor module for handling various document formats.
"""
import os
import io
import logging
import mimetypes
from typing import Dict, Any, List, Union, Optional, BinaryIO

from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.docx_parser import DOCXParser
from src.preprocessing.txt_parser import TXTParser
from src.preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Main document processor class that handles various document formats.
    """
    
    def __init__(self, preserve_structure: bool = True):
        """
        Initialize the document processor.
        
        Args:
            preserve_structure (bool): Whether to preserve document structure
        """
        self.preserve_structure = preserve_structure
        
        # Initialize parsers
        self.pdf_parser = PDFParser(preserve_structure=preserve_structure)
        self.docx_parser = DOCXParser(preserve_structure=preserve_structure)
        self.txt_parser = TXTParser(preserve_structure=preserve_structure)
        
        # Initialize text cleaner
        self.text_cleaner = TextCleaner(
            remove_stopwords=False,
            remove_punctuation=False,
            lowercase=False,
            normalize_whitespace=True
        )
        
        logger.info("Initialized DocumentProcessor with preserve_structure=%s", preserve_structure)
    
    def process_document(self, 
                         file_path: Union[str, BinaryIO], 
                         file_type: Optional[str] = None,
                         clean_text: bool = True) -> Dict[str, Any]:
        """
        Process a document and extract content.
        
        Args:
            file_path: Path to the document or file-like object
            file_type: Document type (pdf, docx, txt) if known
            clean_text: Whether to clean the extracted text
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Determine file type if not specified
            if file_type is None:
                file_type = self._determine_file_type(file_path)
            
            # Parse document based on file type
            if file_type == 'pdf':
                result = self.pdf_parser.parse(file_path)
            elif file_type == 'docx':
                result = self.docx_parser.parse(file_path)
            elif file_type == 'txt':
                result = self.txt_parser.parse(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_type}"
                }
            
            # Clean text if requested
            if clean_text and result['success']:
                result['text'] = self.text_cleaner.clean_text(result['text'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error processing document: {str(e)}"
            }
    
    def _determine_file_type(self, file_path: Union[str, BinaryIO]) -> str:
        """
        Determine the file type based on the file path or content.
        
        Args:
            file_path: Path to the document or file-like object
            
        Returns:
            File type as string (pdf, docx, txt)
        """
        # If file_path is a string, use its extension
        if isinstance(file_path, str):
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return 'pdf'
            elif file_extension in ['.docx', '.doc']:
                return 'docx'
            elif file_extension == '.txt':
                return 'txt'
            else:
                # Try to determine from MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type:
                    if mime_type == 'application/pdf':
                        return 'pdf'
                    elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                      'application/msword']:
                        return 'docx'
                    elif mime_type == 'text/plain':
                        return 'txt'
        
        # If file_path is a file-like object, try to determine from content
        else:
            # Save current position
            current_pos = file_path.tell()
            
            try:
                # Read first few bytes for signature checking
                signature = file_path.read(8)
                file_path.seek(current_pos)  # Reset position
                
                # Check for PDF signature
                if signature.startswith(b'%PDF'):
                    return 'pdf'
                
                # Check for DOCX (ZIP archive) signature
                if signature.startswith(b'PK\x03\x04'):
                    return 'docx'
                
                # For TXT, check if it's readable text
                file_path.seek(current_pos)
                sample = file_path.read(1024)
                file_path.seek(current_pos)  # Reset position
                
                try:
                    # Try to decode as text
                    sample.decode('utf-8')
                    return 'txt'
                except UnicodeDecodeError:
                    pass
                
                # Default to txt if we can't determine
                return 'txt'
                
            except Exception as e:
                logger.warning(f"Error determining file type: {str(e)}")
                # Reset position and default to txt
                file_path.seek(current_pos)
                return 'txt'
        
        # Default to txt if we couldn't determine the type
        return 'txt'
    
    def extract_sentences(self, text: Union[str, List[str]]) -> List[str]:
        """
        Extract sentences from the document text.
        
        Args:
            text: Document text (string or list of strings)
            
        Returns:
            List of sentences
        """
        return self.text_cleaner.extract_sentences(text)
    
    def extract_paragraphs(self, text: Union[str, List[str]]) -> List[str]:
        """
        Extract paragraphs from the document text.
        
        Args:
            text: Document text (string or list of strings)
            
        Returns:
            List of paragraphs
        """
        if isinstance(text, str):
            return self.text_cleaner.extract_paragraphs(text)
        else:
            return [p for p in text if p.strip()]
    
    def extract_keywords(self, text: Union[str, List[str]], top_n: int = 10) -> List[str]:
        """
        Extract keywords from the document text.
        
        Args:
            text: Document text (string or list of strings)
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        return self.text_cleaner.extract_keywords(text, top_n=top_n)