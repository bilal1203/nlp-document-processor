"""
TXT Parser module for extracting content from plain text files.
"""
import os
import io
import logging
import chardet
from typing import Dict, Any, List, Union, Optional

logger = logging.getLogger(__name__)

class TXTParser:
    """
    Parser for extracting text and basic metadata from TXT files.
    """

    def __init__(self, preserve_structure: bool = True, encoding: Optional[str] = None):
        """
        Initialize the TXT parser.
        
        Args:
            preserve_structure (bool): Whether to preserve document structure
                                      (paragraphs, line breaks, etc.)
            encoding (str, optional): Text encoding to use. If None, will try to detect.
        """
        self.preserve_structure = preserve_structure
        self.encoding = encoding
        logger.info("Initialized TXT parser with preserve_structure=%s, encoding=%s", 
                   preserve_structure, encoding)

    def parse(self, file_path: Union[str, io.BytesIO]) -> Dict[str, Any]:
        """
        Parse a TXT file and extract content and basic metadata.
        
        Args:
            file_path: Path to the TXT file or BytesIO object
            
        Returns:
            Dict containing extracted text and metadata
        """
        logger.info("Parsing TXT: %s", file_path if isinstance(file_path, str) else "BytesIO")
        
        try:
            # Read the content and detect encoding if needed
            if isinstance(file_path, str):
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"TXT file not found: {file_path}")
                
                with open(file_path, 'rb') as f:
                    raw_content = f.read()
            else:
                file_path.seek(0)
                raw_content = file_path.read()
                file_path.seek(0)  # Reset the position
            
            encoding = self.encoding
            if encoding is None:
                # Try to detect the encoding
                detection = chardet.detect(raw_content)
                encoding = detection['encoding']
                logger.info(f"Detected encoding: {encoding} with confidence {detection['confidence']}")
            
            # Decode the content
            try:
                content = raw_content.decode(encoding)
            except (UnicodeDecodeError, TypeError):
                # Fallback to UTF-8
                logger.warning(f"Failed to decode with {encoding}, falling back to utf-8")
                try:
                    content = raw_content.decode('utf-8')
                except UnicodeDecodeError:
                    # Last resort fallback
                    logger.warning("Failed to decode with utf-8, using latin-1")
                    content = raw_content.decode('latin-1')
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, content)
            
            # Process content
            processed_content = self._process_content(content)
            
            return {
                "metadata": metadata,
                "text": processed_content,
                "success": True
            }
            
        except Exception as e:
            logger.error("Error parsing TXT: %s", str(e), exc_info=True)
            return {
                "metadata": {},
                "text": "",
                "success": False,
                "error": str(e)
            }

    def _extract_metadata(self, file_path: Union[str, io.BytesIO], content: str) -> Dict[str, Any]:
        """
        Extract basic metadata from a TXT file.
        
        Args:
            file_path: Path to the TXT file or BytesIO object
            content: Decoded text content
            
        Returns:
            Dict containing basic metadata
        """
        # For TXT files, we can only provide basic metadata
        lines = content.splitlines()
        paragraphs = self._count_paragraphs(content)
        
        metadata = {
            "line_count": len(lines),
            "paragraph_count": paragraphs,
            "character_count": len(content),
            "word_count": len(content.split())
        }
        
        # Add file metadata if path is provided
        if isinstance(file_path, str):
            file_stat = os.stat(file_path)
            metadata.update({
                "filename": os.path.basename(file_path),
                "file_size": file_stat.st_size,
                "creation_time": file_stat.st_ctime,
                "modification_time": file_stat.st_mtime
            })
        
        return metadata

    def _process_content(self, content: str) -> Union[str, List[str]]:
        """
        Process the text content based on preserve_structure setting.
        
        Args:
            content: Raw text content
            
        Returns:
            Processed text content (string or list of paragraphs)
        """
        if not self.preserve_structure:
            # Join all lines with spaces
            return " ".join([line.strip() for line in content.splitlines()])
        else:
            # Split into paragraphs (defined as blocks separated by empty lines)
            paragraphs = []
            current_paragraph = []
            
            for line in content.splitlines():
                if line.strip():
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
            
            # Add the last paragraph if it exists
            if current_paragraph:
                paragraphs.append("\n".join(current_paragraph))
            
            return paragraphs

    def _count_paragraphs(self, content: str) -> int:
        """
        Count paragraphs in the text content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Number of paragraphs detected
        """
        # Split the content by empty lines
        paragraphs = [p for p in content.replace('\r\n', '\n').split('\n\n') if p.strip()]
        return len(paragraphs)

    @staticmethod
    def is_valid_txt(file_path: Union[str, io.BytesIO]) -> bool:
        """
        Check if a file is a valid text file.
        
        Args:
            file_path: Path to the file or BytesIO object
            
        Returns:
            Boolean indicating if the file appears to be a valid text file
        """
        try:
            if isinstance(file_path, str):
                # Check if file exists and is not empty
                if not os.path.exists(file_path):
                    return False
                
                # Try to read and decode the content
                with open(file_path, 'rb') as file:
                    content = file.read(1024)  # Read first 1KB
            else:
                # Store original position and reset after checking
                original_pos = file_path.tell()
                file_path.seek(0)
                content = file_path.read(1024)
                file_path.seek(original_pos)
            
            # Try to decode with common encodings
            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    content.decode(encoding)
                    return True
                except UnicodeDecodeError:
                    continue
                    
            # Try chardet detection
            detection = chardet.detect(content)
            if detection['confidence'] > 0.5:
                try:
                    content.decode(detection['encoding'])
                    return True
                except (UnicodeDecodeError, TypeError):
                    pass
                    
            return False
        except Exception:
            return False