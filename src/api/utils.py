"""
API utility functions.
"""
import os
import io
import logging
import tempfile
from typing import Dict, Any, List, Union, Optional, BinaryIO, Tuple

from fastapi import UploadFile, HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from src.preprocessing import DocumentProcessor
from src.nlp import NLPProcessor
from src.api.models import ProcessingOptions, ProcessingResult, DocumentMetadata

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def process_uploaded_file(
    file: UploadFile, 
    options: ProcessingOptions
) -> ProcessingResult:
    """
    Process an uploaded file with the document and NLP processors.
    
    Args:
        file: Uploaded file
        options: Processing options
        
    Returns:
        Processing result
    """
    try:
        # Validate file
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024)} MB"
            )
        
        # Determine file type
        file_type = None
        if file_ext == '.pdf':
            file_type = 'pdf'
        elif file_ext in ['.docx', '.doc']:
            file_type = 'docx'
        elif file_ext == '.txt':
            file_type = 'txt'
            
        # Process the document
        result = await process_document_content(
            file_content=content,
            filename=file.filename,
            file_type=file_type,
            options=options
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Reset file position for potential future operations
        await file.seek(0)

async def process_document_content(
    file_content: bytes,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    options: Optional[ProcessingOptions] = None
) -> ProcessingResult:
    """
    Process document content with the document and NLP processors.
    
    Args:
        file_content: Document content as bytes
        filename: Original filename
        file_type: Document type (pdf, docx, txt)
        options: Processing options
        
    Returns:
        Processing result
    """
    if options is None:
        options = ProcessingOptions()
    
    try:
        # Create file-like object from bytes
        file_obj = io.BytesIO(file_content)
        
        # Initialize processors
        doc_processor = DocumentProcessor(preserve_structure=options.preserve_structure)
        nlp_processor = NLPProcessor()
        
        # Process document to extract text
        doc_result = doc_processor.process_document(
            file_path=file_obj,
            file_type=file_type,
            clean_text=options.clean_text
        )
        
        if not doc_result['success']:
            return ProcessingResult(
                success=False,
                text_length=0,
                error=doc_result.get('error', 'Unknown error processing document')
            )
        
        # Extract text from document result
        text = doc_result['text']
        
        # Process text with NLP
        nlp_result = nlp_processor.process_document(
            text=text,
            perform_ner=options.perform_ner,
            perform_classification=options.perform_classification,
            perform_summarization=options.perform_summarization,
            summary_length=options.summary_length,
            summary_method=options.summary_method
        )
        
        # Create metadata
        metadata = _create_metadata(doc_result, filename, file_type)
        
        # Combine results
        result = ProcessingResult(
            success=True,
            text_length=nlp_result['text_length'],
            metadata=metadata,
            entities=nlp_result.get('entities'),
            entity_types=nlp_result.get('entity_types'),
            classification=nlp_result.get('classification'),
            features=nlp_result.get('features'),
            summary=nlp_result.get('summary')
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing document content: {str(e)}", exc_info=True)
        return ProcessingResult(
            success=False,
            text_length=0,
            error=f"Error processing document: {str(e)}"
        )

def _create_metadata(
    doc_result: Dict[str, Any],
    filename: Optional[str],
    file_type: Optional[str]
) -> DocumentMetadata:
    """
    Create document metadata from processing results.
    
    Args:
        doc_result: Document processing result
        filename: Original filename
        file_type: Document type
        
    Returns:
        Document metadata
    """
    # Extract metadata from document result
    metadata_dict = doc_result.get('metadata', {})
    
    # Create metadata model
    metadata = DocumentMetadata(
        filename=filename,
        filetype=file_type or 'unknown',
        pages=metadata_dict.get('pages'),
        author=metadata_dict.get('author'),
        created=metadata_dict.get('created'),
        modified=metadata_dict.get('modified'),
        title=metadata_dict.get('title'),
        subject=metadata_dict.get('subject'),
        keywords=metadata_dict.get('keywords')
    )
    
    return metadata

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        File extension with dot (e.g., '.pdf')
    """
    return os.path.splitext(filename)[1].lower()

def is_allowed_file(filename: str) -> bool:
    """
    Check if file has an allowed extension.
    
    Args:
        filename: Filename
        
    Returns:
        Whether file is allowed
    """
    return get_file_extension(filename) in ALLOWED_EXTENSIONS