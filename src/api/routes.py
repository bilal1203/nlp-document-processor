"""
API routes for the document processing system.
"""
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import parse_obj_as
import json

from src.api.models import (
    ProcessingOptions, 
    ProcessingResult, 
    SummaryLength, 
    SummaryMethod,
    ErrorResponse
)
from src.api.utils import process_uploaded_file, is_allowed_file

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

@router.post(
    "/process",
    response_model=ProcessingResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Process a document",
    description="Upload and process a document for NLP analysis"
)
async def process_document(
    file: UploadFile = File(..., description="Document file to process"),
    options: Optional[str] = Form(None, description="JSON string with processing options")
) -> ProcessingResult:
    """
    Process an uploaded document file.
    
    Args:
        file: Document file to process
        options: JSON string with processing options
        
    Returns:
        Document processing result
    """
    # Validate file type
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: PDF, DOCX, TXT"
        )
    
    # Parse options if provided
    processing_options = ProcessingOptions()
    if options:
        try:
            options_dict = json.loads(options)
            processing_options = ProcessingOptions(**options_dict)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in options parameter"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid options: {str(e)}"
            )
    
    # Process the document
    result = await process_uploaded_file(file, processing_options)
    return result


@router.post(
    "/extract-entities",
    summary="Extract named entities from a document",
    description="Upload a document and extract named entities"
)
async def extract_entities(
    file: UploadFile = File(..., description="Document file to process")
) -> Dict[str, Any]:
    """
    Extract named entities from an uploaded document.
    
    Args:
        file: Document file to process
        
    Returns:
        Dictionary with extracted entities
    """
    # Create options with only NER enabled
    options = ProcessingOptions(
        perform_ner=True,
        perform_classification=False,
        perform_summarization=False
    )
    
    # Process the document
    result = await process_uploaded_file(file, options)
    
    # Return only the entities part
    return {
        "success": result.success,
        "entities": result.entities,
        "entity_types": result.entity_types,
        "metadata": result.metadata
    }


@router.post(
    "/classify-document",
    summary="Classify a document by type and priority",
    description="Upload a document and classify it"
)
async def classify_document(
    file: UploadFile = File(..., description="Document file to process")
) -> Dict[str, Any]:
    """
    Classify an uploaded document by type and priority.
    
    Args:
        file: Document file to process
        
    Returns:
        Dictionary with classification results
    """
    # Create options with only classification enabled
    options = ProcessingOptions(
        perform_ner=False,
        perform_classification=True,
        perform_summarization=False
    )
    
    # Process the document
    result = await process_uploaded_file(file, options)
    
    # Return only the classification part
    return {
        "success": result.success,
        "classification": result.classification,
        "features": result.features,
        "metadata": result.metadata
    }


@router.post(
    "/summarize",
    summary="Generate a summary of a document",
    description="Upload a document and generate a summary"
)
async def summarize_document(
    file: UploadFile = File(..., description="Document file to process"),
    length: SummaryLength = Query(SummaryLength.MEDIUM, description="Summary length"),
    method: SummaryMethod = Query(SummaryMethod.ABSTRACTIVE, description="Summarization method")
) -> Dict[str, Any]:
    """
    Generate a summary of an uploaded document.
    
    Args:
        file: Document file to process
        length: Summary length ('short', 'medium', 'detailed')
        method: Summarization method ('abstractive', 'extractive', 'hybrid')
        
    Returns:
        Dictionary with summary results
    """
    # Create options with only summarization enabled
    options = ProcessingOptions(
        perform_ner=False,
        perform_classification=False,
        perform_summarization=True,
        summary_length=length,
        summary_method=method
    )
    
    # Process the document
    result = await process_uploaded_file(file, options)
    
    # Return only the summary part
    return {
        "success": result.success,
        "summary": result.summary,
        "metadata": result.metadata
    }


@router.get(
    "/health",
    summary="Check API health",
    description="Check if the API is operational"
)
async def health_check() -> Dict[str, str]:
    """
    Check if the API is operational.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}


@router.get(
    "/info",
    summary="Get API information",
    description="Get information about the API"
)
async def get_info() -> Dict[str, Any]:
    """
    Get information about the API.
    
    Returns:
        API information
    """
    return {
        "name": "NLP Document Processing API",
        "version": "1.0.0",
        "description": "API for processing documents with NLP",
        "endpoints": [
            {"path": "/process", "method": "POST", "description": "Process a document"},
            {"path": "/extract-entities", "method": "POST", "description": "Extract named entities"},
            {"path": "/classify-document", "method": "POST", "description": "Classify a document"},
            {"path": "/summarize", "method": "POST", "description": "Generate a document summary"},
            {"path": "/health", "method": "GET", "description": "Check API health"},
            {"path": "/info", "method": "GET", "description": "Get API information"}
        ],
        "supported_file_types": ["PDF", "DOCX", "TXT"]
    }