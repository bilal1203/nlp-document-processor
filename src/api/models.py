"""
API data models using Pydantic.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator


class SummaryLength(str, Enum):
    """Summary length options."""
    SHORT = "short"
    MEDIUM = "medium"
    DETAILED = "detailed"


class SummaryMethod(str, Enum):
    """Summary method options."""
    ABSTRACTIVE = "abstractive"
    EXTRACTIVE = "extractive"
    HYBRID = "hybrid"


class DocumentType(str, Enum):
    """Document type options."""
    CONTRACT = "Contract"
    INVOICE = "Invoice"
    REPORT = "Report"
    EMAIL = "Email"
    MEMO = "Memo"
    LETTER = "Letter"
    RESUME = "Resume"
    FINANCIAL = "Financial Statement"
    LEGAL = "Legal Brief"
    PRESS = "Press Release"
    MINUTES = "Meeting Minutes"
    PROPOSAL = "Proposal"
    OTHER = "Other"


class PriorityLevel(str, Enum):
    """Priority level options."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


class ProcessingOptions(BaseModel):
    """Options for document processing."""
    perform_ner: bool = Field(True, description="Whether to perform named entity recognition")
    perform_classification: bool = Field(True, description="Whether to perform document classification")
    perform_summarization: bool = Field(True, description="Whether to perform summarization")
    summary_length: SummaryLength = Field(SummaryLength.MEDIUM, description="Length of summary")
    summary_method: SummaryMethod = Field(SummaryMethod.ABSTRACTIVE, description="Summarization method")
    clean_text: bool = Field(True, description="Whether to clean the extracted text")
    preserve_structure: bool = Field(True, description="Whether to preserve document structure")


class Entity(BaseModel):
    """Named entity model."""
    text: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type")
    confidence: float = Field(..., description="Confidence score")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")


class ClassificationResult(BaseModel):
    """Document classification result."""
    document_type: DocumentType = Field(..., description="Document type")
    document_type_confidence: float = Field(..., description="Confidence score for document type")
    priority: PriorityLevel = Field(..., description="Priority level")
    priority_confidence: float = Field(..., description="Confidence score for priority")
    document_type_all: Optional[List[Dict[str, Any]]] = Field(None, description="All document type results")
    priority_all: Optional[List[Dict[str, Any]]] = Field(None, description="All priority results")


class SummaryMetrics(BaseModel):
    """Summary evaluation metrics."""
    compression_ratio: float = Field(..., description="Compression ratio")
    original_length: int = Field(..., description="Original text length in words")
    summary_length: int = Field(..., description="Summary length in words")


class SummaryResult(BaseModel):
    """Document summary result."""
    summary: str = Field(..., description="Generated summary")
    method: SummaryMethod = Field(..., description="Summarization method used")
    length: SummaryLength = Field(..., description="Summary length setting")
    metrics: SummaryMetrics = Field(..., description="Summary metrics")


class DocumentMetadata(BaseModel):
    """Document metadata."""
    filename: Optional[str] = Field(None, description="Original filename")
    filetype: str = Field(..., description="File type (PDF, DOCX, TXT)")
    pages: Optional[int] = Field(None, description="Number of pages")
    author: Optional[str] = Field(None, description="Document author")
    created: Optional[str] = Field(None, description="Creation date")
    modified: Optional[str] = Field(None, description="Last modified date")
    
    # Additional metadata fields
    title: Optional[str] = Field(None, description="Document title")
    subject: Optional[str] = Field(None, description="Document subject")
    keywords: Optional[str] = Field(None, description="Document keywords")
    
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


class DocumentFeatures(BaseModel):
    """Document features extracted for analysis."""
    character_count: int = Field(..., description="Character count")
    word_count: int = Field(..., description="Word count")
    top_terms: List[str] = Field(..., description="Top terms in document")
    has_bullet_points: Optional[bool] = Field(None, description="Document has bullet points")
    has_numbered_lists: Optional[bool] = Field(None, description="Document has numbered lists")
    has_headings: Optional[bool] = Field(None, description="Document has headings")


class ProcessingResult(BaseModel):
    """Complete document processing result."""
    success: bool = Field(..., description="Whether processing was successful")
    text_length: int = Field(..., description="Length of processed text")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    entities: Optional[List[Entity]] = Field(None, description="Extracted entities")
    entity_types: Optional[Dict[str, List[str]]] = Field(None, description="Entities grouped by type")
    classification: Optional[ClassificationResult] = Field(None, description="Classification results")
    features: Optional[DocumentFeatures] = Field(None, description="Extracted document features")
    summary: Optional[SummaryResult] = Field(None, description="Summary results")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class ErrorResponse(BaseModel):
    """API error response."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")