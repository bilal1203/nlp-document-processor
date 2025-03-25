"""
Tests for API module.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import ProcessingResult, ProcessingOptions


client = TestClient(app)


class TestAPIRoutes:
    """Test cases for API routes."""
    
    def test_health(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_info(self):
        """Test info endpoint."""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "supported_file_types" in data
        assert len(data["endpoints"]) > 0
    
    @patch('src.api.routes.process_uploaded_file')
    async def test_process_document(self, mock_process_uploaded_file):
        """Test document processing endpoint."""
        # Configure mock
        mock_process_uploaded_file.return_value = ProcessingResult(
            success=True,
            text_length=100,
            entities=[{"text": "Test", "type": "ORG", "confidence": 0.9, "start": 0, "end": 4}]
        )
        
        # Create a simple test file
        test_file = io.BytesIO(b"Test document content")
        test_file.name = "test.txt"
        
        # Make request
        response = client.post(
            "/api/process",
            files={"file": ("test.txt", test_file, "text/plain")},
            data={"options": json.dumps({"perform_ner": True, "perform_classification": True})}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text_length"] == 100
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "Test"
        
        # Verify processing was called with correct options
        call_args = mock_process_uploaded_file.call_args
        assert call_args is not None
        file_arg = call_args[0][0]
        options_arg = call_args[0][1]
        
        assert file_arg.filename == "test.txt"
        assert options_arg.perform_ner is True
        assert options_arg.perform_classification is True
    
    @patch('src.api.routes.process_uploaded_file')
    async def test_extract_entities(self, mock_process_uploaded_file):
        """Test entity extraction endpoint."""
        # Configure mock
        mock_process_uploaded_file.return_value = ProcessingResult(
            success=True,
            text_length=100,
            entities=[{"text": "Test", "type": "ORG", "confidence": 0.9, "start": 0, "end": 4}],
            entity_types={"ORG": ["Test"]}
        )
        
        # Create a simple test file
        test_file = io.BytesIO(b"Test document content")
        test_file.name = "test.txt"
        
        # Make request
        response = client.post(
            "/api/extract-entities",
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "entities" in data
        assert "entity_types" in data
        assert data["entity_types"]["ORG"] == ["Test"]
        
        # Verify options
        call_args = mock_process_uploaded_file.call_args
        options_arg = call_args[0][1]
        
        assert options_arg.perform_ner is True
        assert options_arg.perform_classification is False
        assert options_arg.perform_summarization is False
    
    @patch('src.api.routes.process_uploaded_file')
    async def test_classify_document(self, mock_process_uploaded_file):
        """Test document classification endpoint."""
        # Configure mock
        mock_process_uploaded_file.return_value = ProcessingResult(
            success=True,
            text_length=100,
            classification={
                "document_type": "Report",
                "document_type_confidence": 0.85,
                "priority": "Medium",
                "priority_confidence": 0.75
            },
            features={"word_count": 100}
        )
        
        # Create a simple test file
        test_file = io.BytesIO(b"This is a report document")
        test_file.name = "test.txt"
        
        # Make request
        response = client.post(
            "/api/classify-document",
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "classification" in data
        assert data["classification"]["document_type"] == "Report"
        assert "features" in data
        
        # Verify options
        call_args = mock_process_uploaded_file.call_args
        options_arg = call_args[0][1]
        
        assert options_arg.perform_ner is False
        assert options_arg.perform_classification is True
        assert options_arg.perform_summarization is False
    
    @patch('src.api.routes.process_uploaded_file')
    async def test_summarize_document(self, mock_process_uploaded_file):
        """Test document summarization endpoint."""
        # Configure mock
        mock_process_uploaded_file.return_value = ProcessingResult(
            success=True,
            text_length=100,
            summary={
                "summary": "This is a summary",
                "method": "abstractive",
                "length": "short",
                "metrics": {
                    "compression_ratio": 0.2,
                    "original_length": 100,
                    "summary_length": 20
                }
            }
        )
        
        # Create a simple test file
        test_file = io.BytesIO(b"This is a document that needs to be summarized")
        test_file.name = "test.txt"
        
        # Make request
        response = client.post(
            "/api/summarize",
            files={"file": ("test.txt", test_file, "text/plain")},
            params={"length": "short", "method": "abstractive"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data
        assert data["summary"]["summary"] == "This is a summary"
        assert data["summary"]["method"] == "abstractive"
        
        # Verify options
        call_args = mock_process_uploaded_file.call_args
        options_arg = call_args[0][1]
        
        assert options_arg.perform_ner is False
        assert options_arg.perform_classification is False
        assert options_arg.perform_summarization is True
        assert options_arg.summary_length == "short"
        assert options_arg.summary_method == "abstractive"


class TestAPIUtils:
    """Test cases for API utility functions."""
    
    @patch('src.api.utils.DocumentProcessor')
    @patch('src.api.utils.NLPProcessor')
    async def test_process_document_content(self, mock_nlp_processor, mock_doc_processor):
        """Test document content processing."""
        from src.api.utils import process_document_content
        
        # Configure mocks
        mock_doc_processor_instance = MagicMock()
        mock_doc_processor_instance.process_document.return_value = {
            'success': True,
            'text': 'Test content',
            'metadata': {'pages': 1}
        }
        mock_doc_processor.return_value = mock_doc_processor_instance
        
        mock_nlp_processor_instance = MagicMock()
        mock_nlp_processor_instance.process_document.return_value = {
            'success': True,
            'text_length': 12,
            'entities': [{"text": "Test", "type": "ORG"}]
        }
        mock_nlp_processor.return_value = mock_nlp_processor_instance
        
        # Test processing
        options = ProcessingOptions()
        result = await process_document_content(
            file_content=b"Test content",
            filename="test.txt",
            file_type="txt",
            options=options
        )
        
        # Verify result
        assert result.success is True
        assert result.text_length == 12
        assert result.metadata is not None
        assert result.metadata.filetype == "txt"
        assert result.metadata.pages == 1
        assert result.entities == [{"text": "Test", "type": "ORG"}]
        
        # Test error handling
        mock_doc_processor_instance.process_document.return_value = {
            'success': False,
            'error': 'Failed to process document'
        }
        
        result = await process_document_content(
            file_content=b"Test content",
            filename="test.txt",
            file_type="txt",
            options=options
        )
        
        assert result.success is False
        assert "Failed to process document" in result.error