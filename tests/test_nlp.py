"""
Tests for NLP module.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch  # Import torch at the beginning
import numpy as np

from src.nlp.ner import NamedEntityRecognizer
from src.nlp.classifier import DocumentClassifier
from src.nlp.summarizer import DocumentSummarizer
from src.nlp.nlp_processor import NLPProcessor


@pytest.mark.parametrize("device", [None, "cpu"])
class TestNER:
    """Test cases for NamedEntityRecognizer."""
    
    @patch('src.nlp.ner.pipeline')
    @patch('src.nlp.ner.AutoTokenizer')
    @patch('src.nlp.ner.AutoModelForTokenClassification')
    def test_init(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test initialization."""
        # Configure mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()
        
        # Test initialization
        ner = NamedEntityRecognizer(device=device)
        
        # Verify model loaded correctly
        mock_tokenizer.from_pretrained.assert_called_once_with("dslim/bert-base-NER")
        mock_model.from_pretrained.assert_called_once_with("dslim/bert-base-NER")
        mock_pipeline.assert_called_once()
        
        # Check initialization properties
        assert ner.model_name == "dslim/bert-base-NER"
        assert ner.confidence_threshold == 0.8
        assert ner.device == "cpu"  # Always cpu in tests
    
    @patch('src.nlp.ner.pipeline')
    @patch('src.nlp.ner.AutoTokenizer')
    @patch('src.nlp.ner.AutoModelForTokenClassification')
    def test_extract_entities(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test entity extraction."""
        # Configure mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [
            {"word": "John", "entity_group": "PER", "score": 0.95, "start": 0, "end": 4},
            {"word": "New York", "entity_group": "LOC", "score": 0.9, "start": 12, "end": 20},
            {"word": "IBM", "entity_group": "ORG", "score": 0.85, "start": 25, "end": 28},
            {"word": "low confidence", "entity_group": "MISC", "score": 0.6, "start": 30, "end": 43}
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create instance
        ner = NamedEntityRecognizer(device=device)
        
        # Patch the pipeline instance
        ner.ner_pipeline = mock_pipeline_instance
        
        # Test extraction
        entities = ner.extract_entities("John works at New York for IBM")
        
        # Verify only high confidence entities are returned (3 out of 4)
        assert len(entities) == 3
        
        # Verify entity details
        assert entities[0]["text"] == "John"
        assert entities[0]["type"] == "PER"
        assert entities[0]["confidence"] > 0.9
        
        assert entities[1]["text"] == "New York"
        assert entities[1]["type"] == "LOC"
        
        assert entities[2]["text"] == "IBM"
        assert entities[2]["type"] == "ORG"
        
        # Verify low confidence entity was filtered out
        assert not any(e["text"] == "low confidence" for e in entities)
        
        # Test with list input
        ner.extract_entities(["John lives in New York", "He works at IBM"])
        # Verify pipeline was called with joined text
        assert mock_pipeline_instance.called
        args, _ = mock_pipeline_instance.call_args
        assert "John lives in New York" in args[0]
        assert "He works at IBM" in args[0]
    
    @patch('src.nlp.ner.pipeline')
    @patch('src.nlp.ner.AutoTokenizer')
    @patch('src.nlp.ner.AutoModelForTokenClassification')
    def test_extract_entity_types(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test extraction of entity types."""
        # Configure mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [
            {"word": "John", "entity_group": "PER", "score": 0.95, "start": 0, "end": 4},
            {"word": "New York", "entity_group": "LOC", "score": 0.9, "start": 12, "end": 20},
            {"word": "IBM", "entity_group": "ORG", "score": 0.85, "start": 25, "end": 28},
            {"word": "Jane", "entity_group": "PER", "score": 0.92, "start": 40, "end": 44}
        ]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Create instance
        ner = NamedEntityRecognizer(device=device)
        
        # Patch the pipeline instance
        ner.ner_pipeline = mock_pipeline_instance
        
        # Test extraction by type
        entity_types = ner.extract_entity_types("John works at New York for IBM with Jane")
        
        # Verify grouping by type
        assert "PER" in entity_types
        assert "LOC" in entity_types
        assert "ORG" in entity_types
        
        assert len(entity_types["PER"]) == 2
        assert "John" in entity_types["PER"]
        assert "Jane" in entity_types["PER"]
        
        assert len(entity_types["LOC"]) == 1
        assert "New York" in entity_types["LOC"]
        
        assert len(entity_types["ORG"]) == 1
        assert "IBM" in entity_types["ORG"]
        
        # Test with specific types filter
        entity_types = ner.extract_entity_types(
            "John works at New York for IBM with Jane",
            entity_types=["PER"]
        )
        
        # Verify only PER entities are returned
        assert "PER" in entity_types
        assert "LOC" not in entity_types
        assert "ORG" not in entity_types
    
    @patch('src.nlp.ner.pipeline')
    @patch('src.nlp.ner.AutoTokenizer')
    @patch('src.nlp.ner.AutoModelForTokenClassification')
    def test_get_supported_entity_types(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test getting supported entity types."""
        # Configure mocks
        mock_model_instance = MagicMock()
        mock_model_instance.config.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Create instance
        ner = NamedEntityRecognizer(device=device, model_name="custom-model")
        
        # Test getting supported entity types
        entity_types = ner.get_supported_entity_types()
        
        # Verify entity types
        assert len(entity_types) == 3
        assert "O" in entity_types
        assert "B-PER" in entity_types
        assert "I-PER" in entity_types


@pytest.mark.parametrize("device", [None, "cpu"])
class TestDocumentClassifier:
    """Test cases for DocumentClassifier."""
    
    @patch('src.nlp.classifier.pipeline')
    @patch('src.nlp.classifier.AutoTokenizer')
    @patch('src.nlp.classifier.AutoModelForSequenceClassification')
    def test_init(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test initialization."""
        # Configure mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()
        
        # Test initialization
        classifier = DocumentClassifier(device=device)
        
        # Verify model loaded correctly
        mock_tokenizer.from_pretrained.assert_called_once_with("cross-encoder/nli-distilroberta-base")
        mock_model.from_pretrained.assert_called_once_with("cross-encoder/nli-distilroberta-base")
        mock_pipeline.assert_called_once()
        
        # Check initialization properties
        assert classifier.model_name == "cross-encoder/nli-distilroberta-base"
        assert classifier.confidence_threshold == 0.7
        assert classifier.device == "cpu"  # Always cpu in tests
        assert len(classifier.document_types) > 0
        assert len(classifier.priority_levels) == 4
    
    @patch('src.nlp.classifier.pipeline')
    @patch('src.nlp.classifier.AutoTokenizer')
    @patch('src.nlp.classifier.AutoModelForSequenceClassification')
    def test_classify_document(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test document classification."""
        # Configure mocks
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.return_value = {
            'labels': ['Contract', 'Invoice', 'Report'],
            'scores': [0.8, 0.15, 0.05]
        }
        mock_pipeline.return_value = mock_classifier_instance
        
        # Create instance
        classifier = DocumentClassifier(device=device)
        
        # Patch the classifier instance
        classifier.classifier = mock_classifier_instance
        
        # Mock _classify_document_priority
        classifier._classify_document_priority = MagicMock(
            return_value=("High", 0.85, [{"label": "High", "confidence": 0.85}])
        )
        
        # Test classification
        result = classifier.classify_document(
            "This agreement is made between Party A and Party B for services."
        )
        
        # Verify document type classification
        assert result["success"] is True
        assert result["document_type"] == "Contract"
        assert result["document_type_confidence"] == 0.8
        assert result["priority"] == "High"
        assert result["priority_confidence"] == 0.85
        
        # Test with only type classification
        result = classifier.classify_document(
            "This agreement is made between Party A and Party B.",
            classify_priority=False
        )
        
        assert result["document_type"] == "Contract"
        assert result["priority"] is None
        
        # Test with only priority classification
        result = classifier.classify_document(
            "This agreement is made between Party A and Party B.",
            classify_type=False
        )
        
        assert result["document_type"] is None
        assert result["priority"] == "High"
    
    @patch('src.nlp.classifier.pipeline')
    @patch('src.nlp.classifier.AutoTokenizer')
    @patch('src.nlp.classifier.AutoModelForSequenceClassification')
    def test_keyword_based_classification(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test keyword-based classification."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        classifier = DocumentClassifier(device=device)
        
        # Test with contract keywords
        doc_type, confidence = classifier._keyword_based_classification(
            "This agreement sets forth the terms and conditions between the parties."
        )
        
        assert doc_type == "Contract"
        assert confidence > 0.7
        
        # Test with invoice keywords
        doc_type, confidence = classifier._keyword_based_classification(
            "INVOICE\nAmount due: $500\nPayment terms: Net 30"
        )
        
        assert doc_type == "Invoice"
        assert confidence > 0.7
    
    @patch('src.nlp.classifier.pipeline')
    @patch('src.nlp.classifier.AutoTokenizer')
    @patch('src.nlp.classifier.AutoModelForSequenceClassification')
    def test_classify_document_priority(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test document priority classification."""
        # Configure mocks
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.return_value = {
            'labels': ['Low', 'Medium', 'High', 'Urgent'],
            'scores': [0.1, 0.2, 0.6, 0.1]
        }
        mock_pipeline.return_value = mock_classifier_instance
        
        # Create instance
        classifier = DocumentClassifier(device=device)
        
        # Patch the classifier instance
        classifier.classifier = mock_classifier_instance
        
        # Test classification
        priority, confidence, all_results = classifier._classify_document_priority(
            "This is a high priority document that needs immediate attention."
        )
        
        assert priority == "High"
        assert confidence >= 0.6
        assert len(all_results) == 4
        
        # Test with urgent keyword
        priority, confidence, all_results = classifier._classify_document_priority(
            "URGENT: Please review this document as soon as possible."
        )
        
        assert priority == "Urgent"
        assert confidence > 0.8  # Keyword match should have high confidence
    
    @patch('src.nlp.classifier.pipeline')
    @patch('src.nlp.classifier.AutoTokenizer')
    @patch('src.nlp.classifier.AutoModelForSequenceClassification')
    def test_extract_features(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test feature extraction."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        classifier = DocumentClassifier(device=device)
        
        # Test feature extraction
        features = classifier.extract_features(
            "This document contains bullet points:\n• Point 1\n• Point 2\nAnd some headings:\n# Heading 1"
        )
        
        # Verify basic features
        assert "character_count" in features
        assert "word_count" in features
        assert features["character_count"] > 0
        assert features["word_count"] > 0
        
        # Verify structure detection
        assert features["has_bullet_points"] is True
        assert features["has_headings"] is True
        
        # Test with list input
        features = classifier.extract_features(
            ["Paragraph 1 with numbers:", "1. First item", "2. Second item"]
        )
        
        assert features["has_numbered_lists"] is True


@pytest.mark.parametrize("device", [None, "cpu"])
class TestDocumentSummarizer:
    """Test cases for DocumentSummarizer."""
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_init(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test initialization."""
        # Configure mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()
        
        # Test initialization
        summarizer = DocumentSummarizer(device=device)
        
        # Verify model loaded correctly
        mock_tokenizer.from_pretrained.assert_called_once_with("sshleifer/distilbart-cnn-12-6")
        mock_model.from_pretrained.assert_called_once_with("sshleifer/distilbart-cnn-12-6")
        mock_pipeline.assert_called_once()
        
        # Check initialization properties
        assert summarizer.model_name == "sshleifer/distilbart-cnn-12-6"
        assert summarizer.device == "cpu"  # Always cpu in tests
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_summarize(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test document summarization."""
        # Configure mocks
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.return_value = [{
            "summary_text": "This is a summary."
        }]
        mock_pipeline.return_value = mock_summarizer_instance
        
        # Create a tokenizer mock with limited model_max_length
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.model_max_length = 1024
        mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Patch the summarizer instance
        summarizer.summarizer = mock_summarizer_instance
        summarizer.tokenizer = mock_tokenizer_instance
        
        # Mock _calculate_metrics
        summarizer._calculate_metrics = MagicMock(
            return_value={
                "compression_ratio": 0.2,
                "original_length": 100,
                "summary_length": 20
            }
        )
        
        # Mock _clean_summary
        summarizer._clean_summary = MagicMock(return_value="This is a summary.")
        
        # Test abstractive summarization
        result = summarizer.summarize(
            "This is a test document that needs to be summarized.",
            summary_length="short",
            method="abstractive"
        )
        
        # Verify summary result
        assert result["success"] is True
        assert result["summary"] == "This is a summary."
        assert result["method"] == "abstractive"
        assert result["length"] == "short"
        assert "metrics" in result
        assert result["metrics"]["compression_ratio"] == 0.2
        
        # Test with list input
        result = summarizer.summarize(
            ["Paragraph 1.", "Paragraph 2."],
            summary_length="medium",
            method="abstractive"
        )
        
        assert result["success"] is True
        assert result["summary"] == "This is a summary."
        assert result["length"] == "medium"
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_extractive_summarization(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test extractive summarization."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Test extractive summarization
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        # Mock the tfidf_matrix for deterministic testing
        summarizer.tfidf = MagicMock()
        mock_tfidf_matrix = MagicMock()
        mock_tfidf_matrix.sum.return_value = [[0.5], [0.8], [0.3], [0.2], [0.6]]
        summarizer.tfidf.fit_transform.return_value = mock_tfidf_matrix
        
        summary = summarizer._extractive_summarization(text, max_sentences=2)
        
        # Verify summary contains some of the original sentences
        assert len(summary) < len(text)
        assert summary.count(".") <= 2  # At most 2 sentences
        
        # Test error handling
        summarizer.tfidf.fit_transform.side_effect = Exception("TFIDF error")
        summary = summarizer._extractive_summarization(text, max_sentences=2)
        
        # Should fall back to a simpler method
        assert len(summary) < len(text)
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_abstractive_summarization_chunked(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test chunked abstractive summarization for long texts."""
        # Configure mocks
        mock_summarizer_instance = MagicMock()
        mock_summarizer_instance.return_value = [{
            "summary_text": "Chunk summary."
        }]
        mock_pipeline.return_value = mock_summarizer_instance
        
        # Create a tokenizer mock with limited model_max_length
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.model_max_length = 50  # Small to force chunking
        mock_tokenizer_instance.encode.return_value = torch.tensor([[i for i in range(60)]])  # Longer than limit
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Patch the summarizer instance
        summarizer.summarizer = mock_summarizer_instance
        summarizer.tokenizer = mock_tokenizer_instance
        
        # Test chunked summarization with a long text
        long_text = "This is a very long text. " * 20
        
        result = summarizer._abstractive_summarization_chunked(
            long_text,
            max_length=50,
            min_length=10
        )
        
        # Verify result is as expected
        assert "Chunk summary" in result
        mock_summarizer_instance.assert_called()  # Should be called for each chunk
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_clean_summary(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test summary cleaning."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Test cleaning with extra whitespace
        dirty_summary = "  This  is  a  summary with  extra  spaces.  "
        clean_summary = summarizer._clean_summary(dirty_summary)
        
        assert clean_summary == "This is a summary with extra spaces."
        
        # Test adding ending punctuation
        no_end_mark = "This summary has no ending punctuation"
        clean_summary = summarizer._clean_summary(no_end_mark)
        
        assert clean_summary.endswith(".")
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_calculate_metrics(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test summary metrics calculation."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Test metrics calculation
        original_text = "This is a long original text with multiple words. It contains information that needs to be summarized."
        summary = "A short summary."
        
        metrics = summarizer._calculate_metrics(original_text, summary)
        
        assert "compression_ratio" in metrics
        assert "original_length" in metrics
        assert "summary_length" in metrics
        assert metrics["original_length"] > metrics["summary_length"]
        assert 0 < metrics["compression_ratio"] < 1
    
    @patch('src.nlp.summarizer.pipeline')
    @patch('src.nlp.summarizer.AutoTokenizer')
    @patch('src.nlp.summarizer.AutoModelForSeq2SeqLM')
    def test_evaluate_summary(self, mock_model, mock_tokenizer, mock_pipeline, device):
        """Test summary evaluation."""
        # Configure mocks
        mock_pipeline.return_value = MagicMock()
        
        # Create instance
        summarizer = DocumentSummarizer(device=device)
        
        # Test evaluation
        original_text = "The quick brown fox jumps over the lazy dog. It was too hot outside."
        summary = "The fox jumps over the dog."
        
        evaluation = summarizer.evaluate_summary(original_text, summary)
        
        assert "coverage" in evaluation
        assert "density" in evaluation
        assert "novelty" in evaluation
        assert 0 <= evaluation["coverage"] <= 1
        assert 0 <= evaluation["density"] <= 1
        assert 0 <= evaluation["novelty"] <= 1