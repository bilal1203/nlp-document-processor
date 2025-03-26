"""
NLP Processor module that combines multiple NLP operations.
"""
import logging
from typing import Dict, Any, List, Union, Optional

from src.nlp.ner import NamedEntityRecognizer
from src.nlp.classifier import DocumentClassifier
from src.nlp.summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    NLP Processor that combines named entity recognition, 
    document classification, and summarization.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 load_ner: bool = True,
                 load_classifier: bool = True,
                 load_summarizer: bool = True):
        """
        Initialize the NLP Processor.
        
        Args:
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
            load_ner: Whether to load the NER module
            load_classifier: Whether to load the document classifier
            load_summarizer: Whether to load the summarizer
        """
        self.device = device
        
        # Load components based on flags
        self.ner = None
        self.classifier = None
        self.summarizer = None
        
        if load_ner:
            try:
                self.ner = NamedEntityRecognizer(device=device)
            except Exception as e:
                logger.error(f"Failed to load NER: {str(e)}")
                
        if load_classifier:
            try:
                self.classifier = DocumentClassifier(device=device)
            except Exception as e:
                logger.error(f"Failed to load classifier: {str(e)}")
                
        if load_summarizer:
            try:
                self.summarizer = DocumentSummarizer(device=device)
            except Exception as e:
                logger.error(f"Failed to load summarizer: {str(e)}")
                
        logger.info(f"Initialized NLP Processor with components: "
                   f"NER={self.ner is not None}, "
                   f"Classifier={self.classifier is not None}, "
                   f"Summarizer={self.summarizer is not None}")
    
    def process_document(self, 
                        text: Union[str, List[str]],
                        perform_ner: bool = True,
                        perform_classification: bool = True,
                        perform_summarization: bool = True,
                        summary_length: str = "medium",
                        summary_method: str = "abstractive") -> Dict[str, Any]:
        """
        Process a document with multiple NLP operations.
        
        Args:
            text: Document text or list of text chunks
            perform_ner: Whether to perform named entity recognition
            perform_classification: Whether to perform document classification
            perform_summarization: Whether to perform summarization
            summary_length: Length of summary ('short', 'medium', 'detailed')
            summary_method: Summarization method ('abstractive', 'extractive', 'hybrid')
            
        Returns:
            Dictionary with combined results
        """
        results = {
            "success": True,
            "text_length": len(text) if isinstance(text, str) else sum(len(t) for t in text)
        }
        
        # Perform NER
        if perform_ner and self.ner:
            try:
                ner_results = self.ner.extract_entities(text)
                results["entities"] = ner_results
                
                # Also extract entity types
                entity_types = self.ner.extract_entity_types(text)
                results["entity_types"] = entity_types
            except Exception as e:
                logger.error(f"Error in NER processing: {str(e)}")
                results["entities"] = {"error": str(e)}
        
        # Perform classification
        if perform_classification and self.classifier:
            try:
                classification_results = self.classifier.classify_document(text)
                results["classification"] = classification_results
                
                # Extract features
                features = self.classifier.extract_features(text)
                results["features"] = features
            except Exception as e:
                logger.error(f"Error in classification: {str(e)}")
                results["classification"] = {"error": str(e)}
        
        # Perform summarization
        if perform_summarization and self.summarizer:
            try:
                summary_results = self.summarizer.summarize(
                    text, 
                    summary_length=summary_length,
                    method=summary_method
                )
                results["summary"] = summary_results
            except Exception as e:
                logger.error(f"Error in summarization: {str(e)}")
                results["summary"] = {"error": str(e)}
        
        return results