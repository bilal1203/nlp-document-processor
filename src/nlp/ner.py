"""
Named Entity Recognition (NER) module for extracting entities from text.
"""
import logging
from typing import List, Dict, Any, Union, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

logger = logging.getLogger(__name__)

class NamedEntityRecognizer:
    """
    Named Entity Recognition using Hugging Face transformers.
    """
    
    def __init__(self, 
                model_name: str = "dslim/bert-base-NER",
                device: Optional[str] = None,
                confidence_threshold: float = 0.8):
        """
        Initialize the Named Entity Recognizer.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence score to keep predictions
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing NER with model {model_name} on {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"  # Group entities
            )
            
            logger.info(f"NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load NER model: {str(e)}")
    
    def extract_entities(self, text: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            List of entity dictionaries with entity, type, confidence, and position
        """
        # If text is a list, join with newlines
        if isinstance(text, list):
            processed_text = "\n".join(text)
        else:
            processed_text = text
            
        try:
            # Run NER inference
            entities = self.ner_pipeline(processed_text)
            
            # Filter by confidence threshold
            filtered_entities = [
                entity for entity in entities 
                if entity['score'] >= self.confidence_threshold
            ]
            
            # Consolidate entities (removing duplicates)
            consolidated = self._consolidate_entities(filtered_entities)
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return []
    
    def _consolidate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate and format entity predictions, removing duplicates.
        
        Args:
            entities: Raw entity predictions from pipeline
            
        Returns:
            Consolidated and formatted entity list
        """
        # Dictionary to track unique entities
        unique_entities = {}
        
        for entity in entities:
            key = f"{entity['word'].lower()}:{entity['entity_group']}"
            
            # If new entity or higher confidence, update
            if key not in unique_entities or entity['score'] > unique_entities[key]['confidence']:
                unique_entities[key] = {
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                }
        
        # Return as list, sorted by position
        return sorted(unique_entities.values(), key=lambda x: x['start'])
    
    def extract_entity_types(self, text: Union[str, List[str]], 
                           entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Extract and group entities by type.
        
        Args:
            text: Input text or list of texts
            entity_types: List of entity types to extract (None for all)
            
        Returns:
            Dictionary mapping entity types to lists of entity text
        """
        # Extract all entities
        entities = self.extract_entities(text)
        
        # Group by type
        result = {}
        for entity in entities:
            entity_type = entity['type']
            
            # Skip if not in requested types
            if entity_types and entity_type not in entity_types:
                continue
                
            if entity_type not in result:
                result[entity_type] = []
            
            result[entity_type].append(entity['text'])
        
        return result
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get list of entity types supported by the model.
        
        Returns:
            List of supported entity types
        """
        # For the dslim/bert-base-NER model, return hardcoded types
        if self.model_name == "dslim/bert-base-NER":
            return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", 
                   "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        
        # For other models, try to extract from the model's config
        try:
            label_list = self.model.config.id2label.values()
            return list(label_list)
        except AttributeError:
            logger.warning("Could not determine entity types from model")
            return []