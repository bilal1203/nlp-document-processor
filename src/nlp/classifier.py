"""
Document Classification module for classifying documents by type and priority.
"""
import logging
import re
from typing import List, Dict, Any, Union, Optional, Tuple

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

class DocumentClassifier:
    """
    Document Classification using transformer models.
    """
    
    def __init__(self, 
                model_name: str = "cross-encoder/nli-distilroberta-base",
                device: Optional[str] = None,
                confidence_threshold: float = 0.7):
        """
        Initialize the Document Classifier.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
            confidence_threshold: Minimum confidence score for classifications
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Define document types
        self.document_types = [
            "Contract", "Invoice", "Report", "Email", "Memo", 
            "Letter", "Resume", "Financial Statement", "Legal Brief", 
            "Press Release", "Meeting Minutes", "Proposal"
        ]
        
        # Define priority levels
        self.priority_levels = ["Low", "Medium", "High", "Urgent"]
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Document Classifier with model {model_name} on {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize TF-IDF vectorizer for feature extraction
            self.tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info(f"Document Classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Document Classifier: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load Document Classifier: {str(e)}")
    
    def classify_document(self, 
                         text: Union[str, List[str]], 
                         classify_type: bool = True,
                         classify_priority: bool = True) -> Dict[str, Any]:
        """
        Classify a document by type and/or priority.
        
        Args:
            text: Document text or list of document text chunks
            classify_type: Whether to classify document type
            classify_priority: Whether to classify document priority
            
        Returns:
            Dictionary with classification results
        """
        # Prepare text for classification
        if isinstance(text, list):
            # Use first chunk and some samples from the rest
            if len(text) > 1:
                # Use first chunk, a middle chunk, and last chunk
                processed_text = text[0]
                
                # Sample from the rest if there are many chunks
                if len(text) > 5:
                    middle_idx = len(text) // 2
                    processed_text += "\n" + text[middle_idx]
                    processed_text += "\n" + text[-1]
            else:
                processed_text = text[0]
        else:
            processed_text = text
        
        # Initialize result dictionary
        result = {
            "success": True,
            "document_type": None,
            "document_type_confidence": 0.0,
            "priority": None,
            "priority_confidence": 0.0
        }
        
        try:
            # Classify document type
            if classify_type:
                type_result = self._classify_document_type(processed_text)
                result["document_type"] = type_result[0]
                result["document_type_confidence"] = type_result[1]
                result["document_type_all"] = type_result[2]
            
            # Classify priority
            if classify_priority:
                priority_result = self._classify_document_priority(processed_text)
                result["priority"] = priority_result[0]
                result["priority_confidence"] = priority_result[1]
                result["priority_all"] = priority_result[2]
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _classify_document_type(self, text: str) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Classify document by type.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (type, confidence, all_results)
        """
        # Perform zero-shot classification
        type_result = self.classifier(
            text, 
            self.document_types, 
            hypothesis_template="This is a {}."
        )
        
        # Get top prediction and confidence
        document_type = type_result['labels'][0]
        confidence = type_result['scores'][0]
        
        # Prepare all results
        all_results = [
            {"label": label, "confidence": score}
            for label, score in zip(type_result['labels'], type_result['scores'])
        ]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # Try keyword-based matching as fallback
            keyword_type, keyword_confidence = self._keyword_based_classification(text)
            
            if keyword_confidence > confidence:
                document_type = keyword_type
                confidence = keyword_confidence
                
                # Update all_results
                for result in all_results:
                    if result["label"] == document_type:
                        result["confidence"] = confidence
                        break
        
        return document_type, confidence, all_results
    
    def _classify_document_priority(self, text: str) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Classify document by priority.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (priority, confidence, all_results)
        """
        # Define priority indicators
        priority_indicators = {
            "Urgent": ["urgent", "immediate", "asap", "emergency", "critical", "crucial", "vital"],
            "High": ["important", "priority", "significant", "key", "essential", "major"],
            "Medium": ["moderate", "standard", "normal", "regular", "typical"],
            "Low": ["low", "minor", "trivial", "routine", "optional", "when convenient"]
        }
        
        # Perform zero-shot classification
        priority_result = self.classifier(
            text, 
            self.priority_levels, 
            hypothesis_template="This document has {} priority."
        )
        
        # Get top prediction and confidence
        priority = priority_result['labels'][0]
        confidence = priority_result['scores'][0]
        
        # Prepare all results
        all_results = [
            {"label": label, "confidence": score}
            for label, score in zip(priority_result['labels'], priority_result['scores'])
        ]
        
        # Check for specific keywords for priority
        text_lower = text.lower()
        for level, keywords in priority_indicators.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    keyword_confidence = 0.85  # Set a high confidence for keyword matches
                    
                    if keyword_confidence > confidence:
                        priority = level
                        confidence = keyword_confidence
                        
                        # Update all_results
                        for result in all_results:
                            if result["label"] == priority:
                                result["confidence"] = confidence
                                break
                    
                    break
        
        return priority, confidence, all_results
    
    def _keyword_based_classification(self, text: str) -> Tuple[str, float]:
        """
        Perform keyword-based document classification as fallback.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (document_type, confidence)
        """
        # Define keyword indicators for document types
        type_indicators = {
            "Contract": ["agreement", "contract", "terms", "parties", "hereby", "clause", "obligations"],
            "Invoice": ["invoice", "payment", "amount", "due", "bill", "paid", "total", "subtotal"],
            "Report": ["report", "analysis", "findings", "conclusion", "summary", "results", "data"],
            "Email": ["from:", "to:", "subject:", "sent:", "wrote:", "reply", "forward", "@"],
            "Memo": ["memo", "memorandum", "note", "reminder", "internal"],
            "Letter": ["dear", "sincerely", "regards", "truly", "address"],
            "Resume": ["resume", "cv", "experience", "skills", "education", "references", "employment"],
            "Financial Statement": ["balance", "assets", "liabilities", "equity", "income", "statement", "financial"],
            "Legal Brief": ["court", "plaintiff", "defendant", "jurisdiction", "filed", "case", "law", "legal"],
            "Press Release": ["press", "release", "announces", "media", "contact", "news"],
            "Meeting Minutes": ["meeting", "attendees", "agenda", "discussed", "minutes", "action items"],
            "Proposal": ["proposal", "proposed", "solution", "recommend", "project", "plan", "scope"]
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count keyword matches for each type
        type_scores = {}
        for doc_type, keywords in type_indicators.items():
            score = 0
            for keyword in keywords:
                # Use regex for word boundary matching
                matches = re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower)
                score += len(matches)
            
            if score > 0:
                # Normalize by number of keywords
                type_scores[doc_type] = score / len(keywords)
        
        # Find type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            # Convert score to confidence (0.7-0.9 range)
            confidence = min(0.7 + (best_type[1] * 0.2), 0.9)
            return best_type[0], confidence
        
        # Default to first type with low confidence if no matches
        return self.document_types[0], 0.5
    
    def extract_features(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Extract features from document text for classification.
        
        Args:
            text: Document text or list of document text chunks
            
        Returns:
            Dictionary with extracted features
        """
        # Prepare text
        if isinstance(text, list):
            processed_text = " ".join(text)
        else:
            processed_text = text
            
        features = {}
        
        try:
            # Length features
            features["character_count"] = len(processed_text)
            features["word_count"] = len(processed_text.split())
            
            # Extract TF-IDF features (top terms)
            try:
                # Fit the vectorizer
                self.tfidf.fit([processed_text])
                
                # Get feature names
                feature_names = self.tfidf.get_feature_names_out()
                
                # Transform the text
                X = self.tfidf.transform([processed_text])
                
                # Get top terms
                indices = np.argsort(X.toarray()[0])[::-1]
                top_terms = [feature_names[i] for i in indices[:10]]
                
                features["top_terms"] = top_terms
            except:
                features["top_terms"] = []
            
            # Check for document structure indicators
            features["has_bullet_points"] = "•" in processed_text or "*" in processed_text
            features["has_numbered_lists"] = bool(re.search(r'^\d+\.', processed_text, re.MULTILINE))
            features["has_headings"] = bool(re.search(r'^#+\s', processed_text, re.MULTILINE))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}", exc_info=True)
            return {}