"""
Document Summarization module for generating summaries from text.
"""
import logging
import re
from typing import List, Dict, Any, Union, Optional, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    Document Summarization using transformer models.
    """
    
    def __init__(self, 
                model_name: str = "sshleifer/distilbart-cnn-12-6",
                device: Optional[str] = None):
        """
        Initialize the Document Summarizer.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to use for inference ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Document Summarizer with model {model_name} on {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            # Initialize TF-IDF vectorizer for extractive summarization
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 1)
            )
            
            logger.info(f"Document Summarizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Document Summarizer: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load Document Summarizer: {str(e)}")
    
    def summarize(self, 
                 text: Union[str, List[str]], 
                 summary_length: str = "medium",
                 method: str = "abstractive") -> Dict[str, Any]:
        """
        Generate a summary of the document.
        
        Args:
            text: Document text or list of document text chunks
            summary_length: Length of summary ('short', 'medium', 'detailed')
            method: Summarization method ('abstractive', 'extractive', 'hybrid')
            
        Returns:
            Dictionary with summarization results
        """
        # Prepare text for summarization
        if isinstance(text, list):
            processed_text = " ".join(text)
        else:
            processed_text = text
        
        # Determine max and min length based on summary_length
        text_word_count = len(processed_text.split())
        
        # Define length parameters for each summary type
        length_params = {
            "short": {
                "max_length": min(100, text_word_count // 5),
                "min_length": min(30, text_word_count // 10)
            },
            "medium": {
                "max_length": min(200, text_word_count // 3),
                "min_length": min(50, text_word_count // 6)
            },
            "detailed": {
                "max_length": min(400, text_word_count // 2),
                "min_length": min(100, text_word_count // 4)
            }
        }
        
        # Get parameters for the requested length
        params = length_params.get(summary_length, length_params["medium"])
        
        try:
            # Generate summary based on requested method
            if method == "extractive":
                summary = self._extractive_summarization(
                    processed_text, 
                    max_sentences=params["max_length"] // 20  # Approximate sentence count
                )
            elif method == "hybrid":
                # Perform extractive first, then abstractive
                extractive_summary = self._extractive_summarization(
                    processed_text, 
                    max_sentences=params["max_length"] // 15
                )
                
                # Run abstractive on the extractive result
                abstractive_result = self.summarizer(
                    extractive_summary,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                    max_new_tokens=params["max_length"],
                    truncation=True
                )
                summary = abstractive_result[0]["summary_text"]
            else:  # abstractive by default
                # Check if text is too long for the model
                max_model_length = self.tokenizer.model_max_length
                input_ids = self.tokenizer.encode(processed_text, return_tensors="pt")
                
                if input_ids.shape[1] > max_model_length:
                    # Text is too long, use chunking
                    summary = self._abstractive_summarization_chunked(
                        processed_text, 
                        max_length=params["max_length"],
                        min_length=params["min_length"]
                    )
                else:
                    # Text fits within model limits
                    abstractive_result = self.summarizer(
                        processed_text,
                        max_length=params["max_length"],
                        min_length=params["min_length"],
                        max_new_tokens=params["max_length"],
                        truncation=True
                    )
                    summary = abstractive_result[0]["summary_text"]
            
            # Clean up summary
            summary = self._clean_summary(summary)
            
            # Calculate metrics
            metrics = self._calculate_metrics(processed_text, summary)
            
            return {
                "success": True,
                "summary": summary,
                "method": method,
                "length": summary_length,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extractive_summarization(self, text: str, max_sentences: int = 5) -> str:
        """
        Generate an extractive summary by selecting important sentences.
        
        Args:
            text: Document text
            max_sentences: Maximum number of sentences to include
            
        Returns:
            Extractive summary
        """
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Compute TF-IDF scores
        try:
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).tolist()
        except:
            # Fallback if TF-IDF fails
            sentence_scores = [[len(s.split())] for s in sentences]
        
        # Find top sentences by score (convert from matrix to list)
        sentence_scores = [score[0] for score in sentence_scores]
        
        # Get indices of top sentences, preserving original order
        top_indices = sorted(
            range(len(sentence_scores)), 
            key=lambda i: sentence_scores[i], 
            reverse=True
        )[:max_sentences]
        
        # Sort indices to maintain original sentence order
        top_indices = sorted(top_indices)
        
        # Construct summary from top sentences
        summary = " ".join([sentences[i] for i in top_indices])
        
        return summary
    
    def _abstractive_summarization_chunked(self, text: str, max_length: int, min_length: int) -> str:
        """
        Generate an abstractive summary using a chunking approach for long text.
        
        Args:
            text: Document text
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            
        Returns:
            Abstractive summary
        """
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Determine chunk size (in sentences)
        max_token_length = self.tokenizer.model_max_length - 100  # Buffer for safety
        chunk_summaries = []
        
        # Process text in chunks
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Get token count for sentence
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If adding this sentence exceeds the limit, process the current chunk
            if current_length + sentence_tokens > max_token_length and current_chunk:
                # Create chunk text
                chunk_text = " ".join(current_chunk)
                
                # Summarize chunk
                chunk_result = self.summarizer(
                    chunk_text,
                    max_length=max(30, max_length // 2),  # Smaller max for chunks
                    min_length=min(20, min_length // 2),  # Smaller min for chunks
                    max_new_tokens=max(30, max_length // 2),
                    truncation=True
                )
                
                chunk_summaries.append(chunk_result[0]["summary_text"])
                
                # Reset chunk
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Process any remaining text
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            
            chunk_result = self.summarizer(
                chunk_text,
                max_length=max(30, max_length // 2),
                min_length=min(20, min_length // 2),
                max_new_tokens=max(30, max_length // 2),
                truncation=True
            )
            
            chunk_summaries.append(chunk_result[0]["summary_text"])
        
        # If we have multiple chunk summaries, summarize them again
        if len(chunk_summaries) > 1:
            combined_summary = " ".join(chunk_summaries)
            
            # Check if combined summary is still too long
            combined_tokens = len(self.tokenizer.encode(combined_summary))
            
            if combined_tokens > max_token_length:
                # Get a subset of the combined summary that fits within limits
                summary_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', combined_summary)
                shortened_summary = ""
                current_length = 0
                
                for sentence in summary_sentences:
                    sentence_tokens = len(self.tokenizer.encode(sentence))
                    if current_length + sentence_tokens < max_token_length:
                        shortened_summary += sentence + " "
                        current_length += sentence_tokens
                    else:
                        break
                
                # Final summarization of the shortened summary
                final_result = self.summarizer(
                    shortened_summary.strip(),
                    max_length=max_length,
                    min_length=min_length,
                    max_new_tokens=max_length,
                    truncation=True
                )
                
                return final_result[0]["summary_text"]
            else:
                # Final summarization of the combined summaries
                final_result = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    max_new_tokens=max_length,
                    truncation=True
                )
                
                return final_result[0]["summary_text"]
        elif chunk_summaries:
            # If only one chunk, return its summary
            return chunk_summaries[0]
        else:
            # Fallback
            return "Could not generate summary."
    
    def _clean_summary(self, summary: str) -> str:
        """
        Clean up the generated summary.
        
        Args:
            summary: Raw summary
            
        Returns:
            Cleaned summary
        """
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure summary ends with proper punctuation
        if summary and summary[-1] not in ['.', '!', '?']:
            summary += '.'
        
        return summary
    
    def _calculate_metrics(self, original_text: str, summary: str) -> Dict[str, Any]:
        """
        Calculate metrics for the summary.
        
        Args:
            original_text: Original document text
            summary: Generated summary
            
        Returns:
            Dictionary with metrics
        """
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        return {
            "compression_ratio": round(summary_words / max(original_words, 1), 2),
            "original_length": original_words,
            "summary_length": summary_words
        }
    
    def evaluate_summary(self, original_text: str, summary: str) -> Dict[str, float]:
        """
        Evaluate the quality of a summary.
        
        Args:
            original_text: Original document text
            summary: Generated summary
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic evaluation metrics (without ROUGE which requires additional dependencies)
        
        # Tokenize text
        original_tokens = set(original_text.lower().split())
        summary_tokens = set(summary.lower().split())
        
        # Calculate coverage
        coverage = len(summary_tokens.intersection(original_tokens)) / max(len(summary_tokens), 1)
        
        # Calculate density (unique words ratio in summary)
        summary_words = summary.lower().split()
        density = len(set(summary_words)) / max(len(summary_words), 1)
        
        # Novelty (words in summary not in original)
        novelty = len(summary_tokens.difference(original_tokens)) / max(len(summary_tokens), 1)
        
        return {
            "coverage": round(coverage, 2),
            "density": round(density, 2),
            "novelty": round(novelty, 2)
        }