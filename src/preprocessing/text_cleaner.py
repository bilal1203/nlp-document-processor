"""
Text Cleaner module for normalizing and cleaning text from various document formats.
"""
import re
import logging
import unicodedata
from typing import List, Union, Dict, Any, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Class for cleaning and normalizing text extracted from documents.
    """
    
    def __init__(self, 
                 language: str = 'english',
                 remove_stopwords: bool = False,
                 remove_punctuation: bool = False,
                 lowercase: bool = True,
                 normalize_whitespace: bool = True):
        """
        Initialize the text cleaner.
        
        Args:
            language (str): Language for stopwords and tokenization
            remove_stopwords (bool): Whether to remove stopwords
            remove_punctuation (bool): Whether to remove punctuation
            lowercase (bool): Whether to convert text to lowercase
            normalize_whitespace (bool): Whether to normalize whitespace
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        
        # Load stopwords if needed
        self.stopwords = set()
        if remove_stopwords:
            try:
                self.stopwords = set(stopwords.words(language))
            except LookupError:
                logger.warning(f"Stopwords not available for language: {language}")
            except Exception as e:
                logger.error(f"Error loading stopwords: {str(e)}")
        
        logger.info(f"Initialized TextCleaner with settings: "
                   f"language={language}, "
                   f"remove_stopwords={remove_stopwords}, "
                   f"remove_punctuation={remove_punctuation}, "
                   f"lowercase={lowercase}, "
                   f"normalize_whitespace={normalize_whitespace}")
    
    def clean_text(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Clean and normalize the input text.
        
        Args:
            text: Input text (str or list of str)
            
        Returns:
            Cleaned text in the same format as input (str or list of str)
        """
        if isinstance(text, list):
            return [self._clean_text_string(t) for t in text]
        else:
            return self._clean_text_string(text)
    
    def _clean_text_string(self, text: str) -> str:
        """
        Clean and normalize a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if self.remove_stopwords and self.stopwords:
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word.lower() not in self.stopwords])
        
        return text
    
    def extract_sentences(self, text: Union[str, List[str]]) -> List[str]:
        """
        Extract sentences from the input text.
        
        Args:
            text: Input text (str or list of str)
            
        Returns:
            List of sentences
        """
        if isinstance(text, list):
            # Combine paragraphs into a single string
            text = ' '.join(text)
        
        try:
            # Use NLTK's sentence tokenizer
            sentences = sent_tokenize(text, language=self.language)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error extracting sentences: {str(e)}")
            # Fallback to basic splitting
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """
        Extract paragraphs from the input text.
        
        Args:
            text: Input text string
            
        Returns:
            List of paragraphs
        """
        if isinstance(text, list):
            # Already in paragraph format
            return [p for p in text if p.strip()]
        
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and return non-empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def extract_keywords(self, text: Union[str, List[str]], top_n: int = 10) -> List[str]:
        """
        Extract potential keywords from the text.
        Uses a simple frequency-based approach.
        
        Args:
            text: Input text (str or list of str)
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        if isinstance(text, list):
            # Combine paragraphs into a single string
            text = ' '.join(text)
        
        # Make a copy of the settings
        orig_remove_stopwords = self.remove_stopwords
        orig_remove_punctuation = self.remove_punctuation
        orig_lowercase = self.lowercase
        
        # Temporarily modify settings
        self.remove_stopwords = True
        self.remove_punctuation = True
        self.lowercase = True
        
        try:
            # Clean the text
            cleaned_text = self._clean_text_string(text)
            words = word_tokenize(cleaned_text)
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top N keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:top_n]]
            
        finally:
            # Restore original settings
            self.remove_stopwords = orig_remove_stopwords
            self.remove_punctuation = orig_remove_punctuation
            self.lowercase = orig_lowercase