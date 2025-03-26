"""
NLP module for document analysis using natural language processing.
"""
from src.nlp.ner import NamedEntityRecognizer
from src.nlp.classifier import DocumentClassifier
from src.nlp.summarizer import DocumentSummarizer
from src.nlp.nlp_processor import NLPProcessor

__all__ = [
    'NamedEntityRecognizer',
    'DocumentClassifier',
    'DocumentSummarizer',
    'NLPProcessor',
]