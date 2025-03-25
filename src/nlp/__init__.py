"""
NLP module for document analysis using natural language processing.
"""
from src.nlp.ner import NamedEntityRecognizer
from src.nlp.classifier import DocumentClassifier
from src.nlp.summarizer import DocumentSummarizer

__all__ = [
    'NamedEntityRecognizer',
    'DocumentClassifier',
    'DocumentSummarizer',
]