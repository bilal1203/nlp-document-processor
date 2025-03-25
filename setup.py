"""
Setup script for NLP Document Processing System.
"""
from setuptools import setup, find_packages

setup(
    name="nlp-document-processor",
    version="1.0.0",
    description="A system for processing documents with natural language processing",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nlp-document-processor",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "PyPDF2>=2.10.0",
        "pdfplumber>=0.7.0",
        "python-docx>=0.8.11",
        "transformers>=4.18.0",
        "torch>=1.11.0",
        "spacy>=3.2.0",
        "nltk>=3.6.5",
        "fastapi>=0.78.0",
        "uvicorn>=0.17.6",
        "pydantic>=1.9.0",
        "python-multipart>=0.0.5",
        "gradio>=3.0.0",
        "chardet>=4.0.0",
        "rapidfuzz>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "nlp-api=src.api.main:main",
            "nlp-ui=src.frontend.app:main",
        ],
    },
)