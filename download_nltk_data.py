"""
Script to download required NLTK data packages.
"""
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Try to download punkt_tab if available (needed for proper sentence tokenization)
try:
    nltk.download('punkt_tab')
except:
    print("Note: punkt_tab not available through standard download. Adding fallback tokenizer handling.")
    
    # Add code to create a fallback method or modify text_cleaner.py to handle missing punkt_tab