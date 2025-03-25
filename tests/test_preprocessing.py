"""
Tests for preprocessing module.
"""
import os
import io
import pytest
from unittest.mock import patch, MagicMock

from src.preprocessing.txt_parser import TXTParser
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.document_processor import DocumentProcessor


class TestTXTParser:
    """Test cases for TXTParser."""
    
    def test_init(self):
        """Test initialization."""
        parser = TXTParser()
        assert parser.preserve_structure is True
        assert parser.encoding is None
        
        parser = TXTParser(preserve_structure=False, encoding='utf-8')
        assert parser.preserve_structure is False
        assert parser.encoding == 'utf-8'
    
    def test_parse_string_content(self):
        """Test parsing from a text string."""
        # Create a temporary file
        with open('temp_test.txt', 'w', encoding='utf-8') as f:
            f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph.")
        
        try:
            parser = TXTParser()
            result = parser.parse('temp_test.txt')
            
            assert result['success'] is True
            assert len(result['text']) == 3  # Three paragraphs
            assert "This is a test document." in result['text'][0]
            assert "It has multiple paragraphs." in result['text'][1]
            assert "Third paragraph." in result['text'][2]
            
            # Test with preserve_structure=False
            parser = TXTParser(preserve_structure=False)
            result = parser.parse('temp_test.txt')
            
            assert result['success'] is True
            assert isinstance(result['text'], str)  # Single string, not list
            assert "This is a test document. It has multiple paragraphs. Third paragraph." == result['text']
            
        finally:
            # Clean up
            if os.path.exists('temp_test.txt'):
                os.remove('temp_test.txt')
    
    def test_parse_bytes_io(self):
        """Test parsing from a BytesIO object."""
        content = b"This is a BytesIO test.\n\nSecond paragraph."
        file_obj = io.BytesIO(content)
        
        parser = TXTParser()
        result = parser.parse(file_obj)
        
        assert result['success'] is True
        assert len(result['text']) == 2  # Two paragraphs
        assert "This is a BytesIO test." in result['text'][0]
        assert "Second paragraph." in result['text'][1]
    
    def test_is_valid_txt(self):
        """Test text file validation."""
        # Create a temporary file
        with open('temp_valid.txt', 'w', encoding='utf-8') as f:
            f.write("This is a valid text file.")
        
        # Create a non-text file
        with open('temp_invalid.bin', 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05')
        
        try:
            assert TXTParser.is_valid_txt('temp_valid.txt') is True
            assert TXTParser.is_valid_txt('temp_invalid.bin') is False
            assert TXTParser.is_valid_txt('nonexistent_file.txt') is False
            
            # Test with BytesIO
            valid_content = io.BytesIO(b"This is valid text")
            invalid_content = io.BytesIO(b'\x00\x01\x02\x03\x04\x05')
            
            assert TXTParser.is_valid_txt(valid_content) is True
            assert TXTParser.is_valid_txt(invalid_content) is False
            
        finally:
            # Clean up
            if os.path.exists('temp_valid.txt'):
                os.remove('temp_valid.txt')
            if os.path.exists('temp_invalid.bin'):
                os.remove('temp_invalid.bin')


class TestTextCleaner:
    """Test cases for TextCleaner."""
    
    def test_init(self):
        """Test initialization."""
        cleaner = TextCleaner()
        assert cleaner.language == 'english'
        assert cleaner.remove_stopwords is False
        assert cleaner.remove_punctuation is False
        assert cleaner.lowercase is True
        assert cleaner.normalize_whitespace is True
    
    def test_clean_text_string(self):
        """Test cleaning a text string."""
        cleaner = TextCleaner()
        
        # Test basic cleaning
        text = "  This is a   test  with extra  spaces.  "
        result = cleaner._clean_text_string(text)
        assert result == "this is a test with extra spaces."
        
        # Test with preserve case
        cleaner = TextCleaner(lowercase=False)
        result = cleaner._clean_text_string(text)
        assert result == "This is a test with extra spaces."
        
        # Test with punctuation removal
        cleaner = TextCleaner(remove_punctuation=True)
        result = cleaner._clean_text_string("Hello, world! How are you?")
        assert result == "hello world how are you"
        
        # Test with stopword removal
        cleaner = TextCleaner(remove_stopwords=True)
        result = cleaner._clean_text_string("This is a test of the system")
        # Removed 'is', 'a', 'of', 'the'
        assert "this" in result
        assert "test" in result
        assert "system" in result
        assert "is" not in result
        assert "a" not in result
        assert "of" not in result
        assert "the" not in result
    
    def test_clean_text_list(self):
        """Test cleaning a list of texts."""
        cleaner = TextCleaner()
        
        texts = ["  First paragraph.  ", "  Second   paragraph with spaces.  "]
        result = cleaner.clean_text(texts)
        
        assert result == ["first paragraph.", "second paragraph with spaces."]
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        cleaner = TextCleaner()
        
        text = "This is the first sentence. This is the second! Is this the third? Yes, it is."
        sentences = cleaner.extract_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "This is the first sentence."
        assert sentences[1] == "This is the second!"
        assert sentences[2] == "Is this the third?"
        assert sentences[3] == "Yes, it is."
        
        # Test with list input
        text_list = ["Paragraph one. Second sentence.", "Paragraph two!"]
        sentences = cleaner.extract_sentences(text_list)
        
        assert len(sentences) == 3
        assert sentences[0] == "Paragraph one."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Paragraph two!"
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        cleaner = TextCleaner()
        
        text = "The quick brown fox jumps over the lazy dog. Fox is quick and brown."
        keywords = cleaner.extract_keywords(text, top_n=3)
        
        assert len(keywords) == 3
        # Keywords should include 'fox', 'brown', 'quick' (most frequent meaningful words)
        assert "fox" in [k.lower() for k in keywords]
        assert "brown" in [k.lower() for k in keywords]
        assert "quick" in [k.lower() for k in keywords]


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_init(self):
        """Test initialization."""
        processor = DocumentProcessor()
        assert processor.preserve_structure is True
        assert processor.pdf_parser is not None
        assert processor.docx_parser is not None
        assert processor.txt_parser is not None
        assert processor.text_cleaner is not None
    
    @patch('src.preprocessing.document_processor.TXTParser')
    def test_process_document_txt(self, mock_txt_parser):
        """Test processing a text document."""
        # Set up mock
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = {
            'success': True,
            'text': ['Test paragraph 1', 'Test paragraph 2'],
            'metadata': {'line_count': 2}
        }
        mock_txt_parser.return_value = mock_parser_instance
        mock_txt_parser.is_valid_txt.return_value = True
        
        processor = DocumentProcessor()
        result = processor.process_document('test.txt', file_type='txt')
        
        assert result['success'] is True
        assert result['text'] == ['Test paragraph 1', 'Test paragraph 2']
        assert result['metadata'] == {'line_count': 2}
        
        # Test auto-detection
        processor.process_document('test.txt')
        mock_txt_parser.is_valid_txt.assert_called()
    
    def test_determine_file_type(self):
        """Test file type determination."""
        processor = DocumentProcessor()
        
        assert processor._determine_file_type('document.pdf') == 'pdf'
        assert processor._determine_file_type('document.docx') == 'docx'
        assert processor._determine_file_type('document.doc') == 'docx'
        assert processor._determine_file_type('document.txt') == 'txt'
        assert processor._determine_file_type('document.unknown') == 'txt'  # Default
        
        # Test with BytesIO (using mocks for brevity)
        with patch('src.preprocessing.document_processor.TXTParser') as mock_txt_parser:
            mock_txt_parser.is_valid_txt.return_value = True
            content = io.BytesIO(b"test content")
            assert processor._determine_file_type(content) == 'txt'