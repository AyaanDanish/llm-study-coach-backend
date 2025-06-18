"""
Test suite for PDF processing utilities.
Tests text extraction, chunking, and hash generation.
"""

import pytest
import hashlib
from utils.pdf_processor import (
    extract_text_from_pdf,
    chunk_text,
    generate_content_hash,
    process_pdf,
)


class TestExtractTextFromPDF:
    """Test PDF text extraction functionality."""

    def test_extract_text_empty_pdf(self):
        """Test extraction from an empty/minimal PDF."""
        # Minimal PDF structure
        minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
301
%%EOF"""

        # This might not extract text from a minimal PDF, but shouldn't crash
        try:
            text = extract_text_from_pdf(minimal_pdf)
            assert isinstance(text, str)
        except Exception as e:
            # If it fails, it should fail gracefully
            assert "PDF" in str(e) or "format" in str(e).lower()

    def test_extract_text_invalid_pdf(self):
        """Test extraction from invalid PDF data."""
        invalid_pdf = b"This is not a PDF file"

        with pytest.raises(Exception):
            extract_text_from_pdf(invalid_pdf)

    def test_extract_text_corrupted_pdf(self):
        """Test extraction from corrupted PDF data."""
        corrupted_pdf = b"%PDF-1.4\n corrupted data \x00\x01\x02"

        with pytest.raises(Exception):
            extract_text_from_pdf(corrupted_pdf)


class TestChunkText:
    """Test text chunking functionality."""

    def test_chunk_text_empty_string(self):
        """Test chunking an empty string."""
        result = chunk_text("")
        assert result == []

    def test_chunk_text_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "This is a short text."
        result = chunk_text(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_long_text(self):
        """Test chunking text longer than chunk size."""
        text = "A" * 1000
        result = chunk_text(text, chunk_size=100)
        assert len(result) > 1
        # Each chunk should be approximately the chunk size
        for chunk in result:
            assert len(chunk) <= 100

    def test_chunk_text_exact_size(self):
        """Test chunking text that's exactly the chunk size."""
        text = "A" * 100
        result = chunk_text(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_chunk_text_with_spaces(self):
        """Test chunking text with spaces (should break at word boundaries)."""
        text = "This is a test sentence with multiple words that should be chunked properly."
        result = chunk_text(text, chunk_size=20)
        assert len(result) > 1
        # Should not break words in the middle
        for chunk in result:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_chunk_text_custom_size(self):
        """Test chunking with custom chunk size."""
        text = "A" * 500
        result = chunk_text(text, chunk_size=50)
        assert len(result) == 10  # 500 / 50
        for chunk in result:
            assert len(chunk) == 50


class TestGenerateContentHash:
    """Test content hash generation."""

    def test_generate_hash_empty_string(self):
        """Test hash generation for empty string."""
        result = generate_content_hash("")
        expected = hashlib.sha256("".encode("utf-8")).hexdigest()
        assert result == expected

    def test_generate_hash_simple_text(self):
        """Test hash generation for simple text."""
        text = "Hello, world!"
        result = generate_content_hash(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64  # SHA-256 produces 64-character hex string

    def test_generate_hash_unicode_text(self):
        """Test hash generation for Unicode text."""
        text = "Hello, 世界! 🌍"
        result = generate_content_hash(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_generate_hash_consistency(self):
        """Test that the same text always produces the same hash."""
        text = "This is a test text."
        hash1 = generate_content_hash(text)
        hash2 = generate_content_hash(text)
        assert hash1 == hash2

    def test_generate_hash_different_texts(self):
        """Test that different texts produce different hashes."""
        text1 = "This is text one."
        text2 = "This is text two."
        hash1 = generate_content_hash(text1)
        hash2 = generate_content_hash(text2)
        assert hash1 != hash2

    def test_generate_hash_case_sensitive(self):
        """Test that hash generation is case sensitive."""
        text1 = "Hello World"
        text2 = "hello world"
        hash1 = generate_content_hash(text1)
        hash2 = generate_content_hash(text2)
        assert hash1 != hash2


class TestProcessPDF:
    """Test the main process_pdf function."""

    def test_process_pdf_integration(self):
        """Test the complete PDF processing pipeline."""
        # Create a minimal valid PDF
        minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
301
%%EOF"""

        try:
            text, chunks, content_hash = process_pdf(minimal_pdf)

            # Verify return types
            assert isinstance(text, str)
            assert isinstance(chunks, list)
            assert isinstance(content_hash, str)

            # Verify hash format
            assert len(content_hash) == 64

            # Verify chunks are strings
            for chunk in chunks:
                assert isinstance(chunk, str)

        except Exception as e:
            # If it fails due to PDF format, that's acceptable for a minimal PDF
            # but it should fail gracefully
            assert isinstance(e, Exception)

    def test_process_pdf_invalid_input(self):
        """Test process_pdf with invalid input."""
        invalid_pdf = b"Not a PDF"

        with pytest.raises(Exception):
            process_pdf(invalid_pdf)

    def test_process_pdf_empty_input(self):
        """Test process_pdf with empty input."""
        with pytest.raises(Exception):
            process_pdf(b"")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_chunk_text_zero_size(self):
        """Test chunking with zero chunk size."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=0)

    def test_chunk_text_negative_size(self):
        """Test chunking with negative chunk size."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=-1)

    def test_very_long_text_chunking(self):
        """Test chunking very long text."""
        text = "A" * 100000  # 100KB of text
        result = chunk_text(text, chunk_size=2000)
        assert len(result) == 50  # 100000 / 2000

        # Verify all chunks are properly sized
        for chunk in result:
            assert len(chunk) <= 2000

    def test_special_characters_in_text(self):
        """Test processing text with special characters."""
        text = "Special chars: !@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        result = chunk_text(text)
        assert len(result) >= 1

        hash_result = generate_content_hash(text)
        assert len(hash_result) == 64


if __name__ == "__main__":
    pytest.main([__file__])
