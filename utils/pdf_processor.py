import fitz  # PyMuPDF
import hashlib
import textwrap
from typing import List, Tuple

# Massive chunk size for GPT-4.1 Nano (1M+ token context window)
# Can handle entire documents in most cases
CHUNK_SIZE = 4000000  # characters - optimized for GPT-4.1 Nano's massive context


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks of specified size with smart boundary detection.

    Tries to split at natural boundaries (paragraphs, sentences) to preserve context.
    """
    # Validate chunk size
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")

    # Handle empty or whitespace-only strings
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break

        # Try to find a good breaking point within the last 500 characters
        search_start = max(end - 500, start)

        # Look for paragraph breaks first
        paragraph_break = text.rfind("\n\n", search_start, end)
        if paragraph_break > start:
            chunks.append(text[start:paragraph_break])
            start = paragraph_break + 2
            continue

        # Look for sentence endings
        sentence_break = text.rfind(". ", search_start, end)
        if sentence_break > start:
            chunks.append(text[start : sentence_break + 1])
            start = sentence_break + 2
            continue

        # Look for any line break
        line_break = text.rfind("\n", search_start, end)
        if line_break > start:
            chunks.append(text[start:line_break])
            start = line_break + 1
            continue

        # No good break found, split at character limit
        chunks.append(text[start:end])
        start = end

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def generate_content_hash(text: str) -> str:
    """Generate SHA-256 hash of the text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_pdf(file_bytes: bytes) -> Tuple[str, List[str], str]:
    """
    Process PDF file and return extracted text, chunks, and content hash.

    Returns:
        Tuple containing:
        - Full extracted text
        - List of text chunks
        - Content hash
    """
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)
    content_hash = generate_content_hash(text)
    return text, chunks, content_hash
