import fitz  # PyMuPDF
import hashlib
import textwrap
from typing import List, Tuple

CHUNK_SIZE = 2000  # character-level estimate

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of specified size."""
    return textwrap.wrap(text, chunk_size)

def generate_content_hash(text: str) -> str:
    """Generate SHA-256 hash of the text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

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