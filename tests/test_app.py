"""
Comprehensive test suite for the LLM Study Coach backend application.
Tests all endpoints, error cases, and edge cases.
"""

import pytest
import io
import os
from unittest.mock import patch, MagicMock
from flask import Flask
from app import app, supabase, llm_client


# Test configuration
@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_pdf_content():
    """Create a sample PDF-like content for testing."""
    # This is just bytes that represent a fake PDF
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def headers():
    """Standard headers for API requests."""
    return {"X-User-ID": "test-user-123"}


class TestHealthCheck:
    """Test basic app functionality."""

    def test_index_route(self, client):
        """Test the index route returns successfully."""
        response = client.get("/")
        assert response.status_code == 200


class TestProcessPDFEndpoint:
    """Test the /api/process-pdf endpoint."""

    def test_process_pdf_no_file(self, client, headers):
        """Test process PDF without file should return 400."""
        response = client.post("/api/process-pdf", headers=headers)
        assert response.status_code == 400
        assert "No file provided" in response.get_json()["error"]

    def test_process_pdf_no_user_id(self, client, sample_pdf_content):
        """Test process PDF without user ID should return 401."""
        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data)
        assert response.status_code == 401
        assert "User ID not provided" in response.get_json()["error"]

    def test_process_pdf_invalid_file_type(self, client, headers):
        """Test process PDF with non-PDF file should return 400."""
        data = {
            "file": (io.BytesIO(b"not a pdf"), "test.txt"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)
        assert response.status_code == 400
        assert "File must be a PDF" in response.get_json()["error"]

    def test_process_pdf_no_subject(self, client, headers, sample_pdf_content):
        """Test process PDF without subject should return 400."""
        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)
        assert response.status_code == 400
        assert "Subject not provided" in response.get_json()["error"]

    def test_process_pdf_no_content_hash(self, client, headers, sample_pdf_content):
        """Test process PDF without content hash should return 400."""
        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "subject": "Test Subject",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)
        assert response.status_code == 400
        assert "Content hash not provided" in response.get_json()["error"]

    @patch("app.supabase")
    @patch("app.process_pdf")
    @patch("app.llm_client.generate_notes_for_chunks")
    def test_process_pdf_success_new_notes(
        self,
        mock_generate_notes,
        mock_process_pdf,
        mock_supabase,
        client,
        headers,
        sample_pdf_content,
    ):
        """Test successful PDF processing with new notes generation."""
        # Mock the database check for existing notes
        mock_supabase.table().select().eq().execute.return_value.data = []

        # Mock PDF processing
        mock_process_pdf.return_value = (
            "extracted text",
            ["chunk1", "chunk2"],
            "test-hash",
        )

        # Mock notes generation
        mock_generate_notes.return_value = ["note1", "note2"]

        # Mock database insertion
        mock_supabase.table().insert().execute.return_value = MagicMock()

        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "Generated new notes" in result["message"]
        assert "content" in result

    @patch("app.supabase")
    def test_process_pdf_existing_notes(
        self, mock_supabase, client, headers, sample_pdf_content
    ):
        """Test PDF processing when notes already exist."""
        # Mock existing notes in database
        existing_note = {
            "content": "existing note content",
            "model_used": "test-model",
            "generated_at": "2023-01-01T00:00:00",
        }
        mock_supabase.table().select().eq().execute.return_value.data = [existing_note]

        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "existing-hash",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "Retrieved existing notes" in result["message"]
        assert result["content"] == existing_note["content"]


class TestGetNotesEndpoint:
    """Test the /api/notes/<content_hash> endpoint."""

    @patch("app.supabase")
    def test_get_notes_success(self, mock_supabase, client):
        """Test successful notes retrieval."""
        note_data = {
            "content": "test note content",
            "model_used": "test-model",
            "generated_at": "2023-01-01T00:00:00",
        }
        mock_supabase.table().select().eq().execute.return_value.data = [note_data]

        response = client.get("/api/notes/test-hash")
        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert result["content"] == note_data["content"]

    @patch("app.supabase")
    def test_get_notes_not_found(self, mock_supabase, client):
        """Test notes retrieval when notes don't exist."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        response = client.get("/api/notes/nonexistent-hash")
        assert response.status_code == 404
        assert "Notes not found" in response.get_json()["error"]


class TestGenerateHashEndpoint:
    """Test the /api/generate-hash endpoint."""

    def test_generate_hash_no_file(self, client, headers):
        """Test generate hash without file should return 400."""
        response = client.post("/api/generate-hash", headers=headers)
        assert response.status_code == 400
        assert "No file provided" in response.get_json()["error"]

    def test_generate_hash_no_user_id(self, client, sample_pdf_content):
        """Test generate hash without user ID should return 401."""
        data = {"file": (io.BytesIO(sample_pdf_content), "test.pdf")}
        response = client.post("/api/generate-hash", data=data)
        assert response.status_code == 401
        assert "User ID not provided" in response.get_json()["error"]

    def test_generate_hash_invalid_file_type(self, client, headers):
        """Test generate hash with non-PDF file should return 400."""
        data = {"file": (io.BytesIO(b"not a pdf"), "test.txt")}
        response = client.post("/api/generate-hash", data=data, headers=headers)
        assert response.status_code == 400
        assert "File must be a PDF" in response.get_json()["error"]

    @patch("app.process_pdf")
    def test_generate_hash_success(
        self, mock_process_pdf, client, headers, sample_pdf_content
    ):
        """Test successful hash generation."""
        mock_process_pdf.return_value = ("text", ["chunks"], "generated-hash")

        data = {"file": (io.BytesIO(sample_pdf_content), "test.pdf")}
        response = client.post("/api/generate-hash", data=data, headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        assert result["content_hash"] == "generated-hash"


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("app.process_pdf")
    def test_process_pdf_exception(
        self, mock_process_pdf, client, headers, sample_pdf_content
    ):
        """Test PDF processing when an exception occurs."""
        mock_process_pdf.side_effect = Exception("PDF processing failed")

        data = {
            "file": (io.BytesIO(sample_pdf_content), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data, headers=headers)

        assert response.status_code == 500
        assert "PDF processing failed" in response.get_json()["error"]

    @patch("app.supabase")
    def test_get_notes_exception(self, mock_supabase, client):
        """Test notes retrieval when database throws exception."""
        mock_supabase.table().select().eq().execute.side_effect = Exception(
            "Database error"
        )

        response = client.get("/api/notes/test-hash")
        assert response.status_code == 500
        assert "Database error" in response.get_json()["error"]


if __name__ == "__main__":
    pytest.main([__file__])
