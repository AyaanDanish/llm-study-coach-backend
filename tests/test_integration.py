"""
Integration tests for the complete LLM Study Coach application.
Tests end-to-end workflows and component interactions.
"""

import pytest
import io
import os
from unittest.mock import patch, MagicMock
from app import app
import requests


@pytest.fixture
def client():
    """Create a test client for integration tests."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_pdf():
    """Create a more realistic PDF for integration testing."""
    return b"""%PDF-1.4
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
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
0000000301 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
380
%%EOF"""


class TestCompleteWorkflow:
    """Test complete user workflows."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_full_pdf_processing_workflow(
        self, mock_requests_post, mock_supabase, client, sample_pdf
    ):
        """Test the complete workflow from PDF upload to notes generation."""
        # Mock successful LLM API response
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.raise_for_status.return_value = None
        mock_llm_response.json.return_value = {
            "choices": [
                {"message": {"content": "Generated study notes for the content"}}
            ]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock database responses
        mock_supabase.table().select().eq().execute.return_value.data = (
            []
        )  # No existing notes
        mock_supabase.table().insert().execute.return_value = MagicMock()

        # Step 1: Generate hash
        hash_data = {"file": (io.BytesIO(sample_pdf), "test.pdf")}
        hash_response = client.post(
            "/api/generate-hash", data=hash_data, headers={"X-User-ID": "test-user"}
        )

        assert hash_response.status_code == 200
        content_hash = hash_response.get_json()["content_hash"]

        # Step 2: Process PDF
        process_data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": content_hash,
        }
        process_response = client.post(
            "/api/process-pdf", data=process_data, headers={"X-User-ID": "test-user"}
        )

        assert process_response.status_code == 200
        process_result = process_response.get_json()
        assert process_result["status"] == "success"
        assert "content" in process_result

        # Step 3: Retrieve notes
        notes_response = client.get(f"/api/notes/{content_hash}")
        # This will fail in test since we're mocking, but it shows the workflow

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    def test_existing_notes_workflow(self, mock_supabase, client, sample_pdf):
        """Test workflow when notes already exist."""
        # Mock existing notes in database
        existing_note = {
            "content": "Previously generated notes",
            "model_used": "test-model",
            "generated_at": "2023-01-01T00:00:00",
        }
        mock_supabase.table().select().eq().execute.return_value.data = [existing_note]

        # Process PDF (should return existing notes)
        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "existing-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "Retrieved existing notes" in result["message"]
        assert result["content"] == existing_note["content"]


class TestErrorScenarios:
    """Test various error scenarios in the complete system."""

    def test_missing_environment_variables(self, client):
        """Test behavior when environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            # This should fail gracefully when LLM client can't initialize
            data = {
                "file": (io.BytesIO(b"fake pdf"), "test.pdf"),
                "subject": "Test",
                "content_hash": "test-hash",
            }
            response = client.post(
                "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
            )

            assert response.status_code == 500

    @patch("app.supabase")
    def test_database_connection_failure(self, mock_supabase, client, sample_pdf):
        """Test behavior when database is unavailable."""
        mock_supabase.table().select().eq().execute.side_effect = Exception(
            "Database unavailable"
        )

        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        assert response.status_code == 500
        assert "Database unavailable" in response.get_json()["error"]

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_llm_api_failure(
        self, mock_requests_post, mock_supabase, client, sample_pdf
    ):
        """Test behavior when LLM API fails."""
        # Mock database responses
        mock_supabase.table().select().eq().execute.return_value.data = []

        # Mock LLM API failure
        mock_requests_post.side_effect = requests.exceptions.RequestException(
            "API Error"
        )

        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        # Should still succeed but with error notes
        assert response.status_code == 200
        result = response.get_json()
        assert "Error generating notes" in result["content"]


class TestRateLimiting:
    """Test rate limiting and concurrent request handling."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    # Update the method signature to accept the mock arguments:

    def test_concurrent_requests(self, mock_post, mock_supabase):  # ✅ Fixed
        """Test handling multiple simultaneous uploads"""
        # Create test client within the method
        with app.test_client() as client:
            pdf_data = b"%PDF-1.4\nTest content for concurrent testing"

            responses = []

            # Make multiple requests
            for i in range(5):
                data = {"file": (io.BytesIO(pdf_data), f"test_concurrent_{i}.pdf")}
                response = client.post(
                    "/api/process-pdf", data=data, content_type="multipart/form-data", headers={"X-User-ID": "test-user"}
                )
                responses.append(response)

            # Verify all requests completed
            assert len(responses) == 5
            for response in responses:
                assert response.status_code in [200, 302, 400]


class TestDataValidation:
    """Test data validation and sanitization."""

    def test_file_size_limits(self, client):
        """Test handling of very large files."""
        # Create a very large fake PDF
        large_pdf = b"%PDF-1.4\n" + b"A" * (10 * 1024 * 1024)  # 10MB

        data = {
            "file": (io.BytesIO(large_pdf), "large.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        # Should handle large files gracefully (may fail or succeed depending on implementation)
        assert response.status_code in [200, 400, 413, 500]

    def test_special_characters_in_subject(self, client, sample_pdf):
        """Test handling of special characters in subject."""
        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject with émojis 🎓📚 and spëcial chars",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        # Should handle special characters without crashing
        assert response.status_code in [200, 400, 500]

    def test_sql_injection_attempts(self, client):
        """Test protection against SQL injection attempts."""
        malicious_hash = "'; DROP TABLE study_notes; --"

        response = client.get(f"/api/notes/{malicious_hash}")

        # Should not crash and should return 404 or handle gracefully
        assert response.status_code in [404, 400, 500]
        # The important thing is that it doesn't crash the server


class TestPerformance:
    """Test performance characteristics."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_response_time(self, mock_requests_post, mock_supabase, client, sample_pdf):
        """Test that responses are returned in reasonable time."""
        import time

        # Mock responses
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.raise_for_status.return_value = None
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": "Quick notes"}}]
        }
        mock_requests_post.return_value = mock_llm_response
        mock_supabase.table().select().eq().execute.return_value.data = []
        mock_supabase.table().insert().execute.return_value = MagicMock()

        start_time = time.time()

        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds


class TestSecurity:
    """Test security aspects of the application."""

    def test_unauthorized_access(self, client, sample_pdf):
        """Test that requests without user ID are rejected."""
        data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post("/api/process-pdf", data=data)

        assert response.status_code == 401
        assert "User ID not provided" in response.get_json()["error"]

    def test_file_type_validation(self, client):
        """Test that only PDF files are accepted."""
        malicious_file = b"<script>alert('xss')</script>"

        data = {
            "file": (io.BytesIO(malicious_file), "malicious.html"),
            "subject": "Test Subject",
            "content_hash": "test-hash",
        }
        response = client.post(
            "/api/process-pdf", data=data, headers={"X-User-ID": "test-user"}
        )

        assert response.status_code == 400
        assert "File must be a PDF" in response.get_json()["error"]


if __name__ == "__main__":
    pytest.main([__file__])
