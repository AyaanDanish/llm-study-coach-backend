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


class TestGenerateFlashcardsEndpoint:
    """Test the /api/generate-flashcards-from-material/<content_hash> endpoint."""

    @patch("app.supabase")
    @patch("app.llm_client.generate_flashcards")
    def test_generate_flashcards_success(
        self, mock_generate_flashcards, mock_supabase, client, headers
    ):
        """Test successful flashcard generation."""
        # Mock existing study material
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "content": "Test content",
                "model_used": "test-model",
                "generated_at": "2023-01-01",
            }
        ]

        # Mock material info lookup
        mock_supabase.table().select().eq().single().execute.return_value.data = {
            "subject": "Test Subject",
            "title": "Test Material",
        }

        # Mock flashcard generation
        mock_generate_flashcards.return_value = [
            {
                "front": "What is Python?",
                "back": "A programming language",
                "category": "Programming",
                "difficulty": "easy",
            }
        ]

        # Mock database insertion
        mock_supabase.table().insert().execute.return_value.data = [
            {
                "id": "1",
                "front": "What is Python?",
                "back": "A programming language",
                "category": "Programming",
                "difficulty": "easy",
            }
        ]

        data = {"category": "Programming"}
        response = client.post(
            "/api/generate-flashcards-from-material/test-hash",
            json=data,
            headers=headers,
        )

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "flashcards" in result
        assert result["total_saved"] == 1

    @patch("app.supabase")
    def test_generate_flashcards_no_material(self, mock_supabase, client, headers):
        """Test flashcard generation when study material doesn't exist."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        data = {"category": "Test Category"}
        response = client.post(
            "/api/generate-flashcards-from-material/nonexistent-hash",
            json=data,
            headers=headers,
        )

        assert response.status_code == 404
        assert "Study material not found" in response.get_json()["error"]

    def test_generate_flashcards_no_user_id(self, client):
        """Test flashcard generation without user ID."""
        response = client.post("/api/generate-flashcards-from-material/test-hash")

        assert response.status_code == 401
        assert "User ID not provided" in response.get_json()["error"]

    @patch("app.supabase")
    @patch("app.llm_client.generate_flashcards")
    def test_generate_flashcards_llm_failure(
        self, mock_generate_flashcards, mock_supabase, client, headers
    ):
        """Test flashcard generation when LLM fails."""
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"content": "Test content"}
        ]
        mock_generate_flashcards.return_value = None

        data = {"category": "Test Category"}
        response = client.post(
            "/api/generate-flashcards-from-material/test-hash",
            json=data,
            headers=headers,
        )

        assert response.status_code == 500
        assert "Failed to generate flashcards" in response.get_json()["error"]


class TestGenerateQuizEndpoint:
    """Test the /generate-quiz endpoint."""

    @patch("app.supabase")
    @patch("app.llm_client.generate_quiz")
    def test_generate_quiz_success(self, mock_generate_quiz, mock_supabase, client):
        """Test successful quiz generation."""
        # Mock study notes lookup
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"content": "Test study content"}
        ]

        # Mock material access check
        mock_supabase.table().select().eq().eq().execute.return_value.data = [
            {"id": "1", "name": "Test Material", "user_id": "test-user"}
        ]

        # Mock quiz generation
        mock_generate_quiz.return_value = [
            {
                "id": "q_1",
                "question": "What is Python?",
                "options": ["Language", "Snake", "Tool", "Framework"],
                "correct_answer": 0,
                "explanation": "Python is a programming language",
                "difficulty": "easy",
            }
        ]

        data = {
            "content_hash": "test-hash",
            "material_title": "Test Material",
            "material_subject": "Programming",
            "quiz_title": "Python Quiz",
            "user_id": "test-user",
        }

        response = client.post("/generate-quiz", json=data)

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "quiz_id" in result
        assert "questions" in result
        assert len(result["questions"]) == 1

    def test_generate_quiz_missing_fields(self, client):
        """Test quiz generation with missing required fields."""
        data = {
            "content_hash": "test-hash",
            # Missing other required fields
        }

        response = client.post("/generate-quiz", json=data)

        assert response.status_code == 400
        assert "Missing required fields" in response.get_json()["error"]

    @patch("app.supabase")
    def test_generate_quiz_no_content(self, mock_supabase, client):
        """Test quiz generation when no study content exists."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        data = {
            "content_hash": "nonexistent-hash",
            "material_title": "Test",
            "material_subject": "Test",
            "quiz_title": "Test Quiz",
            "user_id": "test-user",
        }

        response = client.post("/generate-quiz", json=data)

        assert response.status_code == 404
        assert "No processed content found" in response.get_json()["error"]

    @patch("app.supabase")
    @patch("app.llm_client.generate_quiz")
    def test_generate_quiz_llm_failure(self, mock_generate_quiz, mock_supabase, client):
        """Test quiz generation when LLM fails."""
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"content": "Test content"}
        ]
        mock_supabase.table().select().eq().eq().execute.return_value.data = [
            {"id": "1", "user_id": "test-user"}
        ]
        mock_generate_quiz.return_value = None

        data = {
            "content_hash": "test-hash",
            "material_title": "Test",
            "material_subject": "Test",
            "quiz_title": "Test Quiz",
            "user_id": "test-user",
        }

        response = client.post("/generate-quiz", json=data)

        assert response.status_code == 500
        assert "Failed to generate quiz questions" in response.get_json()["error"]


class TestAskQuestionEndpoint:
    """Test the /api/ask-question endpoint."""

    @patch("app.supabase")
    @patch("app.llm_client.answer_question")
    def test_ask_question_success(self, mock_answer_question, mock_supabase, client):
        """Test successful question answering."""
        # Mock study notes lookup - need to handle multiple calls to table().select().eq().execute()
        notes_data = [{"id": "note-1", "content": "Python is a programming language"}]
        material_data = [
            {"id": "material-1", "user_id": "test-user", "name": "Python Notes"}
        ]

        # Configure the mock to return different data for different calls
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(data=notes_data),  # First call for study notes
            MagicMock(data=material_data),  # Second call for material lookup
        ]

        # Mock LLM response
        mock_answer_question.return_value = (
            "**Summary:** Python is a high-level programming language."
        )

        # Mock Q&A insertion
        mock_supabase.table().insert().execute.return_value.data = [
            {
                "id": "qa-1",
                "question": "What is Python?",
                "answer": "Python is a programming language",
            }
        ]

        data = {"content_hash": "test-hash", "question": "What is Python?"}

        response = client.post("/api/ask-question", json=data)

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "Python is a high-level programming language" in result["answer"]

    def test_ask_question_missing_data(self, client):
        """Test question answering with missing data."""
        response = client.post("/api/ask-question", json={})

        assert response.status_code == 400
        assert "No data provided" in response.get_json()["error"]

    def test_ask_question_missing_fields(self, client):
        """Test question answering with missing required fields."""
        data = {"content_hash": "test-hash"}  # Missing question

        response = client.post("/api/ask-question", json=data)

        assert response.status_code == 400
        assert "content_hash and question are required" in response.get_json()["error"]

    @patch("app.supabase")
    def test_ask_question_no_notes(self, mock_supabase, client):
        """Test question answering when notes don't exist."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        data = {"content_hash": "nonexistent-hash", "question": "What is Python?"}

        response = client.post("/api/ask-question", json=data)

        assert response.status_code == 404
        assert "Study note not found" in response.get_json()["error"]

    @patch("app.supabase")
    @patch("app.llm_client.answer_question")
    def test_ask_question_llm_failure(
        self, mock_answer_question, mock_supabase, client
    ):
        """Test question answering when LLM fails."""
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"id": "note-1", "content": "Test content"}
        ]
        mock_answer_question.return_value = None

        data = {"content_hash": "test-hash", "question": "Test question?"}

        response = client.post("/api/ask-question", json=data)

        assert response.status_code == 500
        assert "Failed to generate answer from LLM" in response.get_json()["error"]


class TestQAListEndpoint:
    """Test the /api/qa-list endpoint."""

    @patch("app.supabase")
    def test_qa_list_success(self, mock_supabase, client):
        """Test successful Q&A list retrieval."""
        # Mock material lookup
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"id": "material-1", "name": "Test Material", "user_id": "test-user"}
        ]

        # Mock Q&A sessions lookup
        mock_supabase.table().select().eq().order().execute.return_value.data = [
            {
                "id": "qa-1",
                "question": "What is Python?",
                "answer": "A programming language",
                "created_at": "2023-01-01T00:00:00",
            }
        ]

        # Mock note lookup (second call)
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(
                data=[
                    {
                        "id": "material-1",
                        "name": "Test Material",
                        "user_id": "test-user",
                    }
                ]
            ),
            MagicMock(data=[{"id": "note-1"}]),
        ]

        response = client.get("/api/qa-list?content_hash=test-hash")

        assert response.status_code == 200
        result = response.get_json()
        assert "qa" in result
        assert len(result["qa"]) == 1
        assert result["qa"][0]["question"] == "What is Python?"

    def test_qa_list_missing_content_hash(self, client):
        """Test Q&A list retrieval without content hash."""
        response = client.get("/api/qa-list")

        assert response.status_code == 400
        assert "content_hash is required" in response.get_json()["error"]

    @patch("app.supabase")
    def test_qa_list_empty(self, mock_supabase, client):
        """Test Q&A list retrieval with no Q&A sessions."""
        # Mock empty material lookup
        mock_supabase.table().select().eq().execute.return_value.data = []

        response = client.get("/api/qa-list?content_hash=test-hash")

        assert response.status_code == 200
        result = response.get_json()
        assert result["qa"] == []


class TestDeleteQAEndpoint:
    """Test the /api/qa/<qa_id> DELETE endpoint."""

    @patch("app.supabase")
    def test_delete_qa_success(self, mock_supabase, client, headers):
        """Test successful Q&A deletion."""
        mock_supabase.table().delete().eq().execute.return_value = MagicMock()

        response = client.delete("/api/qa/test-qa-id", headers=headers)

        assert response.status_code == 200
        result = response.get_json()
        assert result["status"] == "success"
        assert "deleted" in result["message"]

    def test_delete_qa_no_user_id(self, client):
        """Test Q&A deletion without user ID."""
        response = client.delete("/api/qa/test-qa-id")

        assert response.status_code == 401
        assert "User ID not provided" in response.get_json()["error"]

    @patch("app.supabase")
    def test_delete_qa_database_error(self, mock_supabase, client, headers):
        """Test Q&A deletion with database error."""
        mock_supabase.table().delete().eq().execute.side_effect = Exception(
            "Database error"
        )

        response = client.delete("/api/qa/test-qa-id", headers=headers)

        assert response.status_code == 500
        assert "Database error" in response.get_json()["error"]


class TestDebugEndpoints:
    """Test debug endpoints."""

    @patch("app.supabase")
    def test_debug_material_exists(self, mock_supabase, client):
        """Test debug material endpoint with existing material."""
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "id": "test-id",
                "name": "Test Material",
                "subject": "Programming",
                "user_id": "test-user",
                "content_hash": "test-hash",
                "uploaded_at": "2023-01-01",
            }
        ]

        # Mock notes lookup
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(
                data=[
                    {
                        "id": "test-id",
                        "name": "Test Material",
                        "subject": "Programming",
                        "user_id": "test-user",
                        "content_hash": "test-hash",
                        "uploaded_at": "2023-01-01",
                    }
                ]
            ),
            MagicMock(
                data=[
                    {
                        "content_hash": "test-hash",
                        "content": "Test content",
                        "generated_at": "2023-01-01",
                        "model_used": "test-model",
                    }
                ]
            ),
        ]

        response = client.get("/debug-material/test-id")

        assert response.status_code == 200
        result = response.get_json()
        assert result["material_found"] is True
        assert result["material_data"]["name"] == "Test Material"
        assert result["notes_found"] is True

    @patch("app.supabase")
    def test_debug_material_not_found(self, mock_supabase, client):
        """Test debug material endpoint with non-existent material."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        response = client.get("/debug-material/nonexistent-id")

        assert response.status_code == 200
        result = response.get_json()
        assert result["material_found"] is False
        assert result["material_count"] == 0

    @patch("app.supabase")
    def test_debug_content_exists(self, mock_supabase, client):
        """Test debug content endpoint with existing content."""
        # Mock notes lookup
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(
                data=[
                    {
                        "content_hash": "test-hash",
                        "content": "Test content",
                        "generated_at": "2023-01-01",
                        "model_used": "test-model",
                    }
                ]
            ),
            MagicMock(
                data=[
                    {
                        "id": "material-1",
                        "name": "Test Material",
                        "subject": "Programming",
                        "user_id": "test-user",
                        "uploaded_at": "2023-01-01",
                    }
                ]
            ),
        ]

        response = client.get("/debug-content/test-hash")

        assert response.status_code == 200
        result = response.get_json()
        assert result["notes_found"] is True
        assert result["materials_found"] is True
        assert result["notes_data"]["content_length"] > 0


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

    @patch("app.supabase")
    def test_generate_flashcards_database_error(self, mock_supabase, client, headers):
        """Test flashcard generation with database error."""
        mock_supabase.table().select().eq().execute.side_effect = Exception(
            "Database error"
        )

        data = {"category": "Test Category"}
        response = client.post(
            "/api/generate-flashcards-from-material/test-hash",
            json=data,
            headers=headers,
        )

        assert response.status_code == 500
        assert "Database error" in response.get_json()["error"]

    @patch("app.supabase")
    def test_generate_quiz_database_error(self, mock_supabase, client):
        """Test quiz generation with database error."""
        mock_supabase.table().select().eq().execute.side_effect = Exception(
            "Database error"
        )

        data = {
            "content_hash": "test-hash",
            "material_title": "Test",
            "material_subject": "Test",
            "quiz_title": "Test Quiz",
            "user_id": "test-user",
        }

        response = client.post("/generate-quiz", json=data)

        assert response.status_code == 500
        assert "Database error" in response.get_json()["error"]


if __name__ == "__main__":
    pytest.main([__file__])
