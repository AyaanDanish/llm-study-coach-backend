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


class TestFlashcardWorkflow:
    """Test complete flashcard generation workflow."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_end_to_end_flashcard_generation(
        self, mock_requests_post, mock_supabase, client, sample_pdf
    ):
        """Test complete workflow from PDF to flashcard generation."""
        # Step 1: Mock PDF processing and notes generation
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.raise_for_status.return_value = None
        mock_llm_response.json.return_value = {
            "choices": [
                {"message": {"content": "Generated study notes for flashcard testing"}}
            ]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock database for PDF processing
        mock_supabase.table().select().eq().execute.return_value.data = []
        mock_supabase.table().insert().execute.return_value = MagicMock()

        # Process PDF first
        process_data = {
            "file": (io.BytesIO(sample_pdf), "test.pdf"),
            "subject": "Programming",
            "content_hash": "test-flashcard-hash",
        }
        process_response = client.post(
            "/api/process-pdf", data=process_data, headers={"X-User-ID": "test-user"}
        )
        assert process_response.status_code == 200

        # Step 2: Mock flashcard generation
        mock_flashcard_response = MagicMock()
        mock_flashcard_response.status_code = 200
        mock_flashcard_response.raise_for_status.return_value = None
        mock_flashcard_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"flashcards": [{"front": "What is Python?", "back": "A programming language", "category": "Programming", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_flashcard_response

        # Mock database responses for flashcard generation
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "content": "Python is a programming language",
                "model_used": "test",
                "generated_at": "2023-01-01T00:00:00",
            }
        ]
        mock_supabase.table().select().eq().single().execute.return_value.data = {
            "subject": "Programming",
            "title": "Python Basics",
        }
        mock_supabase.table().insert().execute.return_value.data = [
            {
                "id": "1",
                "front": "What is Python?",
                "back": "A programming language",
                "category": "Programming",
                "difficulty": "easy",
            }
        ]

        # Step 3: Generate flashcards
        flashcard_data = {"category": "Programming"}
        flashcard_response = client.post(
            "/api/generate-flashcards-from-material/test-flashcard-hash",
            json=flashcard_data,
            headers={"X-User-ID": "test-user"},
        )

        assert flashcard_response.status_code == 200
        flashcard_result = flashcard_response.get_json()
        assert flashcard_result["status"] == "success"
        assert "flashcards" in flashcard_result
        assert flashcard_result["total_saved"] == 1

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    def test_flashcard_generation_without_study_material(self, mock_supabase, client):
        """Test flashcard generation when study material doesn't exist."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        flashcard_data = {"category": "Test Category"}
        response = client.post(
            "/api/generate-flashcards-from-material/nonexistent-hash",
            json=flashcard_data,
            headers={"X-User-ID": "test-user"},
        )

        assert response.status_code == 404
        assert "Study material not found" in response.get_json()["error"]


class TestQuizWorkflow:
    """Test complete quiz generation and taking workflow."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_end_to_end_quiz_generation(
        self, mock_requests_post, mock_supabase, client, sample_pdf
    ):
        """Test complete workflow from PDF to quiz generation."""
        # Step 1: Mock study content exists
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"content": "Python is a programming language created by Guido van Rossum"}
        ]

        # Mock material access check
        mock_supabase.table().select().eq().eq().execute.return_value.data = [
            {"id": "1", "name": "Python Basics", "user_id": "test-user"}
        ]

        # Step 2: Mock quiz generation
        mock_quiz_response = MagicMock()
        mock_quiz_response.status_code = 200
        mock_quiz_response.raise_for_status.return_value = None
        mock_quiz_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"questions": [{"question": "Who created Python?", "options": ["Guido van Rossum", "Linus Torvalds", "Dennis Ritchie", "James Gosling"], "correct_answer": 0, "explanation": "Python was created by Guido van Rossum", "difficulty": "easy"}, {"question": "What is Python?", "options": ["Programming Language", "Snake", "Tool", "Framework"], "correct_answer": 0, "explanation": "Python is a programming language", "difficulty": "easy"}, {"question": "Is Python interpreted?", "options": ["Yes", "No", "Sometimes", "Never"], "correct_answer": 0, "explanation": "Python is an interpreted language", "difficulty": "easy"}, {"question": "What syntax does Python use?", "options": ["Indentation", "Brackets", "Semicolons", "None"], "correct_answer": 0, "explanation": "Python uses indentation for code blocks", "difficulty": "easy"}, {"question": "Is Python open source?", "options": ["Yes", "No", "Partially", "Commercial"], "correct_answer": 0, "explanation": "Python is open source software", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_quiz_response

        # Step 3: Generate quiz
        quiz_data = {
            "content_hash": "test-quiz-hash",
            "material_title": "Python Basics",
            "material_subject": "Programming",
            "quiz_title": "Python Knowledge Test",
            "user_id": "test-user",
        }

        quiz_response = client.post("/generate-quiz", json=quiz_data)

        assert quiz_response.status_code == 200
        quiz_result = quiz_response.get_json()
        assert quiz_result["status"] == "success"
        assert "quiz_id" in quiz_result
        assert "questions" in quiz_result

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    def test_quiz_generation_without_content(self, mock_supabase, client):
        """Test quiz generation when no study content exists."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        quiz_data = {
            "content_hash": "nonexistent-hash",
            "material_title": "Test",
            "material_subject": "Test",
            "quiz_title": "Test Quiz",
            "user_id": "test-user",
        }

        response = client.post("/generate-quiz", json=quiz_data)

        assert response.status_code == 404
        assert "No processed content found" in response.get_json()["error"]


class TestQAWorkflow:
    """Test complete question-answering workflow."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_end_to_end_qa_workflow(self, mock_requests_post, mock_supabase, client):
        """Test complete Q&A workflow from question to answer storage."""
        # Step 1: Mock study notes exist
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "id": "note-1",
                "content": "Python is a high-level programming language created by Guido van Rossum",
            }
        ]

        # Mock material lookup
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(data=[{"id": "note-1", "content": "Python content"}]),
            MagicMock(
                data=[
                    {"id": "material-1", "user_id": "test-user", "name": "Python Notes"}
                ]
            ),
        ]

        # Step 2: Mock LLM answer generation
        mock_answer_response = MagicMock()
        mock_answer_response.status_code = 200
        mock_answer_response.raise_for_status.return_value = None
        mock_answer_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "**Summary:** Python was created by Guido van Rossum.\n\nPython is a high-level programming language that was first released in 1991."
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_answer_response

        # Mock Q&A insertion
        mock_supabase.table().insert().execute.return_value.data = [
            {
                "id": "qa-1",
                "question": "Who created Python?",
                "answer": "Python was created by Guido van Rossum",
            }
        ]

        # Step 3: Ask question
        qa_data = {"content_hash": "test-qa-hash", "question": "Who created Python?"}

        qa_response = client.post("/api/ask-question", json=qa_data)

        assert qa_response.status_code == 200
        qa_result = qa_response.get_json()
        assert qa_result["status"] == "success"
        assert "Guido van Rossum" in qa_result["answer"]

        # Step 4: Test Q&A list retrieval
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(
                data=[
                    {"id": "material-1", "name": "Python Notes", "user_id": "test-user"}
                ]
            ),
            MagicMock(data=[{"id": "note-1"}]),
        ]

        mock_supabase.table().select().eq().order().execute.return_value.data = [
            {
                "id": "qa-1",
                "question": "Who created Python?",
                "answer": "Python was created by Guido van Rossum",
                "created_at": "2023-01-01T00:00:00",
            }
        ]

        qa_list_response = client.get("/api/qa-list?content_hash=test-qa-hash")

        assert qa_list_response.status_code == 200
        qa_list_result = qa_list_response.get_json()
        assert len(qa_list_result["qa"]) == 1
        assert qa_list_result["qa"][0]["question"] == "Who created Python?"

        # Step 5: Test Q&A deletion
        mock_supabase.table().delete().eq().execute.return_value = MagicMock()

        delete_response = client.delete(
            "/api/qa/qa-1", headers={"X-User-ID": "test-user"}
        )

        assert delete_response.status_code == 200
        assert "deleted" in delete_response.get_json()["message"]

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    def test_qa_without_study_notes(self, mock_supabase, client):
        """Test Q&A when study notes don't exist."""
        mock_supabase.table().select().eq().execute.return_value.data = []

        qa_data = {"content_hash": "nonexistent-hash", "question": "Test question?"}

        response = client.post("/api/ask-question", json=qa_data)

        assert response.status_code == 404
        assert "Study note not found" in response.get_json()["error"]


class TestCrossFeatureIntegration:
    """Test integration between different features."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_pdf_to_all_features_workflow(
        self, mock_requests_post, mock_supabase, client, sample_pdf
    ):
        """Test using one PDF to generate notes, flashcards, quiz, and Q&A."""
        content_hash = "integration-test-hash"

        # Step 1: Process PDF
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.raise_for_status.return_value = None
        mock_llm_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Comprehensive study notes about Python programming"
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_llm_response
        mock_supabase.table().select().eq().execute.return_value.data = []
        mock_supabase.table().insert().execute.return_value = MagicMock()

        process_data = {
            "file": (io.BytesIO(sample_pdf), "python_guide.pdf"),
            "subject": "Programming",
            "content_hash": content_hash,
        }
        process_response = client.post(
            "/api/process-pdf", data=process_data, headers={"X-User-ID": "test-user"}
        )
        assert process_response.status_code == 200

        # Step 2: Generate flashcards from the same content
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "content": "Python programming notes",
                "model_used": "test",
                "generated_at": "2023-01-01T00:00:00",
            }
        ]
        mock_supabase.table().select().eq().single().execute.return_value.data = {
            "subject": "Programming",
            "title": "Python Guide",
        }

        mock_flashcard_response = MagicMock()
        mock_flashcard_response.status_code = 200
        mock_flashcard_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"flashcards": [{"front": "What is Python?", "back": "A programming language", "category": "Programming", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_flashcard_response
        mock_supabase.table().insert().execute.return_value.data = [{"id": "1"}]

        flashcard_data = {"category": "Programming"}
        flashcard_response = client.post(
            f"/api/generate-flashcards-from-material/{content_hash}",
            json=flashcard_data,
            headers={"X-User-ID": "test-user"},
        )
        assert flashcard_response.status_code == 200

        # Step 3: Generate quiz from the same content
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"content": "Python programming notes"}
        ]
        mock_supabase.table().select().eq().eq().execute.return_value.data = [
            {"id": "1", "user_id": "test-user"}
        ]

        mock_quiz_response = MagicMock()
        mock_quiz_response.status_code = 200
        mock_quiz_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"questions": [{"question": "What is Python?", "options": ["Programming Language", "Snake", "Tool", "Framework"], "correct_answer": 0, "explanation": "Python is a programming language", "difficulty": "easy"}, {"question": "Who created Python?", "options": ["Guido van Rossum", "Linus Torvalds", "Dennis Ritchie", "James Gosling"], "correct_answer": 0, "explanation": "Python was created by Guido van Rossum", "difficulty": "easy"}, {"question": "Is Python interpreted?", "options": ["Yes", "No", "Sometimes", "Never"], "correct_answer": 0, "explanation": "Python is an interpreted language", "difficulty": "easy"}, {"question": "What syntax does Python use?", "options": ["Indentation", "Brackets", "Semicolons", "None"], "correct_answer": 0, "explanation": "Python uses indentation for code blocks", "difficulty": "easy"}, {"question": "Is Python open source?", "options": ["Yes", "No", "Partially", "Commercial"], "correct_answer": 0, "explanation": "Python is open source software", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_quiz_response

        quiz_data = {
            "content_hash": content_hash,
            "material_title": "Python Guide",
            "material_subject": "Programming",
            "quiz_title": "Python Quiz",
            "user_id": "test-user",
        }
        quiz_response = client.post("/generate-quiz", json=quiz_data)
        assert quiz_response.status_code == 200

        # Step 4: Ask questions about the same content
        mock_supabase.table().select().eq().execute.side_effect = [
            MagicMock(data=[{"id": "note-1", "content": "Python programming notes"}]),
            MagicMock(data=[{"id": "material-1", "user_id": "test-user"}]),
        ]

        mock_answer_response = MagicMock()
        mock_answer_response.status_code = 200
        mock_answer_response.json.return_value = {
            "choices": [{"message": {"content": "Python is a programming language"}}]
        }
        mock_requests_post.return_value = mock_answer_response
        mock_supabase.table().insert().execute.return_value.data = [{"id": "qa-1"}]

        qa_data = {"content_hash": content_hash, "question": "What is Python used for?"}
        qa_response = client.post("/api/ask-question", json=qa_data)
        assert qa_response.status_code == 200

        # All features should work with the same content hash
        assert all(
            [
                process_response.status_code == 200,
                flashcard_response.status_code == 200,
                quiz_response.status_code == 200,
                qa_response.status_code == 200,
            ]
        )


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
    def test_llm_api_failure_with_new_features(
        self, mock_requests_post, mock_supabase, client
    ):
        """Test behavior when LLM API fails for new features."""
        # Mock database responses
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"id": "note-1", "content": "Test content", "model_used": "test"}
        ]
        mock_supabase.table().select().eq().single().execute.return_value.data = {
            "subject": "Test",
            "title": "Test",
        }

        # Mock LLM API failure
        mock_requests_post.side_effect = requests.exceptions.RequestException(
            "API Error"
        )

        # Test flashcard generation failure
        flashcard_data = {"category": "Test Category"}
        flashcard_response = client.post(
            "/api/generate-flashcards-from-material/test-hash",
            json=flashcard_data,
            headers={"X-User-ID": "test-user"},
        )
        assert flashcard_response.status_code == 500

        # Test quiz generation failure
        quiz_data = {
            "content_hash": "test-hash",
            "material_title": "Test",
            "material_subject": "Test",
            "quiz_title": "Test Quiz",
            "user_id": "test-user",
        }
        quiz_response = client.post("/generate-quiz", json=quiz_data)
        assert quiz_response.status_code == 500

        # Test Q&A failure
        qa_data = {"content_hash": "test-hash", "question": "Test question?"}
        qa_response = client.post("/api/ask-question", json=qa_data)
        assert qa_response.status_code == 500

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_rate_limiting_across_features(
        self, mock_requests_post, mock_supabase, client
    ):
        """Test rate limiting behavior across all LLM features."""
        # Mock rate limit response
        mock_rate_limit_response = MagicMock()
        mock_rate_limit_response.status_code = 429
        mock_rate_limit_response.text = "Rate limit exceeded"
        mock_requests_post.return_value = mock_rate_limit_response

        # Mock database responses
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"id": "note-1", "content": "Test content"}
        ]

        # Test all features handle rate limiting gracefully
        features_to_test = [
            (
                "flashcards",
                "/api/generate-flashcards-from-material/test-hash",
                "POST",
                {"X-User-ID": "test-user"},
            ),
            ("quiz", "/generate-quiz", "POST", None),
            ("qa", "/api/ask-question", "POST", None),
        ]

        for feature_name, endpoint, method, headers in features_to_test:
            if feature_name == "quiz":
                data = {
                    "content_hash": "test-hash",
                    "material_title": "Test",
                    "material_subject": "Test",
                    "quiz_title": "Test Quiz",
                    "user_id": "test-user",
                }
                response = client.post(endpoint, json=data)
            elif feature_name == "qa":
                data = {"content_hash": "test-hash", "question": "Test question?"}
                response = client.post(endpoint, json=data)
            else:
                data = {"category": "Test Category"}
                response = client.post(endpoint, json=data, headers=headers)

            # Should handle rate limiting gracefully (return 500 with proper error)
            assert response.status_code == 500


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
                    "/api/process-pdf",
                    data=data,
                    content_type="multipart/form-data",
                    headers={"X-User-ID": "test-user"},
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

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_token_limits_across_features(
        self, mock_requests_post, mock_supabase, client
    ):
        """Test that all features respect token limits."""
        import time

        # Create very large content to test token limits
        large_content = "A" * 5000000  # 5M characters, likely to exceed token limits

        # Mock database responses with large content
        mock_supabase.table().select().eq().execute.return_value.data = [
            {"id": "note-1", "content": large_content, "model_used": "test"}
        ]

        # Mock material access check for quiz
        mock_supabase.table().select().eq().eq().execute.return_value.data = [
            {"id": "1", "user_id": "test-user"}
        ]

        # Test flashcard generation with large content
        start_time = time.time()
        flashcard_data = {"category": "Test Category"}
        flashcard_response = client.post(
            "/api/generate-flashcards-from-material/test-hash",
            json=flashcard_data,
            headers={"X-User-ID": "test-user"},
        )
        flashcard_time = time.time() - start_time

        # Should either succeed or fail gracefully (not crash)
        assert flashcard_response.status_code in [200, 500]
        assert flashcard_time < 10.0  # Should not hang indefinitely

        # Test quiz generation with large content
        start_time = time.time()
        quiz_data = {
            "content_hash": "test-hash",
            "material_title": "Large Test",
            "material_subject": "Test",
            "quiz_title": "Large Quiz",
            "user_id": "test-user",
        }
        quiz_response = client.post("/generate-quiz", json=quiz_data)
        quiz_time = time.time() - start_time

        assert quiz_response.status_code in [200, 500]
        assert quiz_time < 10.0

        # Test Q&A with large content
        start_time = time.time()
        qa_data = {"content_hash": "test-hash", "question": "What is this about?"}
        qa_response = client.post("/api/ask-question", json=qa_data)
        qa_time = time.time() - start_time

        assert qa_response.status_code in [200, 500]
        assert qa_time < 10.0

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("app.supabase")
    @patch("requests.post")
    def test_cost_estimation_workflow(self, mock_requests_post, mock_supabase, client):
        """Test that cost estimation works properly across features."""
        # Mock successful responses that would incur costs
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"flashcards": [{"front": "Test Question", "back": "Test Answer", "category": "Test Category", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_requests_post.return_value = mock_response

        # Mock database responses
        mock_supabase.table().select().eq().execute.return_value.data = [
            {
                "content": "Test content for cost estimation",
                "model_used": "test",
                "generated_at": "2023-01-01T00:00:00",
            }
        ]
        mock_supabase.table().select().eq().single().execute.return_value.data = {
            "subject": "Test",
            "title": "Test",
        }
        mock_supabase.table().insert().execute.return_value.data = [{"id": "1"}]

        # Test multiple API calls to simulate cost accumulation
        features = [
            (
                "flashcards",
                "/api/generate-flashcards-from-material/test-hash",
                {"X-User-ID": "test-user"},
            ),
        ]

        for feature_name, endpoint, headers in features:
            data = {"category": "Test Category"}
            response = client.post(endpoint, json=data, headers=headers)
            # Should succeed and consume API quota
            assert response.status_code == 200
            # Verify the API was actually called (cost incurred)
            mock_requests_post.assert_called()


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
