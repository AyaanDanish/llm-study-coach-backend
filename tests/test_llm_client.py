"""
Test suite for LLM client functionality.
Tests API calls, prompt generation, and error handling.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import requests
from utils.llm_client import LLMClient


class TestLLMClientInitialization:
    """Test LLM client initialization."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_init_with_api_key(self):
        """Test initialization with valid API key."""
        client = LLMClient()
        assert client.api_key == "test-api-key"
        assert "Bearer test-api-key" in client.headers["Authorization"]
        assert client.headers["Content-Type"] == "application/json"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self):
        """Test initialization without API key should raise error."""
        with pytest.raises(
            ValueError, match="OPENROUTER_API_KEY environment variable not set"
        ):
            LLMClient()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": ""})
    def test_init_with_empty_api_key(self):
        """Test initialization with empty API key should raise error."""
        with pytest.raises(
            ValueError, match="OPENROUTER_API_KEY environment variable not set"
        ):
            LLMClient()


class TestPromptTemplate:
    """Test prompt template functionality."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_prompt_template(self):
        """Test prompt template retrieval."""
        client = LLMClient()
        template = client.get_prompt_template()

        assert isinstance(template, str)
        assert "{chunk}" in template
        assert "study notes" in template.lower()
        assert "detailed notes" in template.lower()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_prompt_template_formatting(self):
        """Test that prompt template can be formatted with chunk."""
        client = LLMClient()
        template = client.get_prompt_template()
        test_chunk = "This is a test chunk of text."

        formatted_prompt = template.format(chunk=test_chunk)
        assert test_chunk in formatted_prompt
        assert "{chunk}" not in formatted_prompt


class TestGenerateStudyNotes:
    """Test study notes generation."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_study_notes_success(self, mock_post):
        """Test successful notes generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated study notes"}}]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_study_notes("Test chunk content")

        assert result == "Generated study notes"
        mock_post.assert_called_once()

        # Verify the API call parameters
        call_args = mock_post.call_args
        assert call_args[1]["headers"] == client.headers
        assert call_args[1]["json"]["model"] == LLMClient.MODEL
        assert len(call_args[1]["json"]["messages"]) == 1

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_study_notes_api_error(self, mock_post):
        """Test notes generation with API error."""
        # Mock API error response
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        client = LLMClient()
        result = client.generate_study_notes("Test chunk content")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_study_notes_http_error(self, mock_post):
        """Test notes generation with HTTP error."""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "HTTP Error"
        )
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_study_notes("Test chunk content")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_study_notes_invalid_response(self, mock_post):
        """Test notes generation with invalid API response format."""
        # Mock response with missing expected fields
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "response"}
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_study_notes("Test chunk content")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_study_notes_empty_chunk(self):
        """Test notes generation with empty chunk."""
        client = LLMClient()

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Notes for empty content"}}]
            }
            mock_post.return_value = mock_response

            result = client.generate_study_notes("")
            assert result == "Notes for empty content"


class TestGenerateNotesForChunks:
    """Test batch notes generation for multiple chunks."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_notes_for_chunks_success(self):
        """Test successful notes generation for multiple chunks."""
        client = LLMClient()

        with patch.object(client, "generate_study_notes") as mock_generate:
            mock_generate.side_effect = ["Notes 1", "Notes 2", "Notes 3"]

            chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
            result = client.generate_notes_for_chunks(chunks)

            assert len(result) == 3
            assert result == ["Notes 1", "Notes 2", "Notes 3"]
            assert mock_generate.call_count == 3

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_notes_for_chunks_partial_failure(self):
        """Test notes generation with some chunks failing."""
        client = LLMClient()

        with patch.object(client, "generate_study_notes") as mock_generate:
            mock_generate.side_effect = ["Notes 1", None, "Notes 3"]

            chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
            result = client.generate_notes_for_chunks(chunks)

            assert len(result) == 3
            assert result[0] == "Notes 1"
            assert "Error generating notes" in result[1]
            assert result[2] == "Notes 3"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_notes_for_chunks_empty_list(self):
        """Test notes generation for empty chunk list."""
        client = LLMClient()

        result = client.generate_notes_for_chunks([])
        assert result == []

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_notes_for_chunks_all_failures(self):
        """Test notes generation when all chunks fail."""
        client = LLMClient()

        with patch.object(client, "generate_study_notes") as mock_generate:
            mock_generate.return_value = None

            chunks = ["Chunk 1", "Chunk 2"]
            result = client.generate_notes_for_chunks(chunks)

            assert len(result) == 2
            for note in result:
                assert "Error generating notes" in note


class TestAPIIntegration:
    """Test API integration and request formatting."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_api_request_format(self, mock_post):
        """Test that API requests are formatted correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        client.generate_study_notes("Test content")

        # Verify the request was made correctly
        mock_post.assert_called_once_with(
            client.api_url,
            headers=client.headers,
            json={
                "model": LLMClient.MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": client.get_prompt_template().format(
                            chunk="Test content"
                        ),
                    }
                ],
                "max_tokens": 8000,
                "temperature": 0.3,
                "top_p": 0.9,
            },
        )

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_api_timeout_handling(self, mock_post):
        """Test handling of API timeouts."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        client = LLMClient()
        result = client.generate_study_notes("Test content")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_api_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        client = LLMClient()
        result = client.generate_study_notes("Test content")

        assert result is None


class TestFlashcardGeneration:
    """Test flashcard generation functionality."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_flashcard_prompt_template(self):
        """Test flashcard prompt template retrieval."""
        client = LLMClient()
        template = client.get_flashcard_prompt_template()

        assert isinstance(template, str)
        assert "{content}" in template
        assert "flashcards" in template.lower()
        assert "Guidelines for Effective Flashcards" in template

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_flashcards_success(self, mock_post):
        """Test successful flashcard generation."""
        # Mock successful API response with structured JSON
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"flashcards": [{"front": "What is Python?", "back": "A programming language", "category": "Programming", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_flashcards(
            "Python is a programming language", "Programming"
        )

        assert result is not None
        assert len(result) == 1
        assert result[0]["front"] == "What is Python?"
        assert result[0]["back"] == "A programming language"
        assert result[0]["category"] == "Programming"
        assert result[0]["difficulty"] == "easy"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_flashcards_api_error(self, mock_post):
        """Test flashcard generation with API error."""
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        client = LLMClient()
        result = client.generate_flashcards("Test content")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_flashcards_empty_content(self, mock_post):
        """Test flashcard generation with empty content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"flashcards": []}'}}]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_flashcards("")

        assert result is None  # Should return None for empty flashcards array

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_generate_flashcards_none_content(self):
        """Test flashcard generation with None content."""
        client = LLMClient()
        result = client.generate_flashcards(None)

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_flashcards_invalid_json(self, mock_post):
        """Test flashcard generation with invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "invalid json"}}]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_flashcards("Test content")

        assert result is None


class TestQuizGeneration:
    """Test quiz generation functionality."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_quiz_prompt_template(self):
        """Test quiz prompt template retrieval."""
        client = LLMClient()
        template = client.get_quiz_prompt_template()

        assert isinstance(template, str)
        assert "{content}" in template
        assert "{subject}" in template
        assert "{title}" in template
        assert "multiple-choice quiz questions" in template.lower()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_quiz_success(self, mock_post):
        """Test successful quiz generation."""
        # Mock successful API response with structured JSON containing exactly 5 questions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"questions": [{"question": "What is Python?", "options": ["A language", "A snake", "A tool", "A framework"], "correct_answer": 0, "explanation": "Python is a programming language", "difficulty": "easy"}, {"question": "Who created Python?", "options": ["Guido van Rossum", "Mark Zuckerberg", "Bill Gates", "Steve Jobs"], "correct_answer": 0, "explanation": "Python was created by Guido van Rossum", "difficulty": "medium"}, {"question": "What year was Python released?", "options": ["1989", "1991", "1995", "2000"], "correct_answer": 1, "explanation": "Python was first released in 1991", "difficulty": "medium"}, {"question": "What is Python used for?", "options": ["Web development", "Data science", "Automation", "All of the above"], "correct_answer": 3, "explanation": "Python is versatile and used for many purposes", "difficulty": "easy"}, {"question": "Is Python interpreted or compiled?", "options": ["Interpreted", "Compiled", "Both", "Neither"], "correct_answer": 0, "explanation": "Python is primarily an interpreted language", "difficulty": "hard"}]}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_quiz(
            "Python is a programming language", "Programming", "Python Basics"
        )

        assert result is not None
        assert len(result) == 5  # Should have exactly 5 questions
        assert result[0]["question"] == "What is Python?"
        assert len(result[0]["options"]) == 4
        assert result[0]["correct_answer"] == 0
        assert "id" in result[0]  # Should have auto-generated ID

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_quiz_api_error(self, mock_post):
        """Test quiz generation with API error."""
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        client = LLMClient()
        result = client.generate_quiz("Test content", "Subject", "Title")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_quiz_insufficient_questions(self, mock_post):
        """Test quiz generation with insufficient questions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"questions": [{"question": "Test?", "options": ["A", "B", "C", "D"], "correct_answer": 0, "explanation": "Test", "difficulty": "easy"}]}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_quiz("Test content", "Subject", "Title")

        assert result is None  # Should return None if not exactly 5 questions

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_generate_quiz_rate_limit(self, mock_post):
        """Test quiz generation with rate limit error."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.generate_quiz("Test content", "Subject", "Title")

        assert result is None


class TestQuestionAnswering:
    """Test question answering functionality."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_qa_prompt_template(self):
        """Test Q&A prompt template retrieval."""
        client = LLMClient()
        template = client.get_qa_prompt_template()

        assert isinstance(template, str)
        assert "{notes}" in template
        assert "{question}" in template
        assert "markdown formatting" in template.lower()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_answer_question_success(self, mock_post):
        """Test successful question answering."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "**Summary:** Python is a programming language.\n\nPython was created by Guido van Rossum."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.answer_question(
            "Python is a programming language created by Guido van Rossum.",
            "Who created Python?",
        )

        assert result is not None
        assert "Python is a programming language" in result
        assert "Guido van Rossum" in result

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_answer_question_api_error(self, mock_post):
        """Test question answering with API error."""
        mock_post.side_effect = requests.exceptions.RequestException("API Error")

        client = LLMClient()
        result = client.answer_question("Test notes", "Test question?")

        assert result is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_clean_llm_answer(self):
        """Test LLM answer cleaning functionality."""
        client = LLMClient()

        # Test removing repeated summary
        answer_with_repeated_summary = "**Summary:** Test answer.\n\nDetailed explanation.\n\n---\n**In brief:** Test answer."
        cleaned = client.clean_llm_answer(answer_with_repeated_summary)
        assert "In brief" not in cleaned
        assert "---" not in cleaned

        # Test removing extra blank lines
        answer_with_extra_lines = "Line 1\n\n\n\nLine 2"
        cleaned = client.clean_llm_answer(answer_with_extra_lines)
        assert "\n\n\n" not in cleaned

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_answer_question_large_context(self):
        """Test question answering with large context that exceeds token limit."""
        client = LLMClient()

        # Create a very large notes string that would exceed token limits
        large_notes = "A" * (client.MAX_INPUT_TOKENS * 5)  # 5x the limit

        result = client.answer_question(large_notes, "Test question?")

        assert result is None  # Should return None when context is too large


class TestUtilityMethods:
    """Test utility methods and helper functions."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a test string with some words."
        estimated = LLMClient.estimate_tokens(text)

        # Should be roughly 1 token per 4 characters
        expected = len(text) // 4
        assert estimated == expected

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_optimal_chunk_size(self):
        """Test optimal chunk size retrieval."""
        chunk_size = LLMClient.get_optimal_chunk_size()

        assert chunk_size == LLMClient.OPTIMAL_CHUNK_SIZE
        assert chunk_size == 4000000  # 4M characters

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_can_process_entire_document(self):
        """Test document size validation."""
        client = LLMClient()

        # Small document should be processable
        small_doc = 1000  # 1K characters
        assert client.can_process_entire_document(small_doc) is True

        # Very large document should not be processable
        large_doc = 10000000  # 10M characters
        assert client.can_process_entire_document(large_doc) is False

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_get_processing_recommendation(self):
        """Test processing recommendation generation."""
        client = LLMClient()

        # Test small document
        small_doc = 1000
        recommendation = client.get_processing_recommendation(small_doc)

        assert recommendation["strategy"] == "single_call"
        assert recommendation["chunks_needed"] == 1
        assert "estimated_cost" in recommendation
        assert "benefits" in recommendation

        # Test large document
        large_doc = 10000000
        recommendation = client.get_processing_recommendation(large_doc)

        assert recommendation["strategy"] == "chunked"
        assert recommendation["chunks_needed"] > 1
        assert "estimated_cost" in recommendation

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_estimate_cost(self):
        """Test cost estimation."""
        client = LLMClient()

        test_text = "This is a test string for cost estimation."
        cost = client.estimate_cost(test_text, output_tokens=1000)

        assert isinstance(cost, float)
        assert cost > 0

        # Larger text should cost more
        larger_text = test_text * 100
        larger_cost = client.estimate_cost(larger_text, output_tokens=1000)
        assert larger_cost > cost

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_test_api_connection_success(self, mock_post):
        """Test successful API connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.test_api_connection()

        assert result is True

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_test_api_connection_failure(self, mock_post):
        """Test failed API connection test."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        client = LLMClient()
        result = client.test_api_connection()

        assert result is False


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_rate_limit_handling(self, mock_post):
        """Test handling of rate limit errors across all methods."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        client = LLMClient()

        # Test all main methods handle rate limits
        assert client.generate_study_notes("test") is None
        assert client.generate_flashcards("test") is None
        assert client.generate_quiz("test", "subject", "title") is None
        assert client.answer_question("notes", "question") is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_payment_required_handling(self, mock_post):
        """Test handling of payment required errors."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.text = "Payment required"
        mock_post.return_value = mock_response

        client = LLMClient()

        assert client.generate_study_notes("test") is None
        assert client.generate_flashcards("test") is None
        assert client.generate_quiz("test", "subject", "title") is None

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_unauthorized_handling(self, mock_post):
        """Test handling of unauthorized errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        client = LLMClient()

        assert client.generate_study_notes("test") is None
        assert client.generate_flashcards("test") is None
        assert client.generate_quiz("test", "subject", "title") is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_very_long_chunk(self):
        """Test processing very long text chunks."""
        client = LLMClient()
        very_long_chunk = "A" * 10000  # 10KB chunk

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Notes for long content"}}]
            }
            mock_post.return_value = mock_response

            result = client.generate_study_notes(very_long_chunk)
            assert result == "Notes for long content"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_special_characters_in_chunk(self):
        """Test processing chunks with special characters."""
        client = LLMClient()
        special_chunk = "Text with émojis 🎓📚 and spëcial chàracters"

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Notes with special chars"}}]
            }
            mock_post.return_value = mock_response

            result = client.generate_study_notes(special_chunk)
            assert result == "Notes with special chars"

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key"})
    def test_chunk_size_boundary(self):
        """Test processing chunks at the size boundary."""
        client = LLMClient()

        # Test chunk well under token limit (safe boundary)
        # Using 50K characters which is well under the 1M token limit
        boundary_chunk = "A" * 50000  # Safe size that won't exceed limits

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Boundary test result"}}]
            }
            mock_post.return_value = mock_response

            result = client.generate_study_notes(boundary_chunk)
            assert result == "Boundary test result"

        # Test chunk over token limit (using very large size)
        over_limit_chunk = "A" * (client.MAX_INPUT_TOKENS * 5)  # 5x the token limit
        result = client.generate_study_notes(over_limit_chunk)
        assert result is None  # Should return None for chunks over limit


if __name__ == "__main__":
    pytest.main([__file__])
