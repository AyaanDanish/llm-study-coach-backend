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
        assert "section by section" in template.lower()

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


if __name__ == "__main__":
    pytest.main([__file__])
