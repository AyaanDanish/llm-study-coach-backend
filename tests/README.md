# Test Suite Documentation

This directory contains comprehensive tests for the LLM Study Coach backend application.

## Test Files Overview

### test_app.py

Tests for Flask application endpoints and HTTP request handling.

**Test Classes:**

- `TestHealthCheck` - Health check endpoint validation
- `TestProcessPDFEndpoint` - PDF processing and file upload functionality
- `TestGetNotesEndpoint` - Study notes retrieval
- `TestGenerateHashEndpoint` - Content hash generation
- `TestGenerateFlashcardsEndpoint` - Flashcard creation API
- `TestGenerateQuizEndpoint` - Quiz generation API
- `TestAskQuestionEndpoint` - Q&A functionality
- `TestQAListEndpoint` - Q&A session management
- `TestDeleteQAEndpoint` - Q&A deletion operations
- `TestDebugEndpoints` - Debug and utility endpoints
- `TestErrorHandling` - HTTP error response validation

### test_llm_client.py

Tests for LLM client functionality and AI feature integration.

**Test Classes:**

- `TestLLMClientInitialization` - Client setup and configuration
- `TestPromptTemplate` - Prompt template management
- `TestGenerateStudyNotes` - Study notes generation
- `TestGenerateNotesForChunks` - Batch notes processing
- `TestAPIIntegration` - OpenRouter API integration
- `TestFlashcardGeneration` - AI-powered flashcard creation
- `TestQuizGeneration` - Quiz question generation
- `TestQuestionAnswering` - Context-aware Q&A responses
- `TestUtilityMethods` - Token estimation and cost calculation
- `TestErrorHandling` - API error scenarios
- `TestEdgeCases` - Boundary conditions and edge cases

### test_pdf_processor.py

Tests for PDF processing utilities and text manipulation.

**Test Classes:**

- `TestExtractTextFromPDF` - PDF text extraction
- `TestChunkText` - Text chunking with smart boundaries
- `TestGenerateContentHash` - Content hash generation
- `TestProcessPDF` - Complete PDF processing workflow
- `TestEdgeCases` - Edge cases and boundary conditions

### test_integration.py

Tests for end-to-end workflows and cross-feature integration.

**Test Classes:**

- `TestCompleteWorkflow` - Full study notes workflow
- `TestFlashcardWorkflow` - Complete flashcard generation process
- `TestQuizWorkflow` - End-to-end quiz creation
- `TestQAWorkflow` - Q&A session management
- `TestCrossFeatureIntegration` - Multi-feature interactions

## Running Tests

### Run All Tests

```bash
python -m pytest tests/
```

### Run Specific Test File

```bash
python -m pytest tests/test_app.py -v
python -m pytest tests/test_llm_client.py -v
python -m pytest tests/test_pdf_processor.py -v
python -m pytest tests/test_integration.py -v
```

### Run Specific Test Class

```bash
python -m pytest tests/test_app.py::TestProcessPDFEndpoint -v
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=. --cov-report=html
```

## Test Configuration

Tests use the following frameworks and libraries:

- **pytest** - Primary testing framework
- **unittest.mock** - Mocking external dependencies
- **Flask test client** - HTTP endpoint testing
- **Supabase mocking** - Database operation simulation
- **OpenRouter API mocking** - LLM API call simulation

## Key Features Tested

- PDF text extraction and processing
- AI-powered study notes generation
- Flashcard creation with structured output
- Quiz generation with multiple-choice questions
- Context-aware question answering
- Token estimation and cost calculation
- Error handling and edge cases
- Cross-browser CORS functionality
- Database operations and data persistence
- API rate limiting and authentication

## Test Data

Tests use mock data and synthetic responses to ensure consistent, isolated testing without external dependencies or API costs.
