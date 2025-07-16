# LLM Study Coach Backend

A Flask backend service that processes PDF study materials and generates AI-powered study notes, flashcards, quizzes, and interactive Q&A sessions using OpenRouter's GPT-4.1 Nano API.

## Features

- **PDF Processing**: Extract and chunk text from PDF documents with smart boundary detection
- **AI-Powered Study Notes**: Generate comprehensive study notes using GPT-4.1 Nano
- **Flashcard Generation**: Create structured flashcards with difficulty levels and categories
- **Quiz Creation**: Generate multiple-choice quizzes with explanations
- **Interactive Q&A**: Context-aware question answering based on study materials
- **Content Management**: Supabase integration for storing materials, notes, and sessions
- **Cost Estimation**: Token usage and cost calculation for AI operations
- **Content Deduplication**: SHA-256 hashing to avoid reprocessing identical content
- **Debug Tools**: Development endpoints for testing and debugging

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key (for GPT-4.1 Nano access)
- Supabase account and project
- Git (for cloning the repository)

## Technology Stack

- **Backend Framework**: Flask 3.0+ with CORS support
- **AI Integration**: OpenRouter API with GPT-4.1 Nano (1M+ token context)
- **Database**: Supabase (PostgreSQL)
- **PDF Processing**: PyMuPDF (fitz)
- **Testing**: pytest with comprehensive test coverage

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm-study-coach-backend.git
cd llm-study-coach-backend
```

2. Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:

```env
# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=1
```

## Running the Application

1. Make sure your virtual environment is activated:

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

2. Start the Flask development server:

```bash
python app.py
```

The server will start at `http://localhost:5000`

## API Endpoints

### Core Functionality

#### 1. Process PDF

```http
POST /api/process-pdf
Content-Type: multipart/form-data
Headers:
  X-User-ID: <user_uuid>

Form Data:
  file: <pdf_file>
  subject: <subject_name>
```

#### 2. Get Notes

```http
GET /api/notes/<content_hash>
```

#### 3. Generate Content Hash

```http
POST /api/generate-hash
Content-Type: multipart/form-data

Form Data:
  file: <pdf_file>
```

### AI-Powered Features

#### 4. Generate Flashcards

```http
POST /api/generate-flashcards-from-material/<content_hash>
Headers:
  X-User-ID: <user_uuid>

Body:
{
  "category": "optional_category"
}
```

#### 5. Generate Quiz

```http
POST /generate-quiz
Headers:
  X-User-ID: <user_uuid>

Body:
{
  "material_id": "<material_uuid>",
  "subject": "<subject_name>",
  "title": "<quiz_title>"
}
```

#### 6. Ask Question

```http
POST /api/ask-question
Headers:
  X-User-ID: <user_uuid>

Body:
{
  "material_id": "<material_uuid>",
  "question": "<your_question>"
}
```

#### 7. Get Q&A Sessions

```http
GET /api/qa-list?material_id=<material_uuid>
Headers:
  X-User-ID: <user_uuid>
```

#### 8. Delete Q&A Session

```http
DELETE /api/qa/<qa_id>
Headers:
  X-User-ID: <user_uuid>
```

### Debug Endpoints

#### 9. Debug Material

```http
GET /debug-material/<material_id>
```

#### 10. Debug Content

```http
GET /debug-content/<content_hash>
```

## Response Examples

### Process PDF Response

```json
{
  "status": "success",
  "message": "Generated new notes",
  "content": "...",
  "content_hash": "...",
  "model_used": "openai/gpt-4.1-nano",
  "generated_at": "2024-03-14T12:34:56.789Z"
}
```

### Generate Flashcards Response

```json
{
  "status": "success",
  "flashcards": [
    {
      "id": "uuid",
      "front": "What is Python?",
      "back": "A high-level programming language",
      "category": "Programming",
      "difficulty": "easy"
    }
  ],
  "count": 10
}
```

### Generate Quiz Response

```json
{
  "status": "success",
  "quiz": [
    {
      "id": "uuid",
      "question": "What is Python primarily used for?",
      "options": [
        "Web development",
        "Data science",
        "Automation",
        "All of the above"
      ],
      "correct_answer": 3,
      "explanation": "Python is versatile and used for many purposes",
      "difficulty": "easy"
    }
  ],
  "count": 5
}
```

### Ask Question Response

```json
{
  "status": "success",
  "answer": "**Summary:** Python is a programming language...\n\nDetailed explanation...",
  "qa_id": "uuid"
}
```

## Development

### Architecture

- **Flask Application**: RESTful API with CORS support
- **PDF Processing**: PyMuPDF for text extraction with smart chunking
- **LLM Integration**: OpenRouter API with GPT-4.1 Nano (1M+ token context)
- **Database**: Supabase for materials, notes, flashcards, quizzes, and Q&A sessions
- **Cost Management**: Token usage tracking and cost estimation

### Key Components

- `app.py` - Main Flask application with all API endpoints
- `utils/llm_client.py` - LLM integration with prompt templates and response handling
- `utils/pdf_processor.py` - PDF text extraction and intelligent chunking
- `database/quiz_schema.sql` - Database schema for quiz functionality
- `tests/` - Comprehensive test suite with 100+ tests

### Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/test_app.py -v           # API endpoints
python -m pytest tests/test_llm_client.py -v   # LLM integration
python -m pytest tests/test_pdf_processor.py -v # PDF processing
python -m pytest tests/test_integration.py -v   # End-to-end workflows
```

## Error Handling

The API returns appropriate HTTP status codes and detailed error messages:

- **200**: Success
- **400**: Bad Request (missing file, invalid file type, malformed request)
- **401**: Unauthorized (missing user ID)
- **404**: Not Found (notes, material, or Q&A session not found)
- **429**: Rate Limited (API quota exceeded)
- **500**: Internal Server Error

### Error Response Format

```json
{
  "status": "error",
  "message": "Detailed error description"
}
```

## Configuration

### Environment Variables

```env
# OpenRouter API Key (required)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Supabase Configuration (required)
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=1
```

### Model Configuration

- **Model**: GPT-4.1 Nano (1,047,576 token context window)
- **Max Input Tokens**: 1,000,000 (leaving room for output)
- **Max Output Tokens**: 33,000
- **Chunk Size**: 4,000,000 characters (~1M tokens)
- **Cost**: $0.10 per 1M input tokens, $0.40 per 1M output tokens

## Performance

### Optimizations

- **Massive Context Window**: Process entire documents without chunking in most cases
- **Smart Chunking**: Intelligent boundary detection for large documents
- **Content Deduplication**: SHA-256 hashing prevents reprocessing
- **Cost Estimation**: Pre-calculate token usage and costs
- **Database Caching**: Store processed content for quick retrieval

### Supported File Types

- PDF files up to 10MB
- Text extraction from complex layouts
- Multi-page document processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
