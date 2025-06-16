# LLM Study Coach Backend

A Flask backend service that processes PDF study materials and generates AI-powered study notes using OpenRouter's LLM API.

## Features

- PDF text extraction and processing
- AI-powered study note generation using OpenRouter's LLM API
- Supabase integration for storing study materials and notes
- Content-based deduplication using SHA-256 hashing
- RESTful API endpoints for PDF processing and note retrieval

## Prerequisites

- Python 3.8 or higher
- OpenRouter API key
- Supabase account and project
- Git (for cloning the repository)

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

### 1. Process PDF
```http
POST /api/process-pdf
Content-Type: multipart/form-data
Headers:
  X-User-ID: <user_uuid>

Form Data:
  file: <pdf_file>
  subject: <subject_name>
```

Response:
```json
{
  "status": "success",
  "message": "Generated new notes",
  "content": "...",
  "content_hash": "...",
  "model_used": "mistralai/mistral-7b-instruct:free",
  "generated_at": "2024-03-14T12:34:56.789Z"
}
```

### 2. Get Notes
```http
GET /api/notes/<content_hash>
```

Response:
```json
{
  "status": "success",
  "content": "...",
  "content_hash": "...",
  "model_used": "mistralai/mistral-7b-instruct:free",
  "generated_at": "2024-03-14T12:34:56.789Z"
}
```

## Development

- The application uses Flask for the web server
- PDF processing is handled by PyMuPDF (fitz)
- LLM integration is done through OpenRouter's API
- Database operations are managed through Supabase

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing file, invalid file type)
- 401: Unauthorized (missing user ID)
- 404: Not Found (notes not found)
- 500: Internal Server Error

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 