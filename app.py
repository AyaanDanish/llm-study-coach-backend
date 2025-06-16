import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from supabase import create_client, Client
from utils.pdf_processor import process_pdf
from utils.llm_client import LLMClient
from dotenv import load_dotenv
from datetime import datetime
import hashlib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize LLM client
llm_client = LLMClient()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process-pdf', methods=['POST'])
def process_pdf_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    # Get user_id from request
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({'error': 'User ID not provided'}), 401

    # Get subject and content_hash from request
    subject = request.form.get('subject')
    content_hash = request.form.get('content_hash')
    if not subject:
        return jsonify({'error': 'Subject not provided'}), 400
    if not content_hash:
        return jsonify({'error': 'Content hash not provided'}), 400

    try:
        # Process PDF
        file_bytes = file.read()
        text, chunks, _ = process_pdf(file_bytes)  # We don't need the hash since it's provided

        # Check if notes exist in database
        existing_notes = supabase.table('study_notes')\
            .select('*')\
            .eq('content_hash', content_hash)\
            .execute()
        
        if existing_notes.data:
            # Return existing notes
            return jsonify({
                'status': 'success',
                'message': 'Retrieved existing notes',
                'content': existing_notes.data[0]['content'],
                'content_hash': content_hash,
                'model_used': existing_notes.data[0]['model_used'],
                'generated_at': existing_notes.data[0]['generated_at']
            })

        # Generate new notes
        notes = llm_client.generate_notes_for_chunks(chunks)
        combined_notes = "\n\n".join([f"\n\n{note}" for i, note in enumerate(notes)])

        # Store in study_notes table
        supabase.table('study_notes').insert({
            'content_hash': content_hash,
            'content': combined_notes,
            'model_used': LLMClient.MODEL,
            'prompt_used': llm_client.get_prompt_template()
        }).execute()

        return jsonify({
            'status': 'success',
            'message': 'Generated new notes',
            'content': combined_notes,
            'content_hash': content_hash,
            'model_used': LLMClient.MODEL,
            'generated_at': datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/notes/<content_hash>', methods=['GET'])
def get_notes(content_hash):
    try:
        response = supabase.table('study_notes').select('*').eq('content_hash', content_hash).execute()
        
        if not response.data:
            return jsonify({'error': 'Notes not found'}), 404

        return jsonify({
            'status': 'success',
            'content': response.data[0]['content'],
            'content_hash': content_hash,
            'model_used': response.data[0]['model_used'],
            'generated_at': response.data[0]['generated_at']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/generate-hash', methods=['POST'])
def generate_hash():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    # Get user ID from headers
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        return jsonify({'error': 'User ID not provided'}), 401

    try:
        # Read the PDF content and generate hash
        file_bytes = file.read()
        _, _, content_hash = process_pdf(file_bytes)  # Reuse your existing process_pdf function
        
        return jsonify({
            'content_hash': content_hash
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For local development
if __name__ == '__main__':
    app.run(debug=True) 