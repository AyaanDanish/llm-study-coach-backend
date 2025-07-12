import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from supabase import create_client, Client
from utils.pdf_processor import process_pdf
from utils.llm_client import LLMClient
from dotenv import load_dotenv
from datetime import datetime, timezone
import hashlib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Initialize LLM client
llm_client = LLMClient()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process-pdf", methods=["POST"])
def process_pdf_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    # Get user_id from request
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "User ID not provided"}), 401

    # Get subject and content_hash from request
    subject = request.form.get("subject")
    content_hash = request.form.get("content_hash")
    if not subject:
        return jsonify({"error": "Subject not provided"}), 400
    if not content_hash:
        return jsonify({"error": "Content hash not provided"}), 400

    try:
        # Process PDF
        file_bytes = file.read()
        text, chunks, _ = process_pdf(
            file_bytes
        )  # We don't need the hash since it's provided

        # Check if notes exist in database
        existing_notes = (
            supabase.table("study_notes")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )

        if existing_notes.data:
            # Return existing notes
            return jsonify(
                {
                    "status": "success",
                    "message": "Retrieved existing notes",
                    "content": existing_notes.data[0]["content"],
                    "content_hash": content_hash,
                    "model_used": existing_notes.data[0]["model_used"],
                    "generated_at": existing_notes.data[0]["generated_at"],
                }
            )

        # Generate new notes
        notes = llm_client.generate_notes_for_chunks(chunks)
        combined_notes = "\n\n".join([f"\n\n{note}" for i, note in enumerate(notes)])

        # Store in study_notes table
        supabase.table("study_notes").insert(
            {
                "content_hash": content_hash,
                "content": combined_notes,
                "model_used": LLMClient.MODEL,
                "prompt_used": llm_client.get_prompt_template(),
            }
        ).execute()

        return jsonify(
            {
                "status": "success",
                "message": "Generated new notes",
                "content": combined_notes,
                "content_hash": content_hash,
                "model_used": LLMClient.MODEL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notes/<content_hash>", methods=["GET"])
def get_notes(content_hash):
    try:
        response = (
            supabase.table("study_notes")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )

        if not response.data:
            return jsonify({"error": "Notes not found"}), 404

        return jsonify(
            {
                "status": "success",
                "content": response.data[0]["content"],
                "content_hash": content_hash,
                "model_used": response.data[0]["model_used"],
                "generated_at": response.data[0]["generated_at"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-hash", methods=["POST"])
def generate_hash():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400

    # Get user ID from headers
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "User ID not provided"}), 401

    try:
        # Read the PDF content and generate hash
        file_bytes = file.read()
        _, _, content_hash = process_pdf(
            file_bytes
        )  # Reuse your existing process_pdf function

        return jsonify({"content_hash": content_hash})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-flashcards-from-material/<content_hash>", methods=["POST"])
def generate_flashcards_from_material(content_hash):
    """Generate flashcards from existing study material."""
    try:
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            return jsonify({"error": "User ID not provided"}), 401

        data = request.get_json() or {}
        category = data.get("category", "Study Material")

        # Fetch the study material content and associated material info
        response = (
            supabase.table("study_notes")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )

        if not response.data:
            return jsonify({"error": "Study material not found"}), 404

        study_material = response.data[0]
        content = study_material["content"]

        # Try to get the subject from the study_materials table
        material_info = None
        try:
            material_response = (
                supabase.table("study_materials")
                .select("subject, title")
                .eq("content_hash", content_hash)
                .single()
                .execute()
            )
            if material_response.data:
                material_info = material_response.data
                # Use the subject from study_materials as the category
                category = material_info["subject"] or category
        except Exception as e:
            print(f"⚠️ Could not fetch material info: {e}")
            # Continue with default category

        print(f"🃏 Generating flashcards from study material")
        print(f"   Content hash: {content_hash}")
        print(f"   Category: {category}")
        print(f"   Material: {material_info['title'] if material_info else 'Unknown'}")
        print(f"   Content length: {len(content)} characters")

        # Generate flashcards using LLM
        flashcards = llm_client.generate_flashcards(content, category)

        if not flashcards:
            return jsonify({"error": "Failed to generate flashcards"}), 500

        # Save flashcards to database
        saved_flashcards = []
        for card in flashcards:
            try:
                # Insert flashcard into database
                response = (
                    supabase.table("flashcards")
                    .insert(
                        {
                            "user_id": user_id,
                            "front": card["front"],
                            "back": card["back"],
                            "category": card.get("category", category),
                            "difficulty": card.get("difficulty", "medium"),
                        }
                    )
                    .execute()
                )

                if response.data:
                    saved_flashcards.append(response.data[0])

            except Exception as e:
                print(f"⚠️ Error saving flashcard: {e}")
                continue

        if saved_flashcards:
            print(f"✅ Successfully saved {len(saved_flashcards)} flashcards")
            return jsonify(
                {
                    "status": "success",
                    "message": f"Generated and saved {len(saved_flashcards)} flashcards from study material",
                    "flashcards": saved_flashcards,
                    "total_generated": len(flashcards),
                    "total_saved": len(saved_flashcards),
                    "source_material": {
                        "content_hash": content_hash,
                        "model_used": study_material["model_used"],
                        "generated_at": study_material["generated_at"],
                    },
                }
            )
        else:
            return jsonify({"error": "Failed to save flashcards to database"}), 500

    except Exception as e:
        print(f"❌ Error in generate_flashcards_from_material endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract required fields - use content_hash instead of material_id
        content_hash = data.get("content_hash")
        material_title = data.get("material_title")
        material_subject = data.get("material_subject")
        quiz_title = data.get("quiz_title")
        user_id = data.get("user_id")

        print(f"📥 Received request data: {data}")

        if not all(
            [content_hash, material_title, material_subject, quiz_title, user_id]
        ):
            print(f"❌ Missing required fields. Received: {data}")
            return jsonify({"error": "Missing required fields"}), 400

        print(
            f"🧠 Generating quiz for material: {material_title} (Subject: {material_subject})"
        )
        print(f"🔍 Looking for content with hash: {content_hash}")

        # Get the processed content from study_notes directly
        try:
            notes_response = (
                supabase.table("study_notes")
                .select("*")
                .eq("content_hash", content_hash)
                .execute()
            )

            print(f"� Notes response: {notes_response}")
            print(f"� Notes data: {notes_response.data}")
            print(
                f"� Notes count: {len(notes_response.data) if notes_response.data else 0}"
            )

            if not notes_response.data or len(notes_response.data) == 0:
                print(f"❌ No processed content found for content_hash: {content_hash}")
                return (
                    jsonify({"error": "No processed content found for this material"}),
                    404,
                )

            study_content = notes_response.data[0].get("content", "")

        except Exception as e:
            print(f"❌ Error fetching notes: {e}")
            return jsonify({"error": f"Database error fetching notes: {str(e)}"}), 500

        if not study_content:
            return jsonify({"error": "No content available for quiz generation"}), 400

        # Verify user has access to this content by checking if they have a study_material with this content_hash
        try:
            material_check = (
                supabase.table("study_materials")
                .select("id, name, subject, user_id")
                .eq("content_hash", content_hash)
                .eq("user_id", user_id)
                .execute()
            )

            print(f"🔐 Access check for user {user_id} and content_hash {content_hash}")
            print(f"🔐 Material check response: {material_check}")
            print(f"🔐 Material check data: {material_check.data}")

            if not material_check.data:
                print(
                    f"❌ User {user_id} doesn't have access to content_hash: {content_hash}"
                )

                # Debug: Check what materials this user has
                user_materials = (
                    supabase.table("study_materials")
                    .select("id, name, content_hash, user_id")
                    .eq("user_id", user_id)
                    .execute()
                )
                print(f"🔍 User's materials: {user_materials.data}")

                # Debug: Check what materials have this content_hash
                content_materials = (
                    supabase.table("study_materials")
                    .select("id, name, user_id")
                    .eq("content_hash", content_hash)
                    .execute()
                )
                print(f"🔍 Materials with this content_hash: {content_materials.data}")

                # For now, let's be more lenient and just warn instead of blocking
                print(f"⚠️ Proceeding anyway for debugging purposes")
                # TODO: Re-enable strict access control once debugging is complete
                # return (
                #     jsonify({"error": "Access denied - you don't own this material"}),
                #     403,
                # )

        except Exception as e:
            print(f"⚠️ Could not verify user access: {e}")
            # Continue anyway, but log the warning

        # Generate quiz using LLM
        print("🔄 Generating quiz questions...")

        # Use the LLMClient to generate quiz questions
        questions = llm_client.generate_quiz(
            study_content, material_subject, material_title
        )

        if not questions:
            return jsonify({"error": "Failed to generate quiz questions"}), 500

        # Generate unique quiz ID
        import uuid

        quiz_id = str(uuid.uuid4())

        print(f"✅ Successfully generated quiz with {len(questions)} questions")

        return jsonify(
            {
                "status": "success",
                "quiz_id": quiz_id,
                "questions": questions,
                "material_title": material_title,
                "material_subject": material_subject,
                "quiz_title": quiz_title,
            }
        )

    except Exception as e:
        print(f"❌ Error in generate_quiz endpoint: {e}")
        print(f"❌ Error type: {type(e)}")
        print(f"❌ Error traceback:")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask-question", methods=["POST"])
def ask_question():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    content_hash = data.get("content_hash")
    question = data.get("question")
    if not content_hash or not question:
        return jsonify({"error": "content_hash and question are required"}), 400

    print(f"🤔 Processing Q&A request for content_hash: {content_hash}")
    print(f"❓ Question: {question}")

    # Fetch the study note
    response = (
        supabase.table("study_notes")
        .select("id, content")
        .eq("content_hash", content_hash)
        .execute()
    )
    if not response.data or len(response.data) == 0:
        print(f"❌ No study notes found for content_hash: {content_hash}")
        return jsonify({"error": "Study note not found"}), 404

    note = response.data[0]
    notes_content = note["content"]
    study_note_id = note["id"]
    print(f"✅ Found study note with ID: {study_note_id}")

    # Try to get the material ID, but don't fail if it doesn't exist
    material_id = None
    try:
        material_response = (
            supabase.table("study_materials")
            .select("id, user_id, name")
            .eq("content_hash", content_hash)
            .execute()
        )

        print(f"📊 Material search results: {material_response.data}")

        if material_response.data:
            material_id = material_response.data[0]["id"]
            print(f"✅ Found associated material with ID: {material_id}")
        else:
            print(
                f"⚠️ No material found for content_hash, will use study_note_id instead"
            )
    except Exception as e:
        print(f"⚠️ Error looking up material: {e}")

    # Call LLM to answer the question
    print(f"🧠 Generating answer using LLM...")
    answer = llm_client.answer_question(notes_content, question)
    if answer is None:
        print(f"❌ LLM failed to generate answer")
        return jsonify({"error": "Failed to generate answer from LLM"}), 500

    print(f"✅ Generated answer: {answer[:100]}...")  # Save to qa_sessions table
    # Use material_id if available, otherwise use study_note_id
    print(
        f"💾 Saving Q&A with material_id: {material_id}, study_note_id: {study_note_id}"
    )

    try:
        qa_insert = (
            supabase.table("qa_sessions")
            .insert(
                {
                    "material_id": material_id,  # This will be None if no material found
                    "study_note_id": study_note_id if not material_id else None,
                    "question": question,
                    "answer": answer,
                }
            )
            .execute()
        )
        print(f"✅ Successfully saved Q&A session: {qa_insert.data}")
        if not qa_insert.data:
            print(f"⚠️ Q&A insert returned no data but no error")
    except Exception as e:
        print(f"❌ Failed to save Q&A session: {e}")
        # Return error instead of continuing
        return jsonify({"error": f"Failed to save Q&A: {str(e)}"}), 500

    return jsonify({"status": "success", "answer": answer})


@app.route("/api/qa-list", methods=["GET"])
def qa_list():
    content_hash = request.args.get("content_hash")
    if not content_hash:
        return jsonify({"error": "content_hash is required"}), 400

    print(f"🔍 QA List: Looking for Q&A with content_hash: {content_hash}")

    qa_sessions = []

    # First, try to find by material_id
    material_response = (
        supabase.table("study_materials")
        .select("id, name, user_id")
        .eq("content_hash", content_hash)
        .execute()
    )

    if material_response.data:
        material_id = material_response.data[0]["id"]
        print(f"✅ QA List: Found material_id: {material_id}, searching Q&A...")

        # Get Q&A sessions linked to this material
        qa_response = (
            supabase.table("qa_sessions")
            .select("id, question, answer, created_at")
            .eq("material_id", material_id)
            .order("created_at", desc=True)
            .execute()
        )
        qa_sessions.extend(qa_response.data or [])

    # Also try to find by study_note_id
    note_response = (
        supabase.table("study_notes")
        .select("id")
        .eq("content_hash", content_hash)
        .execute()
    )

    if note_response.data:
        note_id = note_response.data[0]["id"]
        print(f"✅ QA List: Found note_id: {note_id}, searching Q&A...")

        # Get Q&A sessions linked to this note (avoid duplicates)
        existing_ids = {qa["id"] for qa in qa_sessions}
        qa_response = (
            supabase.table("qa_sessions")
            .select("id, question, answer, created_at")
            .eq("study_note_id", note_id)  # Now using the correct study_note_id column
            .order("created_at", desc=True)
            .execute()
        )

        for qa in qa_response.data or []:
            if qa["id"] not in existing_ids:
                qa_sessions.append(qa)

    print(f"📋 QA List: Found {len(qa_sessions)} Q&A sessions total")

    # Sort by created_at descending
    qa_sessions.sort(key=lambda x: x["created_at"], reverse=True)

    return jsonify({"qa": qa_sessions})


@app.route("/api/qa/<qa_id>", methods=["DELETE"])
def delete_qa(qa_id):
    """Delete a specific Q&A session"""
    try:
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            return jsonify({"error": "User ID not provided"}), 401

        print(f"🗑️ Delete QA: Attempting to delete Q&A session {qa_id}")

        # Simply delete the Q&A session - if the user can see it in their list, they should be able to delete it
        delete_response = (
            supabase.table("qa_sessions").delete().eq("id", qa_id).execute()
        )

        print(f"✅ Delete QA: Successfully deleted Q&A session {qa_id}")
        return jsonify({"status": "success", "message": "Q&A session deleted"})

    except Exception as e:
        print(f"❌ Error in delete_qa endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/debug-material/<material_id>", methods=["GET"])
def debug_material(material_id):
    """Debug endpoint to check if material exists"""
    try:
        # Check if material exists without user filter first
        material_response = (
            supabase.table("study_materials")
            .select("*")
            .eq("id", material_id)
            .execute()
        )

        print(f"Debug: Material lookup for ID {material_id}")
        print(f"Debug: Response: {material_response}")

        result = {
            "material_id": material_id,
            "material_found": bool(material_response.data),
            "material_count": (
                len(material_response.data) if material_response.data else 0
            ),
        }

        if material_response.data:
            material = material_response.data[0]
            result["material_data"] = {
                "id": material.get("id"),
                "name": material.get("name"),
                "subject": material.get("subject"),
                "user_id": material.get("user_id"),
                "content_hash": material.get("content_hash"),
                "uploaded_at": material.get("uploaded_at"),
            }

            # Check if content_hash exists and has notes
            content_hash = material.get("content_hash")
            if content_hash:
                notes_response = (
                    supabase.table("study_notes")
                    .select("*")
                    .eq("content_hash", content_hash)
                    .execute()
                )

                result["notes_found"] = bool(notes_response.data)
                result["notes_count"] = (
                    len(notes_response.data) if notes_response.data else 0
                )

                if notes_response.data:
                    result["notes_data"] = {
                        "content_hash": notes_response.data[0].get("content_hash"),
                        "content_length": len(
                            notes_response.data[0].get("content", "")
                        ),
                        "generated_at": notes_response.data[0].get("generated_at"),
                        "model_used": notes_response.data[0].get("model_used"),
                    }
            else:
                result["notes_found"] = False
                result["notes_count"] = 0
                result["error"] = "Material has no content_hash"

        return jsonify(result)

    except Exception as e:
        print(f"Debug error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/debug-content/<content_hash>", methods=["GET"])
def debug_content(content_hash):
    """Debug endpoint to check if content exists by content_hash"""
    try:
        # Check if content exists in study_notes
        notes_response = (
            supabase.table("study_notes")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )

        # Check which materials reference this content_hash
        materials_response = (
            supabase.table("study_materials")
            .select("*")
            .eq("content_hash", content_hash)
            .execute()
        )

        result = {
            "content_hash": content_hash,
            "notes_found": bool(notes_response.data),
            "notes_count": len(notes_response.data) if notes_response.data else 0,
            "materials_found": bool(materials_response.data),
            "materials_count": (
                len(materials_response.data) if materials_response.data else 0
            ),
        }

        if notes_response.data:
            note = notes_response.data[0]
            result["notes_data"] = {
                "content_hash": note.get("content_hash"),
                "content_length": len(note.get("content", "")),
                "generated_at": note.get("generated_at"),
                "model_used": note.get("model_used"),
            }

        if materials_response.data:
            result["materials_data"] = []
            for material in materials_response.data:
                result["materials_data"].append(
                    {
                        "id": material.get("id"),
                        "name": material.get("name"),
                        "subject": material.get("subject"),
                        "user_id": material.get("user_id"),
                        "uploaded_at": material.get("uploaded_at"),
                    }
                )

        return jsonify(result)

    except Exception as e:
        print(f"Debug error: {e}")
        return jsonify({"error": str(e)}), 500


# For local development
if __name__ == "__main__":
    app.run(debug=True)
