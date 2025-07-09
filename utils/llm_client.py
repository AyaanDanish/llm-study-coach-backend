import os
import requests
from typing import Optional


class LLMClient:
    MODEL = "openai/gpt-4.1-nano"
    # Massive chunk size for GPT-4.1 Nano's 1M+ token context window
    OPTIMAL_CHUNK_SIZE = 4000000  # characters (~1M tokens)

    # Token limits for GPT-4.1 Nano
    MAX_INPUT_TOKENS = 1000000  # Leave room for output (1,047,576 total)
    MAX_OUTPUT_TOKENS = 33000

    # Cost per 1M tokens
    INPUT_COST_PER_1M = 0.10
    OUTPUT_COST_PER_1M = 0.40

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_prompt_template(self) -> str:
        """Enhanced prompt template for GPT-4.1 Nano's capabilities."""
        return """
You are an expert study assistant with access to a comprehensive document. Generate detailed, well-structured study notes that cover ALL important content. Use markdown formatting for clarity and organization.

## Your Task:
Create comprehensive study notes following this structure:

## Document Overview
- Brief summary of the main topics covered
- Key learning objectives

## Detailed Notes

For each major section/topic:

### [Section Title]

**Key Concepts:**
- **[Important Term]**: Clear definition and explanation
- **[Important Term]**: Clear definition and explanation

**Main Points:**
- Detailed explanations in bullet points
- Use **bold** for critical terms and concepts
- Include specific examples, formulas, or procedures
- Explain complex ideas step-by-step

**Examples & Applications:**
- Real-world applications when mentioned
- Worked examples with step-by-step solutions
- Practice problems or case studies

**Important Notes:**
- Critical details that shouldn't be missed
- Common misconceptions or pitfalls
- Connections to other concepts

---

## Key Takeaways
- Most important concepts to remember
- Detailed explanation of critical formulas/procedures
- Any additional insights or tips for understanding the material

## Instructions:
1. **Process the ENTIRE document** - don't skip any sections
2. **Maintain logical flow** - follow the document's structure
3. **Be comprehensive** - include all important details
4. **Use clear formatting** - markdown with headers, bullets, bold text
5. **Explain thoroughly** - assume reader needs detailed explanations
6. **Preserve exact formulas/equations** when present
7. **Include all examples** mentioned in the source
8. **Markdown formatting is required for clarity**

Document content:
\"\"\"{chunk}\"\"\""""

    def generate_study_notes(self, chunk: str) -> Optional[str]:
        """
        Generate study notes for a text chunk using GPT-4.1 Nano.

        Args:
            chunk: Text chunk to generate notes for

        Returns:
            Generated notes as string, or None if API call fails
        """
        # Validate chunk size for GPT-4.1 Nano's massive context
        estimated_tokens = self.estimate_tokens(chunk)
        prompt_tokens = self.estimate_tokens(self.get_prompt_template())
        total_input_tokens = estimated_tokens + prompt_tokens

        print(f"📊 Processing with GPT-4.1 Nano:")
        print(f"   Input tokens: {total_input_tokens:,} / {self.MAX_INPUT_TOKENS:,}")

        if total_input_tokens > self.MAX_INPUT_TOKENS:
            print(
                f"⚠️ Chunk too large ({total_input_tokens:,} tokens). Consider splitting."
            )
            return None

        # Calculate estimated cost
        estimated_input_cost = (total_input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        estimated_output_cost = (
            8000 / 1_000_000
        ) * self.OUTPUT_COST_PER_1M  # Assume ~8k output
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        print(f"💰 Estimated cost: ${total_estimated_cost:.4f}")

        prompt = self.get_prompt_template().format(chunk=chunk)

        # Enhanced data payload for GPT-4.1 Nano
        data = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(8000, self.MAX_OUTPUT_TOKENS),  # Reasonable output size
            "temperature": 0.3,  # Slightly creative but focused
            "top_p": 0.9,
        }

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=data
            )  # Check for specific error codes
            if response.status_code == 429:
                print(f"❌ Rate limited by OpenRouter API.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 402:
                print(f"❌ Payment required - insufficient credits.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 400:
                print(f"❌ Bad request - possibly chunk too large or invalid format.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 401:
                print(f"❌ Unauthorized - check your OPENROUTER_API_KEY.")
                print(f"Response: {response.text}")
                return None

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content
                else:
                    print(f"❌ Empty response from API")
                    return None
            else:
                print(f"❌ Invalid response format: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error calling OpenRouter API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
        except (KeyError, IndexError) as e:
            print(f"❌ Error parsing API response: {e}")
            return None

    def generate_notes_for_chunks(self, chunks: list[str]) -> list[str]:
        """
        Generate notes for multiple chunks using GPT-4.1 Nano.

        Args:
            chunks: List of text chunks

        Returns:
            List of generated notes for each chunk
        """
        notes = []
        total_cost = 0.0

        print(f"🚀 Processing {len(chunks)} chunks with GPT-4.1 Nano...")

        for i, chunk in enumerate(chunks):
            print(f"\n📝 Processing chunk {i + 1}/{len(chunks)}...")

            result = self.generate_study_notes(chunk)
            if result:
                notes.append(result)
                print(
                    f"✅ Successfully generated notes for chunk {i + 1}"
                )  # Calculate actual cost (rough estimate)
                chunk_tokens = self.estimate_tokens(chunk)
                output_tokens = self.estimate_tokens(result)
                chunk_cost = (chunk_tokens / 1_000_000) * self.INPUT_COST_PER_1M + (
                    output_tokens / 1_000_000
                ) * self.OUTPUT_COST_PER_1M
                total_cost += chunk_cost
            else:
                error_msg = f"❌ Error generating notes for chunk {i + 1}/{len(chunks)}"
                notes.append(error_msg)
                print(error_msg)

        print(f"\n💰 Total estimated cost: ${total_cost:.4f}")
        return notes

    @staticmethod
    def get_optimal_chunk_size() -> int:
        """
        Returns the optimal chunk size in characters for GPT-4.1 Nano.

        Based on:
        - Model context window: 1,047,576 tokens
        - Prompt overhead: ~200 tokens
        - Expected output: ~8,000 tokens
        - Token-to-char ratio: ~1:4
        - Safety buffer: ~1,000 tokens

        Returns:
            Optimal chunk size in characters (~4,000,000)
        """
        return LLMClient.OPTIMAL_CHUNK_SIZE

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a given text.

        Args:
            text: Input text to estimate tokens for

        Returns:
            Estimated token count (using 1 token ≈ 4 characters)
        """
        return len(text) // 4

    def test_api_connection(self) -> bool:
        """
        Test the API connection and authentication.

        Returns:
            True if connection is successful, False otherwise
        """
        test_data = {
            "model": self.MODEL,
            "messages": [
                {"role": "user", "content": "Hello, just testing the connection."}
            ],
            "max_tokens": 10,
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=test_data)

            if response.status_code == 429:
                print("❌ Rate limited - free model has strict limits")
                return False
            elif response.status_code == 401:
                print("❌ Unauthorized - check your OPENROUTER_API_KEY")
                return False
            elif response.status_code == 402:
                print("❌ Payment required - may have exceeded free tier")
                return False
            elif response.status_code == 200:
                print("✅ API connection successful")
                return True
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    @staticmethod
    def can_process_entire_document(document_size_chars: int) -> bool:
        """
        Check if entire document can be processed in a single API call.

        Args:
            document_size_chars: Size of document in characters

        Returns:
            True if can be processed in single call with GPT-4.1 Nano
        """
        estimated_tokens = document_size_chars // 4
        prompt_tokens = 200  # Prompt overhead
        output_tokens = 8000  # Expected output
        total_tokens = estimated_tokens + prompt_tokens + output_tokens

        return total_tokens <= LLMClient.MAX_INPUT_TOKENS

    def get_processing_recommendation(self, document_size_chars: int) -> dict:
        """
        Get processing recommendations for a document with GPT-4.1 Nano.

        Args:
            document_size_chars: Size of document in characters

        Returns:
            Dictionary with processing recommendations
        """
        estimated_tokens = document_size_chars // 4

        if self.can_process_entire_document(document_size_chars):
            estimated_cost = (estimated_tokens / 1_000_000) * self.INPUT_COST_PER_1M + (
                8000 / 1_000_000
            ) * self.OUTPUT_COST_PER_1M

            return {
                "strategy": "single_call",
                "chunks_needed": 1,
                "estimated_tokens": estimated_tokens,
                "estimated_cost": estimated_cost,
                "benefits": [
                    "No context loss",
                    "Comprehensive analysis",
                    "Single API call",
                    "Maximum coherence",
                ],
            }
        else:
            # Calculate chunks needed
            max_tokens_per_chunk = 800000  # Conservative estimate
            chunks_needed = (
                estimated_tokens + max_tokens_per_chunk - 1
            ) // max_tokens_per_chunk

            estimated_cost = chunks_needed * (
                (800000 / 1_000_000) * self.INPUT_COST_PER_1M
                + (8000 / 1_000_000) * self.OUTPUT_COST_PER_1M
            )

            return {
                "strategy": "chunked",
                "chunks_needed": chunks_needed,
                "estimated_tokens": estimated_tokens,
                "estimated_cost": estimated_cost,
                "benefits": [
                    "Still very large chunks",
                    "Minimal context loss",
                    f"Only {chunks_needed} API calls needed",
                ],
            }

    def estimate_cost(self, text: str, output_tokens: int = 8000) -> float:
        """
        Estimate the cost for processing given text.

        Args:
            text: Input text to process
            output_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        input_tokens = self.estimate_tokens(text)
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M

        return input_cost + output_cost

    def get_flashcard_prompt_template(self) -> str:
        """Prompt template specifically designed for generating flashcards with structured outputs."""
        return """You are an expert study assistant specialized in creating effective flashcards for learning and memorization. Generate a set of high-quality flashcards based on the provided study material.

## Guidelines for Effective Flashcards:
1. **Focus on key concepts** - Extract the most important information
2. **Keep it concise** - Front should be a clear question, back should be a focused answer
3. **One concept per card** - Don't try to cover multiple ideas in one flashcard
4. **Use clear language** - Avoid ambiguous wording
5. **Include context when needed** - But keep it brief
6. **Mix question types** - Definitions, examples, applications, comparisons

## Difficulty Guidelines:
- **easy**: Basic definitions, simple facts
- **medium**: Concepts requiring understanding, applications
- **hard**: Complex processes, analysis, synthesis

## Instructions:
- Generate 8-15 flashcards covering the most important concepts
- Ensure variety in question types (what, how, why, when, examples)
- Use the same category for related flashcards
- Focus on information that students need to memorize or understand deeply
- Avoid yes/no questions - prefer open-ended questions that require explanation

Content to create flashcards from:
\"\"\"{content}\"\"\""""

    def generate_flashcards(self, content: str, category: str = None) -> Optional[list]:
        """
        Generate flashcards from study content using GPT-4.1 Nano.

        Args:
            content: Study material content to create flashcards from
            category: Optional category name for the flashcards

        Returns:
            List of flashcard dictionaries, or None if API call fails
        """
        # Validate content size
        estimated_tokens = self.estimate_tokens(content)
        prompt_tokens = self.estimate_tokens(self.get_flashcard_prompt_template())
        total_input_tokens = estimated_tokens + prompt_tokens

        print(f"📚 Generating flashcards with GPT-4.1 Nano:")
        print(f"   Input tokens: {total_input_tokens:,} / {self.MAX_INPUT_TOKENS:,}")

        if total_input_tokens > self.MAX_INPUT_TOKENS:
            print(
                f"⚠️ Content too large ({total_input_tokens:,} tokens). Consider using summary."
            )
            return None

        # Calculate estimated cost
        estimated_input_cost = (total_input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        estimated_output_cost = (
            3000 / 1_000_000
        ) * self.OUTPUT_COST_PER_1M  # Assume ~3k output
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        print(f"💰 Estimated cost: ${total_estimated_cost:.4f}")

        prompt = self.get_flashcard_prompt_template().format(
            content=content
        )  # Enhanced data payload for flashcard generation with structured outputs
        data = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,  # Enough for multiple flashcards
            "temperature": 0.1,  # Low temperature for consistent formatting
            "top_p": 0.8,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "flashcards",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "flashcards": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "front": {
                                            "type": "string",
                                            "description": "The question or prompt for the flashcard",
                                        },
                                        "back": {
                                            "type": "string",
                                            "description": "The answer or explanation for the flashcard",
                                        },
                                        "category": {
                                            "type": "string",
                                            "description": "The subject or topic category",
                                        },
                                        "difficulty": {
                                            "type": "string",
                                            "enum": ["easy", "medium", "hard"],
                                            "description": "The difficulty level of the flashcard",
                                        },
                                    },
                                    "required": [
                                        "front",
                                        "back",
                                        "category",
                                        "difficulty",
                                    ],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["flashcards"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)

            # Handle specific error codes
            if response.status_code == 429:
                print(f"❌ Rate limited by OpenRouter API.")
                print(f"Response: {response.text}")

                return None
            elif response.status_code == 402:
                print(f"❌ Payment required - insufficient credits.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 400:
                print(f"❌ Bad request - possibly content too large or invalid format.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 401:
                print(f"❌ Unauthorized - check your OPENROUTER_API_KEY.")
                print(f"Response: {response.text}")
                return None

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content_result = response_data["choices"][0]["message"]["content"]
                if content_result and content_result.strip():
                    try:
                        # Parse the structured JSON response
                        import json

                        print(f"🔍 Parsing structured output...")

                        response_json = json.loads(content_result)

                        # Extract flashcards from structured response
                        if "flashcards" in response_json and isinstance(
                            response_json["flashcards"], list
                        ):
                            flashcards = response_json["flashcards"]
                        else:
                            print(
                                f"❌ Invalid response structure: missing 'flashcards' array"
                            )
                            return None

                        # Validate each flashcard (should be valid due to structured output, but double-check)
                        valid_flashcards = []
                        for card in flashcards:
                            if (
                                isinstance(card, dict)
                                and "front" in card
                                and "back" in card
                                and card["front"].strip()
                                and card["back"].strip()
                            ):

                                # Set default values if missing (shouldn't happen with structured output)
                                if "category" not in card:
                                    card["category"] = category or "General"
                                if "difficulty" not in card:
                                    card["difficulty"] = "medium"

                                valid_flashcards.append(card)

                        if valid_flashcards:
                            print(f"✅ Generated {len(valid_flashcards)} flashcards")
                            return valid_flashcards
                        else:
                            print(f"❌ No valid flashcards found in response")
                            return None

                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing JSON response: {e}")
                        print(f"Raw response: {content_result}")
                        return None
                else:
                    print(f"❌ Empty response from API")
                    return None
            else:
                print(f"❌ Invalid response format: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error calling OpenRouter API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def get_quiz_prompt_template(self) -> str:
        """Prompt template specifically designed for generating quiz questions with structured outputs."""
        return """You are an expert study assistant specialized in creating effective multiple-choice quiz questions for learning assessment. Generate exactly 5 high-quality quiz questions based on the provided study material.

## Guidelines for Effective Quiz Questions:
1. **Test understanding** - Focus on comprehension, application, and analysis
2. **Clear and unambiguous** - Questions should have one clearly correct answer
3. **Realistic distractors** - Wrong answers should be plausible but clearly incorrect
4. **Variety in difficulty** - Mix of easy, medium, and hard questions
5. **Cover key concepts** - Focus on the most important material
6. **Answerable from content** - All questions must be answerable from the provided material

## Question Types to Include:
- **Definitions**: What is...?
- **Applications**: How would you use...?
- **Analysis**: Why does...?
- **Comparisons**: What's the difference between...?
- **Examples**: Which of the following is an example of...?

## Instructions:
- Generate exactly 5 multiple-choice questions
- Each question must have exactly 4 options (A, B, C, D)
- Only one option should be correct
- Include brief explanations for correct answers
- Ensure questions test different aspects of the material
- Questions should be clear and professional

Subject: {subject}
Material Title: {title}

Study Material Content:
\"\"\"{content}\"\"\""""

    def generate_quiz(self, content: str, subject: str, title: str) -> Optional[list]:
        """
        Generate quiz questions from study content using GPT-4.1 Nano.

        Args:
            content: Study material content to create quiz questions from
            subject: Subject/category of the material
            title: Title of the study material

        Returns:
            List of quiz question dictionaries, or None if API call fails
        """
        # Validate content size
        estimated_tokens = self.estimate_tokens(content)
        prompt_tokens = self.estimate_tokens(self.get_quiz_prompt_template())
        total_input_tokens = estimated_tokens + prompt_tokens

        print(f"🧠 Generating quiz questions with GPT-4.1 Nano:")
        print(f"   Input tokens: {total_input_tokens:,} / {self.MAX_INPUT_TOKENS:,}")

        if total_input_tokens > self.MAX_INPUT_TOKENS:
            print(
                f"⚠️ Content too large ({total_input_tokens:,} tokens). Consider using summary."
            )
            return None

        # Calculate estimated cost
        estimated_input_cost = (total_input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        estimated_output_cost = (
            2000 / 1_000_000
        ) * self.OUTPUT_COST_PER_1M  # Assume ~2k output for quiz
        total_estimated_cost = estimated_input_cost + estimated_output_cost
        print(f"💰 Estimated cost: ${total_estimated_cost:.4f}")

        prompt = self.get_quiz_prompt_template().format(
            content=content, subject=subject, title=title
        )

        # Enhanced data payload for quiz generation with structured outputs
        data = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,  # Sufficient for 5 questions with explanations
            "temperature": 0.1,  # Low temperature for consistent formatting
            "top_p": 0.8,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "quiz",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": "The question text",
                                        },
                                        "options": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "minItems": 4,
                                            "maxItems": 4,
                                            "description": "Exactly 4 answer options",
                                        },
                                        "correct_answer": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 3,
                                            "description": "Index of correct answer (0-3)",
                                        },
                                        "explanation": {
                                            "type": "string",
                                            "description": "Brief explanation of correct answer",
                                        },
                                        "difficulty": {
                                            "type": "string",
                                            "enum": ["easy", "medium", "hard"],
                                            "description": "Question difficulty level",
                                        },
                                    },
                                    "required": [
                                        "question",
                                        "options",
                                        "correct_answer",
                                        "explanation",
                                        "difficulty",
                                    ],
                                    "additionalProperties": False,
                                },
                                "minItems": 5,
                                "maxItems": 5,
                                "description": "Exactly 5 quiz questions",
                            }
                        },
                        "required": ["questions"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        try:
            print(f"🔄 Calling OpenRouter API...")
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=60
            )

            if response.status_code == 429:
                print(f"⚠️ Rate limit exceeded. Please wait and try again.")
                return None
            elif response.status_code == 400:
                print(f"❌ Bad request - possibly content too large or invalid format.")
                print(f"Response: {response.text}")
                return None
            elif response.status_code == 401:
                print(f"❌ Unauthorized - check your OPENROUTER_API_KEY.")
                print(f"Response: {response.text}")
                return None

            response.raise_for_status()
            response_data = response.json()

            if "choices" in response_data and len(response_data["choices"]) > 0:
                content_result = response_data["choices"][0]["message"]["content"]
                if content_result and content_result.strip():
                    try:
                        # Parse the structured JSON response
                        import json

                        print(f"🔍 Parsing structured output...")

                        response_json = json.loads(content_result)

                        # Extract questions from structured response
                        if "questions" in response_json and isinstance(
                            response_json["questions"], list
                        ):
                            questions = response_json["questions"]
                        else:
                            print(
                                f"❌ Invalid response structure: missing 'questions' array"
                            )
                            return None

                        # Validate each question (should be valid due to structured output, but double-check)
                        valid_questions = []
                        for i, question in enumerate(questions):
                            if (
                                isinstance(question, dict)
                                and "question" in question
                                and "options" in question
                                and "correct_answer" in question
                                and "explanation" in question
                                and isinstance(question["options"], list)
                                and len(question["options"]) == 4
                                and isinstance(question["correct_answer"], int)
                                and 0 <= question["correct_answer"] <= 3
                            ):
                                # Add unique ID for each question
                                question["id"] = f"q_{i+1}"
                                valid_questions.append(question)
                            else:
                                print(f"❌ Invalid question format at index {i}")

                        if len(valid_questions) == 5:
                            print(f"✅ Generated {len(valid_questions)} quiz questions")
                            return valid_questions
                        else:
                            print(
                                f"❌ Expected 5 questions, got {len(valid_questions)} valid questions"
                            )
                            return None

                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing JSON response: {e}")
                        print(f"Raw response: {content_result}")
                        return None
                else:
                    print(f"❌ Empty response from API")
                    return None
            else:
                print(f"❌ Invalid response format: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error calling OpenRouter API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
