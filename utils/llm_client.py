import os
import requests
from typing import Optional

class LLMClient:
    MODEL = "mistralai/mistral-7b-instruct:free"

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_prompt_template(self) -> str:
        """Returns the template used for generating study notes."""
        return """
You are a helpful study assistant. Based on the following text from a study document, generate clear and concise study notes. The notes should be divided **section by section** according to the structure of the original content. For each section, do the following:

1. Provide **detailed, easy-to-understand notes** in bullet points. Use markdown formatting. Use **bold** for important words and phrases. Use whole sentences and not just phrases. Explain every concept as if the user does not know anything about the topic.
2. Highlight **important definitions, formulas, and concepts**.
3. If the section includes examples, explain them clearly.
4. Keep the tone educational but not too formal.

Avoid copying the text verbatim. Here is the document content:
\"\"\"{chunk}\"\"\"
"""

    def generate_study_notes(self, chunk: str) -> Optional[str]:
        """
        Generate study notes for a text chunk using the LLM.
        
        Args:
            chunk: Text chunk to generate notes for
            
        Returns:
            Generated notes as string, or None if API call fails
        """
        prompt = self.get_prompt_template().format(chunk=chunk)

        data = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            return None

    def generate_notes_for_chunks(self, chunks: list[str]) -> list[str]:
        """
        Generate notes for multiple chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of generated notes for each chunk
        """
        notes = []
        for chunk in chunks:
            result = self.generate_study_notes(chunk)
            if result:
                notes.append(result)
            else:
                notes.append("❌ Error generating notes for this chunk")
        return notes 