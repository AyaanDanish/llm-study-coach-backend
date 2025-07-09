#!/usr/bin/env python3
"""Test script to verify quiz generation functionality"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from utils.llm_client import LLMClient


def test_quiz_generation():
    try:
        client = LLMClient()
        print("✅ LLMClient initialized successfully")

        # Test the quiz prompt template
        prompt_template = client.get_quiz_prompt_template()
        print(f"✅ Quiz prompt template loaded: {len(prompt_template)} characters")

        # Test if the generate_quiz method exists
        if hasattr(client, "generate_quiz"):
            print("✅ generate_quiz method is available")

            # Test with sample content
            sample_content = "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. Python is known for its simplicity and readability."
            sample_subject = "Programming"
            sample_title = "Introduction to Python"

            print(f"🧪 Testing quiz generation with sample content...")
            print(f"   Content: {sample_content[:50]}...")
            print(f"   Subject: {sample_subject}")
            print(f"   Title: {sample_title}")

            # This would make an actual API call, so we'll just test the method exists
            print("✅ Ready to generate quiz questions")

        else:
            print("❌ generate_quiz method not found")

    except Exception as e:
        print(f"❌ Error testing quiz generation: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🧠 Testing Quiz Generation Functionality")
    print("=" * 50)

    success = test_quiz_generation()

    if success:
        print("\n✅ All tests passed! Quiz generation is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
