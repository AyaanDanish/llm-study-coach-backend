#!/usr/bin/env python3
"""
Debug script to identify API issues with OpenRouter.
"""

import os
from utils.llm_client import LLMClient


def debug_api_issues():
    """Debug API connection and rate limiting issues."""

    print("🔍 Debugging OpenRouter API Issues...")
    print("=" * 50)

    # Check environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("Please set your API key first.")
        return

    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")

    # Test API connection
    try:
        client = LLMClient()
        print(f"✅ LLM Client initialized")
        print(f"Using model: {client.MODEL}")

        # Test basic connection
        print("\n🧪 Testing API connection...")
        if client.test_api_connection():
            print("✅ Basic API connection works")
        else:
            print("❌ API connection failed")
            return

        # Test with small chunk
        print("\n🧪 Testing note generation with small chunk...")
        test_chunk = """
        This is a small test document for debugging purposes.
        It contains minimal content to test the API response.
        If this fails, there might be an issue with the API or rate limits.
        """

        result = client.generate_study_notes(test_chunk)
        if result:
            print("✅ Note generation successful!")
            print(f"Generated notes length: {len(result)} characters")
            print(f"Preview: {result[:200]}...")
        else:
            print("❌ Note generation failed - check console logs above for details")

    except Exception as e:
        print(f"❌ Error during debugging: {e}")

    print("\n" + "=" * 50)
    print("COMMON ISSUES WITH FREE MODELS:")
    print(
        "1. Rate Limits: Free models have very strict rate limits (often 1 request per minute)"
    )
    print("2. Daily Limits: May have daily token/request limits")
    print("3. Context Size: Large chunks might be rejected")
    print("4. Availability: Free models may have limited availability")
    print("\nSUGGESTIONS:")
    print("- Wait longer between requests (try 60+ seconds)")
    print("- Use smaller chunks for testing")
    print("- Check OpenRouter dashboard for usage limits")
    print("- Consider upgrading to a paid model for testing")


if __name__ == "__main__":
    debug_api_issues()
