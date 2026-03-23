"""
Claude Haiku client for streaming chat responses.

Uses the Anthropic Python SDK with claude-haiku-4-5-20251001.
"""

import logging
from typing import Generator, List, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_ID = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024


def stream_chat_response(
    api_key: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS,
) -> Generator[str, None, None]:
    """
    Stream a response from Claude Haiku.

    Args:
        api_key: Anthropic API key
        system_prompt: System context (ratings, track record, etc.)
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
        max_tokens: Maximum response tokens

    Yields:
        Text chunks as they arrive
    """
    if not api_key:
        yield "AI chat is not configured. Please set the ANTHROPIC_API_KEY environment variable."
        return

    try:
        import anthropic
    except ImportError:
        yield "The `anthropic` package is not installed. Run: pip install anthropic"
        return

    client = anthropic.Anthropic(api_key=api_key)

    try:
        with client.messages.stream(
            model=MODEL_ID,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    except anthropic.AuthenticationError:
        yield "Invalid API key. Please check your ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        yield "Rate limited. Please try again in a moment."
    except Exception as e:
        logger.error(f"Haiku API error: {e}")
        yield f"Sorry, I encountered an error: {str(e)}"


def get_chat_response(
    api_key: str,
    system_prompt: str,
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Get a non-streaming response from Claude Haiku.

    Useful for testing and simple interactions.
    """
    chunks = list(stream_chat_response(api_key, system_prompt, messages, max_tokens))
    return "".join(chunks)
