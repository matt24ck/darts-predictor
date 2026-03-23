"""Chat module — Claude Haiku AI assistant for darts betting."""

from .context_builder import build_chat_context
from .haiku_client import stream_chat_response

__all__ = ["build_chat_context", "stream_chat_response"]
