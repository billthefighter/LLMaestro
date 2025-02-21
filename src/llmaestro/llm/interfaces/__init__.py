"""LLM interface definitions."""

from llmaestro.llm.enums import MediaType

from .base import BaseLLMInterface, LLMResponse

__all__ = ["LLMResponse", "BaseLLMInterface", "MediaType"]
