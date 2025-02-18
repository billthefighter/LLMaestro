"""LLM interface definitions."""

from llmaestro.llm.models import MediaType
from .base import LLMResponse, BaseLLMInterface

__all__ = ["LLMResponse", "BaseLLMInterface", "MediaType"]
