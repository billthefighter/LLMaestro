"""Tests for Session class functionality."""

import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from llmaestro.session.session import Session
from llmaestro.core.models import BaseResponse, LLMResponse, TokenUsage
from llmaestro.prompts.base import BasePrompt
from llmaestro.llm.models import ModelDescriptor, ModelCapabilities
from llmaestro.prompts.types import PromptMetadata, ResponseFormat, VersionInfo
from llmaestro.llm.rate_limiter import RateLimitConfig
