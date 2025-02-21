"""Chain testing configuration and fixtures."""

import pytest
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Union
from uuid import uuid4
from enum import Enum

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.chains.chains import (
    NodeType,
    AgentType,
    RetryStrategy,
    ChainMetadata,
    ChainState,
    ChainContext,
    ChainStep,
    ChainNode,
    ChainEdge,
    ChainGraph,
)
from llmaestro.core.models import TokenUsage
from llmaestro.llm.interfaces import BaseLLMInterface, LLMResponse
from llmaestro.llm.models import ModelFamily, LLMProfile, LLMCapabilities
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from llmaestro.prompts.types import (
    PromptMetadata,
    VersionInfo,
    ResponseFormat,
)
from llmaestro.llm.interfaces.base import ConversationContext
from llmaestro.llm.token_utils import TokenCounter


class ChangeType(str, Enum):
    """Change type for version info."""
    NEW = "new"
    UPDATE = "update"
    FIX = "fix"
    REFACTOR = "refactor"


@pytest.fixture
def test_response(mock_LLMProfile) -> LLMResponse:
    """Test LLM response using the mock model descriptor from root conftest."""
    return LLMResponse(
        content="Test response",
        model=mock_LLMProfile,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        context_metrics=None,
        success=True,
        metadata={}
    )


@pytest.fixture
def mock_llm(test_response, config_manager):
    """Mock LLM interface using configuration from root conftest."""
    class MockLLM(BaseLLMInterface):
        def __init__(self, config, llm_registry=None):
            self.config = config
            self.context = ConversationContext([])
            self._total_tokens = 0
            self._token_counter = TokenCounter()
            self._model_descriptor = test_response.model

        @property
        def model_family(self) -> ModelFamily:
            return ModelFamily.CLAUDE

        async def process(self, prompt: Union[BasePrompt, str], variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
            return test_response

        async def process_async(self, prompt: Union[BasePrompt, str], variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
            return test_response

        async def batch_process(
            self,
            prompts: List[Union[BasePrompt, str]],
            variables: Optional[List[Optional[Dict[str, Any]]]] = None
        ) -> List[LLMResponse]:
            return [test_response for _ in prompts]

    return MockLLM


@pytest.fixture
def prompt() -> BasePrompt:
    """Test prompt."""
    class TestPrompt(BasePrompt):
        def render(self, **kwargs) -> tuple[str, str, list]:
            return "system prompt", "user prompt", []

        async def save(self) -> bool:
            return True

        @classmethod
        async def load(cls, identifier: str) -> Optional[BasePrompt]:
            return cls(
                name="test_prompt",
                description="Test prompt",
                system_prompt="Test system prompt",
                user_prompt="Test user prompt",
                metadata=PromptMetadata(
                    type="test",
                    expected_response=ResponseFormat(
                        format="json",
                        schema='{"type": "object"}'
                    ),
                ),
                current_version=VersionInfo(
                    number="1.0.0",
                    author="test",
                    timestamp=datetime.now(),
                    description="Initial version",
                    change_type=ChangeType.NEW,
                ),
            )

    return TestPrompt(
        name="test_prompt",
        description="Test prompt",
        system_prompt="Test system prompt",
        user_prompt="Test user prompt",
        metadata=PromptMetadata(
            type="test",
            expected_response=ResponseFormat(
                format="json",
                schema='{"type": "object"}'
            ),
        ),
        current_version=VersionInfo(
            number="1.0.0",
            author="test",
            timestamp=datetime.now(),
            description="Initial version",
            change_type=ChangeType.NEW,
        ),
    )


@pytest.fixture
def prompt_loader(monkeypatch, prompt) -> PromptLoader:
    """PromptLoader with monkeypatched load_prompt method."""
    async def mock_load_prompt(*args, **kwargs) -> BasePrompt:
        return prompt

    loader = PromptLoader()
    monkeypatch.setattr(loader, "load_prompt", mock_load_prompt)
    return loader


@pytest.fixture
def retry_strategy() -> RetryStrategy:
    """Basic retry strategy."""
    return RetryStrategy(max_retries=3, delay=0.1)


@pytest.fixture
def chain_metadata() -> ChainMetadata:
    """Basic chain metadata."""
    return ChainMetadata(
        description="Test chain",
        tags={"test"},
        version="1.0.0",
    )


@pytest.fixture
def chain_state() -> ChainState:
    """Basic chain state."""
    return ChainState(
        status="pending",
        current_step=None,
        completed_steps=set(),
        failed_steps=set(),
        step_results={},
        variables={},
    )


@pytest.fixture
def chain_context(chain_metadata, chain_state) -> ChainContext:
    """Chain context with basic configuration."""
    return ChainContext(
        metadata=chain_metadata,
        state=chain_state,
        variables={}
    )


@pytest.fixture
async def chain_step(prompt) -> ChainStep:
    """Basic chain step."""
    return await ChainStep.create(
        prompt=prompt,
        retry_strategy=RetryStrategy()
    )


@pytest.fixture
async def chain_node(chain_step, chain_metadata) -> ChainNode:
    """Basic chain node."""
    return ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_metadata,
    )


@pytest.fixture
def chain_edge(chain_node) -> ChainEdge:
    """Basic chain edge."""
    target_node = ChainNode(
        id=str(uuid4()),
        step=chain_node.step,
        node_type=NodeType.SEQUENTIAL,
        metadata=chain_node.metadata,
    )
    return ChainEdge(
        source_id=chain_node.id,
        target_id=target_node.id,
        edge_type="next",
    )


@pytest.fixture
def agent_pool(config_manager, mock_llm, monkeypatch):
    """Fixture for AgentPool using base configuration from root conftest."""
    monkeypatch.setattr("llmaestro.core.config.get_config", lambda: config_manager)
    monkeypatch.setattr("llmaestro.agents.agent_pool.get_config", lambda: config_manager)
    monkeypatch.setattr("llmaestro.llm.interfaces.factory.create_llm_interface", lambda config: mock_llm(config))
    monkeypatch.setattr("llmaestro.llm.interfaces.factory.AnthropicLLM", mock_llm)

    return AgentPool(config=config_manager.user_config.agents)


@pytest.fixture
def chain_graph(chain_context, chain_node, chain_edge, agent_pool) -> ChainGraph:
    """Basic chain graph."""
    graph = ChainGraph(
        id=str(uuid4()),
        nodes={chain_node.id: chain_node},
        edges=[chain_edge],
        context=chain_context,
        agent_pool=agent_pool
    )
    return graph


@pytest.fixture
def input_transform() -> Callable[[ChainContext, Dict[str, Any]], Dict[str, Any]]:
    """Sample input transform function."""
    def transform(context: ChainContext, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"transformed": data}
    return transform


@pytest.fixture
def output_transform() -> Callable[[LLMResponse], Dict[str, Any]]:
    """Sample output transform function."""
    def transform(response: LLMResponse) -> Dict[str, Any]:
        return {"processed": response.content}
    return transform
