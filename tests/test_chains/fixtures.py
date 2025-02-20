"""Fixtures for chain testing."""

import pytest
from datetime import datetime
from typing import Dict, Any, Set, Callable, List, Optional, Union
from uuid import uuid4
from enum import Enum

from llmaestro.agents.agent_pool import AgentPool
from llmaestro.agents.models.models import AgentCapability
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
    AgentChainNode,
    ChainGraph,
    AgentAwareChainGraph,
)
from llmaestro.core.models import Task, SubTask, Artifact, BaseResponse, AgentConfig, TokenUsage
from llmaestro.core.storage import ArtifactStorage
from llmaestro.core.task_manager import TaskManager, DecompositionConfig
from llmaestro.llm.interfaces import BaseLLMInterface, LLMResponse
from llmaestro.llm.models import ModelFamily, ModelDescriptor, ModelCapabilities, ModelRegistry
from llmaestro.prompts.base import BasePrompt
from llmaestro.prompts.loader import PromptLoader
from llmaestro.prompts.types import (
    PromptMetadata,
    VersionInfo,
    ResponseFormat,
)
from llmaestro.llm.interfaces.base import ConversationContext
from llmaestro.llm.token_utils import TokenCounter
from llmaestro.agents.models.config import AgentPoolConfig, AgentTypeConfig, AgentRuntimeConfig
from llmaestro.core.config import (
    UserConfig,
    ConfigurationManager,
    SystemConfig,
    ProviderConfig,
)
from llmaestro.llm.provider_registry import ModelConfig
from llmaestro.core.models import RateLimitConfig


class ChangeType(str, Enum):
    """Change type for version info."""
    NEW = "new"
    UPDATE = "update"
    FIX = "fix"
    REFACTOR = "refactor"


@pytest.fixture
def test_response() -> LLMResponse:
    """Test LLM response."""
    return LLMResponse(
        content="Test response",
        model=ModelDescriptor(name="test", family="test", capabilities=ModelCapabilities()),
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        context_metrics=None,
        success=True,
        metadata={}
    )


@pytest.fixture
def mock_llm(test_response):
    """Mock LLM interface."""
    class MockLLM(BaseLLMInterface):
        def __init__(self, config: AgentConfig, model_registry: Optional[ModelRegistry] = None):
            self.config = config
            self.context = ConversationContext([])
            self._total_tokens = 0
            self._token_counter = TokenCounter()
            self._model_descriptor = ModelDescriptor(
                name="test",
                family="test",
                capabilities=ModelCapabilities()
            )

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
        artifacts={},
        metadata=chain_metadata,
        state=chain_state,
    )


@pytest.fixture
async def chain_step() -> ChainStep:
    """Basic chain step."""
    return ChainStep(
        task=Task(
            id=str(uuid4()),
            type="test_task",
            input_data={"test": "data"},
            config={},
        ),
        retry_strategy=RetryStrategy(),
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
async def agent_chain_node(chain_step, chain_metadata) -> AgentChainNode:
    """Agent chain node."""
    return AgentChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=NodeType.AGENT,
        agent_type=AgentType.GENERAL,
        metadata=chain_metadata,
        required_capabilities={AgentCapability.TEXT},
    )


@pytest.fixture
def test_model_config():
    """Test model configuration."""
    return {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "description": "Test Agent using Claude 3 Sonnet",
        "capabilities": {AgentCapability.TEXT, AgentCapability.CODE},
        "runtime": AgentRuntimeConfig(
            max_tokens=32000,
            temperature=0.7,
            stream=True,
            max_context_tokens=32000
        ),
    }


@pytest.fixture
def mock_config(monkeypatch, test_model_config):
    """Mock configuration manager."""
    class MockConfigManager:
        def __init__(self):
            self._user_config = UserConfig(
                api_keys={"anthropic": "test-key"},
                default_model={
                    "provider": "anthropic",
                    "name": "claude-3-sonnet-20240229",
                },
                agents={
                    "max_agents": 10,
                    "default_agent_type": "general",
                    "agent_types": {
                        "test_task": test_model_config,
                        "general": test_model_config,
                    }
                },
                storage={
                    "path": "test_storage",
                    "format": "json"
                },
                visualization={
                    "enabled": False,
                    "host": "localhost",
                    "port": 8501,
                    "debug": False
                },
                logging={
                    "level": "INFO",
                    "file": "test.log"
                }
            )

            model_config = ModelConfig(
                family="claude",
                context_window=200000,
                typical_speed=100.0,
                features=set([
                    "streaming",
                    "function_calling",
                    "vision",
                    "json_mode",
                    "system_prompt"
                ]),
                cost={
                    "input_per_1k": 0.015,
                    "output_per_1k": 0.075
                }
            )

            provider_config = ProviderConfig(
                api_base="https://api.anthropic.com/v1",
                models={"claude-3-sonnet-20240229": model_config},
                rate_limits={
                    "requests_per_minute": 50,
                    "tokens_per_minute": 100000
                },
                capabilities_detector="llm.models.ModelCapabilitiesDetector._detect_anthropic_capabilities"
            )

            self._system_config = SystemConfig(
                providers={"anthropic": provider_config}
            )

        @property
        def user_config(self) -> UserConfig:
            return self._user_config

        @property
        def system_config(self) -> SystemConfig:
            return self._system_config

        def load_configs(self, *args, **kwargs) -> None:
            """Mock load_configs to do nothing since configs are already set."""
            pass

    # Mock the get_config function
    mock_manager = MockConfigManager()
    monkeypatch.setattr("llmaestro.core.config.get_config", lambda: mock_manager)
    return mock_manager


@pytest.fixture
def agent_pool(test_model_config, mock_config, mock_llm, monkeypatch):
    """Fixture for AgentPool."""
    # First ensure the mock config is set up
    monkeypatch.setattr("llmaestro.core.config.get_config", lambda: mock_config)
    monkeypatch.setattr("llmaestro.agents.agent_pool.get_config", lambda: mock_config)

    # Mock the LLM interface creation at both potential import paths
    monkeypatch.setattr("llmaestro.llm.interfaces.factory.create_llm_interface", lambda config: mock_llm(config))
    monkeypatch.setattr("llmaestro.llm.interfaces.factory.AnthropicLLM", mock_llm)

    config = AgentPoolConfig(
        agent_types={
            "test_task": AgentTypeConfig(**test_model_config),
            "general": AgentTypeConfig(**test_model_config),  # Add default general agent type
        },
        default_agent_type="general"  # Set default agent type
    )
    return AgentPool(config=config)


@pytest.fixture
def task_manager(monkeypatch, agent_pool):
    """Fixture for TaskManager."""
    class MockArtifactStorage(ArtifactStorage):
        def save_artifact(self, artifact: Artifact) -> bool:
            return True

        def load_artifact(self, artifact_id: str) -> Optional[Artifact]:
            return None

        def delete_artifact(self, artifact_id: str) -> bool:
            return True

        def list_artifacts(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Artifact]:
            return []

    manager = TaskManager()
    manager.set_agent_pool(agent_pool)  # Set the agent pool
    manager.storage = MockArtifactStorage()  # Set mock storage
    return manager


@pytest.fixture
def task() -> Task:
    """Test task."""
    return Task(
        id=str(uuid4()),
        type="test_task",
        input_data={"test": "data"},
        config={},
    )


@pytest.fixture
async def chain_graph(chain_context, chain_node, chain_edge, task_manager) -> ChainGraph:
    """Basic chain graph."""
    return ChainGraph(
        id=str(uuid4()),
        nodes={chain_node.id: chain_node},
        edges=[chain_edge],
        context=chain_context,
        task_manager=task_manager,
    )


@pytest.fixture
async def agent_chain_graph(chain_graph, agent_pool, task_manager) -> AgentAwareChainGraph:
    """Agent-aware chain graph."""
    return AgentAwareChainGraph(
        id=chain_graph.id,
        nodes=chain_graph.nodes,
        edges=chain_graph.edges,
        context=chain_graph.context,
        agent_pool=agent_pool,
        task_manager=task_manager,
    )


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
