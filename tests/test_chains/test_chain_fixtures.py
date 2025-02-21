"""Tests for chain fixtures instantiation."""

import pytest
from uuid import uuid4
from typing import Dict, Any

from llmaestro.chains.chains import (
    NodeType,
    RetryStrategy,
    ChainMetadata,
    ChainState,
    ChainContext,
    ChainNode,
)
from llmaestro.llm.interfaces import LLMResponse

# Remove old fixtures import since we're using conftest.py
# pytest_plugins = ["tests.test_chains.fixtures"]

@pytest.mark.parametrize("model_cls,params", [
    (RetryStrategy, {"max_retries": 3, "delay": 0.1}),
    (ChainMetadata, {"description": "Test", "tags": {"test"}, "version": "1.0"}),
    (ChainState, {"status": "pending"}),
])
def test_basic_models_instantiation(model_cls, params):
    """Test that basic chain models can be instantiated."""
    instance = model_cls(**params)
    assert instance is not None


def test_chain_context_instantiation(chain_context):
    """Test ChainContext instantiation."""
    assert chain_context is not None
    assert chain_context.metadata is not None
    assert chain_context.state is not None
    assert isinstance(chain_context.variables, dict)


@pytest.mark.asyncio
async def test_chain_step_instantiation(chain_step):
    """Test ChainStep instantiation."""
    assert chain_step is not None
    assert chain_step.prompt is not None
    assert chain_step.retry_strategy is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("node_type", [
    NodeType.SEQUENTIAL,
    NodeType.PARALLEL,
    NodeType.CONDITIONAL,
    NodeType.AGENT,
])
async def test_chain_node_instantiation(chain_step, chain_metadata, node_type):
    """Test ChainNode instantiation with different node types."""
    node = ChainNode(
        id=str(uuid4()),
        step=chain_step,
        node_type=node_type,
        metadata=chain_metadata,
    )
    assert node is not None
    assert node.node_type == node_type


def test_chain_edge_instantiation(chain_edge):
    """Test ChainEdge instantiation."""
    assert chain_edge is not None
    assert chain_edge.source_id is not None
    assert chain_edge.target_id is not None
    assert chain_edge.edge_type == "next"


@pytest.mark.asyncio
async def test_chain_graph_instantiation(chain_graph):
    """Test ChainGraph instantiation."""
    assert chain_graph is not None
    assert chain_graph.nodes is not None
    assert chain_graph.edges is not None
    assert chain_graph.context is not None
    assert chain_graph.agent_pool is not None


def test_transform_functions_instantiation(input_transform, output_transform, test_response):
    """Test that transform functions can be called."""
    context = ChainContext(
        metadata=ChainMetadata(),
        state=ChainState(),
        variables={}
    )
    data: Dict[str, Any] = {"test": "data"}

    # Test input transform
    input_result = input_transform(context, data)
    assert input_result is not None
    assert "transformed" in input_result

    # Test output transform
    output_result = output_transform(test_response)
    assert output_result is not None
    assert "processed" in output_result
