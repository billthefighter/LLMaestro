import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, List, Any

from src.llm.chains import (
    ChainContext, ChainStep, SequentialChain, ParallelChain,
    ChordChain, GroupChain, MapChain, ReminderChain, RecursiveChain
)
from src.llm.base import BaseLLMInterface, LLMResponse
from src.prompts.loader import PromptLoader
from src.utils.storage import StorageManager

@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.return_value = LLMResponse(content="test response", metadata={})
    return llm

@pytest.fixture
def mock_prompt_loader():
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    return loader

@pytest.fixture
def mock_storage():
    return Mock(spec=StorageManager)

@pytest.mark.asyncio
async def test_chain_step_execution():
    """Test basic chain step execution."""
    # Setup
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.return_value = LLMResponse(content="test response", metadata={})
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    context = ChainContext(artifacts={}, metadata={}, state={})
    
    # Test basic execution
    step = ChainStep[str]("test_task")
    result = await step.execute(llm, loader, context, test_input="value")
    
    assert isinstance(result, LLMResponse)
    assert result.content == "test response"
    loader.format_prompt.assert_called_once_with("test_task", test_input="value")
    
    # Test with transforms
    def input_transform(context: ChainContext, **kwargs) -> Dict[str, Any]:
        return {"transformed": kwargs["input"]}
        
    def output_transform(response: LLMResponse) -> LLMResponse:
        return LLMResponse(
            content=response.content.upper(),
            metadata=response.metadata
        )
        
    step = ChainStep[LLMResponse](
        "test_task",
        input_transform=input_transform,
        output_transform=output_transform
    )
    result = await step.execute(llm, loader, context, input="value")
    
    assert isinstance(result, LLMResponse)
    assert result.content == "TEST RESPONSE"
    loader.format_prompt.assert_called_with("test_task", transformed="value")

@pytest.mark.asyncio
async def test_sequential_chain():
    """Test sequential chain execution."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="first response", metadata={}),
        LLMResponse(content="second response", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain
    steps = [
        ChainStep[LLMResponse]("first_task"),
        ChainStep[LLMResponse]("second_task")
    ]
    chain = SequentialChain[LLMResponse](
        steps,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(input="test")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "second response"
    assert len(chain.context.artifacts) == 2
    assert all(f"step_{step.id}" in chain.context.artifacts for step in steps)

@pytest.mark.asyncio
async def test_parallel_chain():
    """Test parallel chain execution."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content=f"response {i}", metadata={})
        for i in range(3)
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain
    steps = [ChainStep[LLMResponse]("task") for _ in range(3)]
    chain = ParallelChain[LLMResponse](
        steps,
        max_concurrent=2,  # Test concurrency limiting
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    results = await chain.execute(input="test")
    
    # Verify execution
    assert len(results) == 3
    assert all(isinstance(r, LLMResponse) for r in results)
    assert [r.content for r in results] == ["response 0", "response 1", "response 2"]
    assert len(chain.context.artifacts) == 3

@pytest.mark.asyncio
async def test_chord_chain():
    """Test chord chain execution."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="parallel 1", metadata={}),
        LLMResponse(content="parallel 2", metadata={}),
        LLMResponse(content="callback result", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain
    parallel_steps = [
        ChainStep[LLMResponse]("parallel_task_1"),
        ChainStep[LLMResponse]("parallel_task_2")
    ]
    callback_step = ChainStep[LLMResponse]("callback_task")
    
    chain = ChordChain[LLMResponse](
        parallel_steps,
        callback_step,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(input="test")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "callback result"
    assert "parallel_results" in chain.context.artifacts
    parallel_results = chain.context.artifacts["parallel_results"]
    assert len(parallel_results) == 2
    assert [r.content for r in parallel_results] == ["parallel 1", "parallel 2"]

@pytest.mark.asyncio
async def test_map_chain():
    """Test map chain execution."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content=f"mapped {i}", metadata={})
        for i in range(3)
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create template chain
    template_step = ChainStep[LLMResponse]("map_task")
    template_chain = SequentialChain[LLMResponse](
        [template_step],
        llm=llm,
        prompt_loader=loader
    )
    
    # Create map chain
    chain = MapChain[LLMResponse](
        template_chain,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    inputs = [
        {"input": "first"},
        {"input": "second"},
        {"input": "third"}
    ]
    results = await chain.execute(inputs=inputs)
    
    # Verify execution
    assert len(results) == 3
    assert all(isinstance(r, LLMResponse) for r in results)
    assert [r.content for r in results] == ["mapped 0", "mapped 1", "mapped 2"]
    assert "map_results" in chain.context.artifacts

@pytest.mark.asyncio
async def test_chain_error_handling():
    """Test error handling in chains."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        Exception("API Error"),  # First call fails
        LLMResponse(content="retry success", metadata={})  # Retry succeeds
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain with retry strategy
    step = ChainStep[LLMResponse](
        "test_task",
        retry_strategy={"max_retries": 1, "delay": 0}
    )
    chain = SequentialChain[LLMResponse](
        [step],
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(input="test")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "retry success"
    assert llm.process.call_count == 2  # Original call + retry 

@pytest.mark.asyncio
async def test_reminder_chain():
    """Test reminder chain execution with context injection."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="First analysis: The text discusses AI", metadata={}),
        LLMResponse(content="Second analysis: Focus on neural networks", metadata={}),
        LLMResponse(content="Final analysis: Comprehensive overview of AI and neural networks", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain
    steps = [
        ChainStep[LLMResponse]("analyze_section_1"),
        ChainStep[LLMResponse]("analyze_section_2"),
        ChainStep[LLMResponse]("synthesize")
    ]
    chain = ReminderChain[LLMResponse](
        steps,
        reminder_template="Original task: {initial_prompt}\nContext so far:\n{context}\n\nNow: {current_prompt}",
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    initial_prompt = "Analyze this technical paper and maintain consistent terminology"
    result = await chain.execute(prompt=initial_prompt)
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "Final analysis: Comprehensive overview of AI and neural networks"
    
    # Verify context was maintained
    assert chain.initial_prompt == initial_prompt
    assert len(chain.context_history) == 3
    
    # Verify prompts included reminders
    expected_calls = [
        # First call - just initial prompt
        (("analyze_section_1", ), {"prompt": f"Original task: {initial_prompt}\nContext so far:\n\nNow: {initial_prompt}"}),
        
        # Second call - includes first result
        (("analyze_section_2", ), {
            "prompt": f"Original task: {initial_prompt}\nContext so far:\nStep 1: First analysis: The text discusses AI\nNow: {initial_prompt}",
            "previous_result": chain.context_history[0]
        }),
        
        # Third call - includes both previous results
        (("synthesize", ), {
            "prompt": f"Original task: {initial_prompt}\nContext so far:\nStep 1: First analysis: The text discusses AI\nStep 2: Second analysis: Focus on neural networks\nNow: {initial_prompt}",
            "previous_result": chain.context_history[1]
        })
    ]
    
    assert loader.format_prompt.call_count == 3
    for (expected_args, expected_kwargs), real_call in zip(expected_calls, loader.format_prompt.call_args_list):
        assert real_call[0] == expected_args
        for key, value in expected_kwargs.items():
            assert key in real_call[1]
            assert real_call[1][key] == value

@pytest.mark.asyncio
async def test_reminder_chain_with_transforms():
    """Test reminder chain with custom input transforms."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="First step result", metadata={}),
        LLMResponse(content="Second step result", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create custom transform
    def custom_transform(context: ChainContext, **kwargs) -> Dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "extra_context": "Additional information"
        }
    
    # Create chain
    steps = [
        ChainStep[LLMResponse](
            "step_1",
            input_transform=custom_transform
        ),
        ChainStep[LLMResponse](
            "step_2",
            input_transform=custom_transform
        )
    ]
    chain = ReminderChain[LLMResponse](
        steps,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(prompt="Test prompt")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "Second step result"
    
    # Verify transforms were composed correctly
    for call in loader.format_prompt.call_args_list:
        assert "extra_context" in call[1]
        assert call[1]["extra_context"] == "Additional information"
        assert "prompt" in call[1]
        assert "Remember the original task" in call[1]["prompt"] 

@pytest.mark.asyncio
async def test_reminder_chain_context_pruning():
    """Test reminder chain context pruning functionality."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="First step: Analysis started", metadata={}),
        LLMResponse(content="Second step completed: Found key patterns", metadata={}),
        LLMResponse(content="Third step: Additional analysis", metadata={}),
        LLMResponse(content="Final step completed: Summary of findings", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create chain with pruning enabled
    steps = [
        ChainStep[LLMResponse]("step_1"),
        ChainStep[LLMResponse]("step_2"),
        ChainStep[LLMResponse]("step_3"),
        ChainStep[LLMResponse]("step_4")
    ]
    chain = ReminderChain[LLMResponse](
        steps,
        prune_completed=True,
        completion_markers=["completed"],
        max_context_items=2,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(prompt="Test analysis")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "Final step completed: Summary of findings"
    
    # Verify context pruning
    pruned_context = chain._prune_context()
    assert len(pruned_context) == 2  # Limited by max_context_items
    assert pruned_context[0].content == "First step: Analysis started"
    assert pruned_context[1].content == "Third step: Additional analysis"
    
    # Verify completed steps were removed
    assert all("completed" not in r.content.lower() for r in pruned_context)
    
    # Verify prompts used pruned context
    for call in loader.format_prompt.call_args_list[2:]:  # Check calls after first two steps
        prompt = call[1]["prompt"]
        assert "Second step completed" not in prompt  # Completed step should be pruned
        assert "First step: Analysis started" in prompt  # Non-completed step should remain 

@pytest.mark.asyncio
async def test_recursive_chain():
    """Test recursive chain execution with dynamic step generation."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="initial response", metadata={"needs_followup": True}),
        LLMResponse(content="followup response 1", metadata={}),
        LLMResponse(content="followup response 2", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create step generator
    def step_generator(response: LLMResponse, context: ChainContext) -> List[ChainStep[LLMResponse]]:
        if response.metadata.get("needs_followup"):
            return [
                ChainStep[LLMResponse]("followup_1"),
                ChainStep[LLMResponse]("followup_2")
            ]
        return []
    
    # Create chain
    initial_step = ChainStep[LLMResponse]("initial_task")
    chain = RecursiveChain[LLMResponse](
        initial_step=initial_step,
        step_generator=step_generator,
        max_recursion_depth=3,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(input="test")
    
    # Verify execution
    assert isinstance(result, LLMResponse)
    assert result.content == "followup response 2"  # Last response
    assert len(chain.context.artifacts) == 3  # Initial + 2 followup steps
    assert chain.context.recursion_depth == 0  # Should be reset after execution
    assert not chain.context.recursion_path  # Should be empty after execution

@pytest.mark.asyncio
async def test_recursive_chain_depth_limit():
    """Test recursive chain depth limit enforcement."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.return_value = LLMResponse(content="response", metadata={"needs_followup": True})
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create step generator that always generates new steps
    def infinite_generator(response: LLMResponse, context: ChainContext) -> List[ChainStep[LLMResponse]]:
        return [ChainStep[LLMResponse]("infinite_task")]
    
    # Create chain with low depth limit
    initial_step = ChainStep[LLMResponse]("initial_task")
    chain = RecursiveChain[LLMResponse](
        initial_step=initial_step,
        step_generator=infinite_generator,
        max_recursion_depth=2,  # Set low limit
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain and expect RecursionError
    with pytest.raises(RecursionError) as exc_info:
        await chain.execute(input="test")
    
    # Verify error message contains path
    assert "Maximum recursion depth (2) exceeded" in str(exc_info.value)
    assert "->" in str(exc_info.value)  # Should show chain path

@pytest.mark.asyncio
async def test_recursive_chain_context_tracking():
    """Test recursive chain context maintenance."""
    # Setup mocks
    llm = AsyncMock(spec=BaseLLMInterface)
    llm.process.side_effect = [
        LLMResponse(content="level 1", metadata={"level": 1}),
        LLMResponse(content="level 2", metadata={"level": 2}),
        LLMResponse(content="level 3", metadata={})
    ]
    
    loader = Mock(spec=PromptLoader)
    loader.format_prompt.return_value = ("system prompt", "user prompt")
    
    # Create step generator that tracks depth in context
    def depth_tracking_generator(response: LLMResponse, context: ChainContext) -> List[ChainStep[LLMResponse]]:
        level = response.metadata.get("level", 0)
        if level < 3:
            return [ChainStep[LLMResponse](f"level_{level + 1}_task")]
        return []
    
    # Create chain
    initial_step = ChainStep[LLMResponse]("level_1_task")
    chain = RecursiveChain[LLMResponse](
        initial_step=initial_step,
        step_generator=depth_tracking_generator,
        max_recursion_depth=5,
        llm=llm,
        prompt_loader=loader
    )
    
    # Execute chain
    result = await chain.execute(input="test")
    
    # Verify execution and context tracking
    assert isinstance(result, LLMResponse)
    assert result.content == "level 3"
    assert chain.context.recursion_depth == 0  # Should be reset
    assert not chain.context.recursion_path  # Should be empty
    assert len(chain.context.artifacts) == 3  # One artifact per level
    
    # Verify artifacts contain the expected results
    assert any(v.content == "level 1" for v in chain.context.artifacts.values())
    assert any(v.content == "level 2" for v in chain.context.artifacts.values())
    assert any(v.content == "level 3" for v in chain.context.artifacts.values()) 