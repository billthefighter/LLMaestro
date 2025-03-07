from typing import Any, Dict, List, Optional

from llmaestro.core.models import LLMResponse
from llmaestro.llm.chains import (
    ChainContext,
    ChainStep,
    ChordChain,
    GroupChain,
    MapChain,
    ParallelChain,
    RecursiveChain,
    ReminderChain,
    SequentialChain,
)
from llmaestro.llm.interfaces import BaseLLMInterface
from llmaestro.llm.models import ModelFamily, TokenUsage
from llmaestro.prompts.base import BasePrompt


class MockLLMInterface(BaseLLMInterface):
    """Mock LLM interface for examples."""

    @property
    def model_family(self) -> ModelFamily:
        return ModelFamily.MOCK

    async def process(self, prompt: BasePrompt, variables: Optional[Dict[str, Any]] = None) -> LLMResponse:
        """Mock process method that returns a fixed response."""
        return LLMResponse(
            content="Mock response",
            success=True,
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model="mock-model",
            model_params={},
            finish_reason="stop",
        )


def create_example_chains():
    """Create example chains of each type for visualization."""
    llm = MockLLMInterface()

    # Helper to create a basic step
    def make_step(name: str) -> ChainStep:
        return ChainStep(task_type=name)

    # 1. Sequential Chain
    sequential = SequentialChain(
        steps=[
            make_step("analyze_input"),
            make_step("process_data"),
            make_step("generate_summary"),
        ],
        llm=llm,
    )

    # 2. Parallel Chain
    parallel = ParallelChain(
        steps=[
            make_step("analyze_sentiment"),
            make_step("extract_keywords"),
            make_step("identify_topics"),
        ],
        max_concurrent=2,
        llm=llm,
    )

    # 3. Chord Chain
    chord = ChordChain(
        parallel_steps=[
            make_step("research_topic_1"),
            make_step("research_topic_2"),
            make_step("research_topic_3"),
        ],
        callback_step=make_step("synthesize_research"),
        max_concurrent=2,
        llm=llm,
    )

    # 4. Group Chain
    group = GroupChain(
        chains=[
            SequentialChain(steps=[make_step("task_1a"), make_step("task_1b")], llm=llm),
            SequentialChain(steps=[make_step("task_2a"), make_step("task_2b")], llm=llm),
        ],
        llm=llm,
    )

    # 5. Map Chain
    template_chain = SequentialChain(steps=[make_step("process_item"), make_step("validate_result")], llm=llm)
    map_chain = MapChain(chain_template=template_chain, llm=llm)

    # 6. Reminder Chain
    reminder = ReminderChain(
        steps=[
            make_step("initial_analysis"),
            make_step("follow_up_questions"),
            make_step("final_synthesis"),
        ],
        max_context_items=2,
        llm=llm,
    )

    # 7. Recursive Chain
    def step_generator(response: LLMResponse, context: ChainContext) -> List[ChainStep]:
        # For demonstration, always return one more step
        if context.recursion_depth < 2:
            return [make_step(f"recursive_step_{context.recursion_depth + 1}")]
        return []

    recursive = RecursiveChain(
        initial_step=make_step("start_recursion"),
        step_generator=step_generator,
        max_recursion_depth=3,
        llm=llm,
    )

    # 8. Complex Nested Chain
    nested = GroupChain(
        chains=[
            ChordChain(
                parallel_steps=[make_step("parallel_task_1"), make_step("parallel_task_2")],
                callback_step=make_step("combine_results"),
                llm=llm,
            ),
            ReminderChain(steps=[make_step("nested_step_1"), make_step("nested_step_2")], llm=llm),
        ],
        llm=llm,
    )

    return {
        "sequential": sequential,
        "parallel": parallel,
        "chord": chord,
        "group": group,
        "map": map_chain,
        "reminder": reminder,
        "recursive": recursive,
        "nested": nested,
    }
