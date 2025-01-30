import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.chains import ChainContext, ChainStep, LLMResponse, RecursiveChain
from ..llm.interfaces import BaseLLMInterface
from ..prompts.loader import PromptLoader
from ..utils.storage import StorageManager


@dataclass
class DevelopmentResult:
    """Result of a development cycle."""

    implementation_changes: List[Dict[str, Any]]
    test_changes: List[Dict[str, Any]]
    documentation_changes: List[Dict[str, Any]]
    bug_fixes: List[Dict[str, Any]]

    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> "DevelopmentResult":
        """Create a DevelopmentResult from an LLM response."""
        try:
            result = json.loads(response.content)
            implementation_changes = []
            if "implementation" in result:
                implementation_changes.extend(result["implementation"].get("files_to_modify", []))
                implementation_changes.extend(result["implementation"].get("new_files", []))

            test_changes = []
            if "tests" in result and "test_files" in result["tests"]:
                test_changes.extend(result["tests"]["test_files"])

            return cls(
                implementation_changes=implementation_changes,
                test_changes=test_changes,
                documentation_changes=[],  # Will be populated later
                bug_fixes=[],  # Will be populated from test results
            )
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from LLM")


def development_transform(response: LLMResponse) -> Any:
    """Transform LLM responses into DevelopmentResult objects."""
    return DevelopmentResult.from_llm_response(response)


class DevelopmentCycleChain(RecursiveChain[DevelopmentResult]):
    """Chain for managing a complete development cycle."""

    def __init__(
        self,
        llm: BaseLLMInterface,
        storage: Optional[StorageManager] = None,
        prompt_loader: Optional[PromptLoader] = None,
        max_iterations: int = 3,
    ):
        # Create output transform instance using the function
        output_transform = development_transform

        # Initial step is the feature implementation
        initial_step = ChainStep[DevelopmentResult](
            task_type="development_cycle",
            input_transform=self._prepare_implementation_input,
            output_transform=output_transform,
        )

        super().__init__(
            initial_step=initial_step,
            step_generator=self._generate_next_steps,
            max_recursion_depth=max_iterations,
            llm=llm,
            storage=storage,
            prompt_loader=prompt_loader,
        )

        self.result = DevelopmentResult(
            implementation_changes=[], test_changes=[], documentation_changes=[], bug_fixes=[]
        )

    def _prepare_implementation_input(self, context: ChainContext, **kwargs) -> Dict[str, Any]:
        """Prepare input for implementation step."""
        return {
            "task_description": kwargs.get("task_description", ""),
            "codebase_context": kwargs.get("codebase_context", ""),
            "requirements": kwargs.get("requirements", ""),
        }

    def _generate_next_steps(
        self, response: DevelopmentResult, context: ChainContext
    ) -> List[ChainStep[DevelopmentResult]]:
        """Generate next steps based on implementation results."""
        next_steps: List[ChainStep[DevelopmentResult]] = []
        output_transform = development_transform

        # Add test execution step if there are test changes
        if response.test_changes:
            next_steps.append(
                ChainStep[DevelopmentResult](
                    task_type="test_execution",
                    input_transform=lambda context, **kwargs: {
                        "test_files": response.test_changes,
                        "implementation_changes": response.implementation_changes,
                    },
                    output_transform=output_transform,
                )
            )

        # Add documentation update step
        next_steps.append(
            ChainStep[DevelopmentResult](
                task_type="documentation_update",
                input_transform=lambda context, **kwargs: {
                    "implementation_changes": response.implementation_changes,
                    "test_changes": response.test_changes,
                    "existing_docs": kwargs.get("existing_docs", ""),
                },
                output_transform=output_transform,
            )
        )

        return next_steps

    async def execute(self, **kwargs) -> DevelopmentResult:
        """Execute the development cycle chain."""
        result = await super().execute(**kwargs)
        self.result = result
        return result


# Example usage
async def run_development_cycle(task_description: str, codebase_context: str, requirements: str, llm: BaseLLMInterface):
    """Run a complete development cycle for a new feature."""
    chain = DevelopmentCycleChain(llm=llm)

    result = await chain.execute(
        task_description=task_description,
        codebase_context=codebase_context,
        requirements=requirements,
    )

    print("Development Cycle Results:")
    print(f"Implementation Changes: {len(result.implementation_changes)}")
    print(f"Test Changes: {len(result.test_changes)}")
    print(f"Documentation Changes: {len(result.documentation_changes)}")
    print(f"Bug Fixes: {len(result.bug_fixes)}")

    return result
