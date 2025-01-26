from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Any, Dict, Optional
from ..core.models import SubTask, AgentConfig
from ..llm.base import create_llm_interface, LLMResponse
from ..llm.chains import TaskProcessor
from ..prompts.loader import PromptLoader

class Agent:
    def __init__(self, config: AgentConfig, prompt_loader: Optional[PromptLoader] = None):
        self.config = config
        self.busy = False
        self.llm_interface = create_llm_interface(config)
        self.task_processor = TaskProcessor(self.llm_interface, prompt_loader)
        
    async def process_task(self, subtask: SubTask) -> Any:
        """Process a single subtask using the configured LLM."""
        try:
            if isinstance(subtask.input_data, dict):
                result = await self._process_typed_task(subtask)
            else:
                # Fallback for untyped tasks
                response = await self.llm_interface.process(subtask.input_data)
                result = {
                    "status": "completed",
                    "data": response.content,
                    "metadata": response.metadata
                }
            
            return {
                "status": "completed",
                "data": result,
                "task_id": subtask.id
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "task_id": subtask.id
            }
    
    async def _process_typed_task(self, subtask: SubTask) -> Dict[str, Any]:
        """Process a task using the task processor and prompt templates."""
        if not isinstance(subtask.input_data, dict):
            raise ValueError("Typed tasks require dictionary input data")
            
        # Get the prompt template for this task type
        prompt = self.task_processor.prompt_loader.get_prompt(subtask.type)
        if not prompt:
            raise ValueError(f"No prompt template found for task type: {subtask.type}")
            
        # Format the prompt with input data
        system_prompt, user_prompt = self.task_processor.prompt_loader.format_prompt(
            subtask.type,
            **subtask.input_data
        )
        
        if not system_prompt or not user_prompt:
            raise ValueError(f"Failed to format prompt for task type: {subtask.type}")
            
        # Process with LLM
        response = await self.llm_interface.process(user_prompt, system_prompt=system_prompt)
        
        # Parse response according to expected format
        if prompt.metadata.expected_response.format == "json":
            return self.task_processor._parse_json_response(response)
        else:
            return response.content

class AgentPool:
    def __init__(self, max_agents: int = 5):
        self.max_agents = max_agents
        self.agents: Dict[str, Agent] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.loop = asyncio.get_event_loop()
        
    def submit_task(self, subtask: SubTask) -> None:
        """Submit a subtask to be processed by an available agent."""
        # Get or create an agent for the task
        agent = self._get_or_create_agent(subtask.id)
        
        # Create and store the async task
        task = self.loop.create_task(agent.process_task(subtask))
        self.tasks[subtask.id] = task
        
    def _get_or_create_agent(self, task_id: str) -> Agent:
        """Get an existing agent or create a new one if possible."""
        if task_id in self.agents:
            return self.agents[task_id]
            
        if len(self.agents) < self.max_agents:
            agent = Agent(AgentConfig(
                model_name="gpt-4",
                max_tokens=8192
            ))
            self.agents[task_id] = agent
            return agent
            
        raise RuntimeError("No available agents and cannot create more")
        
    def wait_for_result(self, task_id: str) -> Any:
        """Wait for a specific task to complete and return its result."""
        if task_id not in self.tasks:
            raise ValueError(f"No task found with id: {task_id}")
            
        task = self.tasks[task_id]
        result = self.loop.run_until_complete(task)
        
        # Cleanup
        del self.tasks[task_id]
        if task_id in self.agents:
            del self.agents[task_id]
            
        return result 