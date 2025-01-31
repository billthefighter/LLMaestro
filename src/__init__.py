"""LLM Orchestrator package."""

__version__ = "0.1.0"


def get_exports():
    """Get package exports lazily to avoid circular imports."""
    from .agents.agent_pool import Agent, AgentPool
    from .core.models import AgentConfig, StorageConfig, SubTask, Task, TaskStatus
    from .core.task_manager import TaskManager
    from .prompts.loader import PromptLoader
    from .utils.storage import StorageManager

    return {
        "Task": Task,
        "TaskStatus": TaskStatus,
        "SubTask": SubTask,
        "AgentConfig": AgentConfig,
        "StorageConfig": StorageConfig,
        "TaskManager": TaskManager,
        "Agent": Agent,
        "AgentPool": AgentPool,
        "StorageManager": StorageManager,
        "PromptLoader": PromptLoader,
    }


__all__ = list(get_exports().keys())
