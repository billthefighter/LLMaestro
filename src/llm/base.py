from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from dataclasses import dataclass

from litellm import completion
from pydantic import BaseModel

from src.core.models import AgentConfig, TokenUsage, ContextMetrics
from .interfaces.base import BaseLLMInterface
from .interfaces import OpenAIInterface, AnthropicLLM

@dataclass
class ConversationContext:
    """Represents the current conversation context."""
    messages: List[Dict[str, str]]
    summary: Optional[Dict[str, Any]] = None
    initial_task: Optional[str] = None
    message_count: int = 0

class LLMResponse(BaseModel):
    """Standardized response from LLM processing."""
    content: str
    metadata: Dict[str, Any] = {}
    token_usage: Optional[TokenUsage] = None
    context_metrics: Optional[ContextMetrics] = None

    def model_dump(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "token_usage": self.token_usage.model_dump() if self.token_usage else None,
            "context_metrics": self.context_metrics.model_dump() if self.context_metrics else None
        }

    def model_dump_json(self) -> str:
        """Convert the model to a JSON string."""
        return json.dumps(self.model_dump())

class BaseLLMInterface(ABC):
    """Base interface for LLM interactions."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._total_tokens_used = 0
        self._context = ConversationContext(messages=[])
        
    @property
    def total_tokens_used(self) -> int:
        """Get total tokens used by this interface instance."""
        return self._total_tokens_used

    def set_initial_task(self, task: str) -> None:
        """Set the initial task context for reminders."""
        self._context.initial_task = task
        # Add initial task as first message
        self._context.messages.append({
            "role": "system",
            "content": self.config.summarization.reminder_template.format(initial_task=task)
        })

    async def _maybe_add_reminder(self) -> bool:
        """Add a reminder of the initial task if needed."""
        if not self._context.initial_task or not self.config.summarization.reminder_frequency:
            return False

        if (self._context.message_count > 0 and 
            self._context.message_count % self.config.summarization.reminder_frequency == 0):
            
            # Get progress indicators from the latest summary if available
            progress_indicators = {
                "completed_steps": [],
                "in_progress": [],
                "pending": []
            }
            if self._context.summary and "state" in self._context.summary:
                state = self._context.summary["state"]
                if "variables" in state:
                    vars = state["variables"]
                    progress_indicators["completed_steps"] = vars.get("completed_areas", [])
                    progress_indicators["in_progress"] = [vars.get("current_focus")] if vars.get("current_focus") else []
                    progress_indicators["pending"] = vars.get("remaining_tasks", [])
            
            # Call task reminder prompt
            reminder_response = await self.process(
                {
                    "initial_task": self._context.initial_task,
                    "context": self._context.messages[-5:],  # Last 5 messages for recent context
                    "messages_since_reminder": self.config.summarization.reminder_frequency,
                    "progress_indicators": progress_indicators
                },
                system_prompt="You are a task reminder system. Generate a focused reminder of the current task and progress."
            )
            
            if reminder_response.content:
                try:
                    reminder_data = json.loads(reminder_response.content)
                    # Add structured reminder to context
                    self._context.messages.extend([
                        {
                            "role": "system",
                            "content": f"Task Context: {reminder_data['reminder']['task_context']}\nCurrent Focus: {reminder_data['reminder']['current_focus']}"
                        },
                        {
                            "role": "system",
                            "content": f"Progress ({reminder_data['metrics']['completion_estimate']*100:.0f}% complete): {reminder_data['reminder']['progress_summary']}"
                        },
                        {
                            "role": "system",
                            "content": "Next Steps:\n" + "\n".join(f"- {step}" for step in reminder_data['guidance']['next_steps'])
                        }
                    ])
                    return True
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback to simple reminder if parsing fails
                    self._context.messages.append({
                        "role": "system",
                        "content": self.config.summarization.reminder_template.format(
                            initial_task=self._context.initial_task
                        )
                    })
                    return True
            
        return False

    async def _maybe_summarize_context(self, current_metrics: ContextMetrics) -> bool:
        """Check if context needs summarization and perform if necessary."""
        if not self.config.summarization.enabled:
            return False
            
        if (current_metrics.context_utilization >= self.config.summarization.target_utilization and
            current_metrics.current_context_tokens >= self.config.summarization.min_tokens_for_summary):
            
            # Prepare messages for summarization
            preserve_count = self.config.summarization.preserve_last_n_messages
            messages_to_summarize = self._context.messages[:-preserve_count] if preserve_count > 0 else self._context.messages
            preserved_messages = self._context.messages[-preserve_count:] if preserve_count > 0 else []
            
            if not messages_to_summarize:
                return False
                
            # Call summarization
            summary_response = await self.process(
                {
                    "context": messages_to_summarize,
                    "target_tokens": int(self.config.max_context_tokens * 0.3),  # Target 30% of context window
                    "current_utilization": current_metrics.context_utilization
                },
                system_prompt="You are a context summarizer. Summarize the conversation while preserving key information."
            )
            
            if summary_response.content:
                try:
                    summary_data = json.loads(summary_response.content)
                    # Update context with summary and preserved messages
                    self._context = ConversationContext(
                        messages=[
                            {"role": "system", "content": summary_data["summary"]},
                            *[{"role": "system", "content": f"Key Decision: {d}"} for d in summary_data["key_decisions"]],
                            {"role": "system", "content": f"Progress: {summary_data['state']['progress']}"},
                            *preserved_messages
                        ],
                        summary=summary_data,
                        initial_task=self._context.initial_task,
                        message_count=self._context.message_count
                    )
                    return True
                except (json.JSONDecodeError, KeyError) as e:
                    # Log error but continue without summarization
                    print(f"Error summarizing context: {str(e)}")
                    
            return False
        
        return False
        
    def _update_metrics(self, response: Any) -> Tuple[Optional[TokenUsage], Optional[ContextMetrics]]:
        """Update and return token usage and context metrics."""
        usage = getattr(response, 'usage', None)
        if not usage:
            return None, None
            
        # Calculate token usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=None
        )
        
        # Update total tokens
        self._total_tokens_used += token_usage.total_tokens
        
        # Calculate cost if configured
        if self.config.token_tracking and self.config.cost_per_1k_tokens:
            token_usage.estimated_cost = (
                token_usage.total_tokens * self.config.cost_per_1k_tokens / 1000
            )
            
        # Calculate context metrics
        context_metrics = ContextMetrics(
            max_context_tokens=self.config.max_context_tokens,
            current_context_tokens=token_usage.prompt_tokens,
            available_tokens=self.config.max_context_tokens - token_usage.prompt_tokens,
            context_utilization=token_usage.prompt_tokens / self.config.max_context_tokens
        )
            
        return token_usage, context_metrics

    def _format_messages(self, input_data: Any, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format input data and system prompt into messages."""
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add conversation history
        messages.extend(self._context.messages)
        
        # Add the current input
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, dict) and "task" in input_data:
            messages.append({"role": "user", "content": input_data["task"]})
        else:
            messages.append({"role": "user", "content": str(input_data)})
            
        return messages

    async def _handle_response(self, response: Any, messages: List[Dict[str, str]]) -> LLMResponse:
        """Process the LLM response and update context."""
        # Extract token usage and context metrics
        token_usage, context_metrics = self._update_metrics(response)
        
        # Update conversation context
        self._context.messages.append({
            "role": "user",
            "content": messages[-1]["content"]
        })
        self._context.messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        self._context.message_count += 2
        
        # Check if we need to summarize the context
        if context_metrics and await self._maybe_summarize_context(context_metrics):
            # Context was summarized, metrics need to be updated
            token_usage, context_metrics = self._update_metrics(response)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            metadata={
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            },
            token_usage=token_usage,
            context_metrics=context_metrics
        )

    def _handle_error(self, e: Exception) -> LLMResponse:
        """Standardized error handling for LLM requests."""
        error_msg = f"Error processing request: {str(e)}"
        return LLMResponse(
            content=error_msg,
            metadata={"error": str(e)},
            token_usage=None,
            context_metrics=None
        )

    @abstractmethod
    async def process(self, input_data: Any, system_prompt: Optional[str] = None) -> LLMResponse:
        """Process input data and return a response. Must be implemented by subclasses."""
        pass

def create_llm_interface(config: AgentConfig) -> BaseLLMInterface:
    """Create an LLM interface based on the configuration."""
    if config.provider.lower() == "openai":
        return OpenAIInterface(config)
    elif config.provider.lower() == "anthropic":
        return AnthropicLLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}") 