import asyncio
import random
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from src.core.logging_config import configure_logging
from src.core.config import Config
from src.llm.chains import ChainStep, SequentialChain, ChainContext
from src.llm.interfaces.base import BaseLLMInterface, LLMResponse
from src.core.models import AgentConfig
from src.visualization.live_visualizer import LiveChainVisualizer
from src.llm.interfaces import AnthropicLLM

# Configure module logger
logger = configure_logging(module_name=__name__)

async def simulate_step_execution(step: ChainStep, visualizer: LiveChainVisualizer) -> bool:
    """Simulate step execution with random delay and possible errors."""
    logger.info(f"Executing step: {step.task_type}")
    delay = random.uniform(0.5, 2.0)
    logger.info(f"Simulating work for {delay:.1f} seconds...")
    await asyncio.sleep(delay)
    
    # Randomly simulate errors (10% chance)
    if random.random() < 0.1:
        logger.warning(f"Step {step.task_type} encountered an error!")
        await visualizer.on_step_error(step, Exception("Random error occurred"))
        return False
    
    logger.info(f"Step {step.task_type} completed successfully")
    return True

async def wait_for_input(prompt: str = "Press Enter to continue...") -> None:
    """Wait for user input in an async-friendly way."""
    def _input() -> None:
        input(prompt)
        return None
        
    # Run input in a thread pool since it's blocking
    await asyncio.get_event_loop().run_in_executor(None, _input)

async def main():
    logger.info("Starting visualization demo...")
    
    # Create an event to handle shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()
        print("\nShutting down gracefully... Press Ctrl+C again to force exit.")
    
    # Initialize live visualizer
    logger.info("Initializing live visualizer...")
    visualizer = LiveChainVisualizer()
    await visualizer.start_server()
    logger.info("WebSocket server started successfully")
    logger.info("Open visualization.html in your browser now and press Enter when ready...")
    await wait_for_input()

    try:
        # Set up signal handlers
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        
        # Load configuration and create Claude LLM
        logger.info("Loading configuration...")
        config = Config.load()
        logger.info("Creating Claude LLM client...")
        llm = AnthropicLLM(
            config=AgentConfig(
                provider="anthropic",
                model_name=config.llm.model,
                api_key=config.llm.api_key,
                max_tokens=1024,  # Default max tokens
                max_context_tokens=8192  # Default context window for Claude
            )
        )

        # Create a sample chain for text analysis
        logger.info("Creating sequential chain with 3 steps...")
        chain = SequentialChain(
            steps=[
                ChainStep(
                    task_type="initial_analysis",
                    input_transform=lambda context, **kwargs: {
                        "task": "Analyze the sentiment and key themes in this text: 'The new AI developments have sparked both excitement and concern in the scientific community.'"
                    }
                ),
                ChainStep(
                    task_type="data_processing",
                    input_transform=lambda context, **kwargs: {
                        "task": f"Based on the initial analysis: {context.artifacts.get('step_initial_analysis', 'No previous analysis')}, identify the main stakeholders and their perspectives."
                    }
                ),
                ChainStep(
                    task_type="result_generation",
                    input_transform=lambda context, **kwargs: {
                        "task": f"Synthesize the findings from both analyses: {context.artifacts.get('step_initial_analysis', 'No initial analysis')} and {context.artifacts.get('step_data_processing', 'No data processing')}, into a coherent summary."
                    }
                )
            ],
            llm=llm  # Use the Claude LLM
        )

        # Process each step with live updates
        logger.info("Beginning chain execution...")
        for step in chain.steps:
            logger.info(f"Starting step: {step.task_type}")
            await visualizer.on_step_start(step)
            await wait_for_input(f"Press Enter to execute {step.task_type}...")
            
            success = await simulate_step_execution(step, visualizer)
            if success:
                logger.info(f"Completing step: {step.task_type}")
                await visualizer.on_step_complete(step)
                await wait_for_input(f"Step {step.task_type} completed. Press Enter for next step...")

        logger.info("Chain execution completed")
        
        # Keep server running to view results
        print("Visualization server running. Press Ctrl+C to exit.")
        logger.info("Waiting for shutdown signal...")
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
    finally:
        # Clean up signal handler
        loop = asyncio.get_running_loop()
        loop.remove_signal_handler(signal.SIGINT)
        
        logger.info("Stopping visualization server...")
        await visualizer.stop_server()
        logger.info("Server stopped successfully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Force exit requested. Goodbye!") 