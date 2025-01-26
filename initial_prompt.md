You are a large language model with a limited token size. You are asked to accomplish a task that vastly exceeds your token limit. For example:

- Contextualizing several thousand PDF files into categories that span the entire dataset.
- Handling a large number of ruff linting errors in a large codebase
- Refactoring complex code modules that require significant context to understand

You can use the following tools to help you accomplish the task:

- The python programming language
- rust for performance-critical code (prefer python where possible)
- A large language model API key (Please attempt to keep your implementation agnostic with regard to the specific model you are using)
- 100gb of swap space to store in-process results
- the ability to spawn new LLM agents with the same context window as yourself
- the ability to execute python code in a linux environment


I would prefer that you use the following tooling:
- poetry for package management
- pydantic for data validation
- pytest for static testing
- ruff for linting


Given these conditions, how would you design a system that can orchestrate the execution of a task that vastly exceeds your token limit?

Open questions:

How are you going to handle keeping track of the state of the task?
How are you going to encode the input data small enough to fit in your context window while still being able to process it?

Later functionality:
- Docker image to run the code in
- React or Angular frontend to interact with the system