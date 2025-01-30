# Task Decomposition

The LLM Orchestrator supports both static and dynamic task decomposition strategies. This allows tasks to be broken down into subtasks that can be processed in parallel, with results aggregated back together.

## Built-in Strategies

The following static decomposition strategies are available:

### Chunk Strategy
Splits input data into chunks of specified size:
```yaml
metadata:
  decomposition:
    strategy: "chunk"
    chunk_size: 1000  # characters for text, items for lists
    max_parallel: 5
    aggregation: "concatenate"  # or "merge" for structured data
```

### File Strategy
Processes multiple files in parallel:
```yaml
metadata:
  decomposition:
    strategy: "file"
    max_parallel: 10
    aggregation: "merge"
```

### Error Strategy
Handles multiple errors or issues in parallel:
```yaml
metadata:
  decomposition:
    strategy: "error"
    max_parallel: 5
    aggregation: "merge"
```

## Dynamic Decomposition

For tasks that require custom decomposition logic, you can use the dynamic decomposition feature:

```yaml
metadata:
  decomposition:
    strategy: "custom"
    max_parallel: 5
    aggregation: "custom"
```

When a task uses the "custom" strategy, the system will:

1. Use the `task_decomposer` prompt to analyze the task
2. Generate Python code for decomposition and aggregation
3. Cache the strategy for future use with the same task type

### Task Decomposer

The task decomposer generates strategies based on:
- Task type and description
- Input data format
- Expected output format
- Model requirements

Example response format:
```json
{
  "strategy": {
    "name": "unique_strategy_name",
    "description": "Strategy description",
    "max_parallel": 5
  },
  "decomposition": {
    "method": "Python code for decomposition",
    "aggregation": "Python code for aggregation"
  },
  "validation": {
    "input_requirements": ["requirement1", "requirement2"],
    "output_format": "Expected output format"
  }
}
```

### Caching

Dynamic strategies are cached by task type to avoid regenerating the same strategy multiple times. The cache persists for the lifetime of the TaskManager instance.

### Error Handling

The system includes robust error handling for:
- Invalid strategy generation
- Code execution errors
- Input validation
- Result aggregation

## Best Practices

1. **Choose the Right Strategy**
   - Use "chunk" for large text or list processing
   - Use "file" for multi-file operations
   - Use "error" for parallel error fixing
   - Use "custom" for complex decomposition needs

2. **Configure Parallel Processing**
   - Set appropriate `max_parallel` based on task complexity
   - Consider resource constraints
   - Balance parallelism with result coherence

3. **Handle Results Properly**
   - Choose appropriate aggregation strategy
   - Validate aggregated results
   - Consider partial success scenarios

4. **Monitor and Optimize**
   - Track strategy performance
   - Adjust chunk sizes and parallel limits
   - Cache frequently used strategies

## Examples of Custom Strategies

### 1. Document Clustering
This strategy processes large document collections by first generating embeddings, then clustering and summarizing:

```python
def decompose_document_clustering(task: Task) -> List[SubTask]:
    documents = task.input_data
    if not isinstance(documents, list):
        raise ValueError("Input must be a list of documents")

    # Phase 1: Generate embeddings in batches
    embedding_tasks = []
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embedding_tasks.append(
            SubTask(
                id=str(uuid.uuid4()),
                type="embedding_generation",
                input_data={"documents": batch},
                parent_task_id=task.id
            )
        )

    return embedding_tasks

def aggregate_document_clustering_results(results: List[Any]) -> Dict[str, Any]:
    # Combine embeddings and cluster documents
    all_embeddings = []
    for result in results:
        if isinstance(result, dict) and "embeddings" in result:
            all_embeddings.extend(result["embeddings"])

    # Perform clustering and create summaries
    clusters = {
        "clusters": [
            {
                "id": cluster_id,
                "documents": docs,
                "summary": summary,
                "keywords": keywords
            }
            for cluster_id, docs, summary, keywords in cluster_documents(all_embeddings)
        ],
        "metadata": {
            "num_clusters": len(clusters),
            "total_documents": len(all_embeddings)
        }
    }
    return clusters
```

### 2. Code Review
This strategy analyzes code by splitting it into logical components and reviewing each separately:

```python
def decompose_code_review(task: Task) -> List[SubTask]:
    code_files = task.input_data
    if not isinstance(code_files, dict):
        raise ValueError("Input must be a dictionary of file paths to content")

    subtasks = []

    # Group related files
    file_groups = group_related_files(code_files)

    for group_name, files in file_groups.items():
        subtasks.append(
            SubTask(
                id=str(uuid.uuid4()),
                type="component_review",
                input_data={
                    "group": group_name,
                    "files": files,
                    "context": task.config.get("review_context", {})
                },
                parent_task_id=task.id
            )
        )

    return subtasks

def aggregate_code_review_results(results: List[Any]) -> Dict[str, Any]:
    aggregated = {
        "components": [],
        "issues": {
            "critical": [],
            "major": [],
            "minor": []
        },
        "suggestions": [],
        "metrics": {
            "files_reviewed": 0,
            "lines_analyzed": 0
        }
    }

    for result in results:
        if isinstance(result, dict):
            # Add component review
            aggregated["components"].append({
                "name": result.get("group", "unknown"),
                "summary": result.get("summary", ""),
                "issues": result.get("issues", [])
            })

            # Categorize issues
            for issue in result.get("issues", []):
                severity = issue.get("severity", "minor")
                aggregated["issues"][severity].append(issue)

            # Add suggestions
            aggregated["suggestions"].extend(result.get("suggestions", []))

            # Update metrics
            aggregated["metrics"]["files_reviewed"] += result.get("files_reviewed", 0)
            aggregated["metrics"]["lines_analyzed"] += result.get("lines_analyzed", 0)

    return aggregated
```

### 3. Data Pipeline Validation
This strategy validates complex data pipelines by testing different components in parallel:

```python
def decompose_pipeline_validation(task: Task) -> List[SubTask]:
    pipeline_config = task.input_data
    if not isinstance(pipeline_config, dict):
        raise ValueError("Input must be a pipeline configuration dictionary")

    subtasks = []

    # Test data sources
    for source in pipeline_config.get("sources", []):
        subtasks.append(
            SubTask(
                id=str(uuid.uuid4()),
                type="source_validation",
                input_data={
                    "source": source,
                    "validation_rules": task.config.get("source_rules", {})
                },
                parent_task_id=task.id
            )
        )

    # Test transformations
    for transform in pipeline_config.get("transformations", []):
        subtasks.append(
            SubTask(
                id=str(uuid.uuid4()),
                type="transform_validation",
                input_data={
                    "transform": transform,
                    "test_data": task.config.get("test_data", {}),
                    "validation_rules": task.config.get("transform_rules", {})
                },
                parent_task_id=task.id
            )
        )

    # Test outputs
    for output in pipeline_config.get("outputs", []):
        subtasks.append(
            SubTask(
                id=str(uuid.uuid4()),
                type="output_validation",
                input_data={
                    "output": output,
                    "validation_rules": task.config.get("output_rules", {})
                },
                parent_task_id=task.id
            )
        )

    return subtasks

def aggregate_pipeline_validation_results(results: List[Any]) -> Dict[str, Any]:
    validation_report = {
        "status": "passed",
        "components": {
            "sources": [],
            "transformations": [],
            "outputs": []
        },
        "issues": [],
        "metrics": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        }
    }

    for result in results:
        if isinstance(result, dict):
            component_type = result.get("type")
            if component_type in validation_report["components"]:
                validation_report["components"][component_type].append({
                    "name": result.get("name"),
                    "status": result.get("status"),
                    "tests": result.get("tests", []),
                    "issues": result.get("issues", [])
                })

            # Update metrics
            validation_report["metrics"]["total_tests"] += result.get("total_tests", 0)
            validation_report["metrics"]["passed_tests"] += result.get("passed_tests", 0)
            validation_report["metrics"]["failed_tests"] += result.get("failed_tests", 0)

            # Collect issues
            validation_report["issues"].extend(result.get("issues", []))

    # Update overall status
    if validation_report["metrics"]["failed_tests"] > 0:
        validation_report["status"] = "failed"

    return validation_report
```

Each example demonstrates different aspects of custom decomposition:

1. **Document Clustering**
   - Multi-phase processing with batching
   - Complex result aggregation with clustering
   - Metadata tracking

2. **Code Review**
   - Logical component grouping
   - Hierarchical issue categorization
   - Metric collection

3. **Data Pipeline Validation**
   - Component-based decomposition
   - Comprehensive validation reporting
   - Status aggregation with metrics

These examples can be adapted and combined to create strategies for other complex tasks. Remember to:
- Include proper error handling
- Validate input data
- Track progress and metrics
- Structure results consistently
