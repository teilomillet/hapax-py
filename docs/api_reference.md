# API Reference

This document provides a comprehensive reference for Hapax's public API.

## Core API

### Decorators

#### `@ops`

Creates a type-safe operation from a function.

```python
from hapax import ops

@ops(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None
)
def my_operation(input_data: InputType) -> OutputType:
    # Implementation
    return output
```

**Parameters:**
- `name`: Optional name of the operation (defaults to function name)
- `description`: Optional description
- `tags`: Optional list of tags for categorization
- `metadata`: Optional dictionary of metadata
- `openlit_config`: Optional OpenLIT configuration

**Returns:**
- An `Operation` instance

#### `@graph`

Creates a graph from a function that returns a composition of operations.

```python
from hapax import graph

@graph(
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
)
def my_graph(input_data: InputType) -> OutputType:
    return op1 >> op2 >> op3
```

**Parameters:**
- `name`: Optional name of the graph (defaults to function name)
- `description`: Optional description
- `metadata`: Optional dictionary of metadata

**Returns:**
- A `Graph` instance

#### `@eval`

Adds evaluation capabilities to a function.

```python
from hapax import eval

@eval(
    evals: List[str] = ["all"],
    threshold: float = 0.5,
    metadata: Dict[str, Any] = {},
    openlit_config: Dict[str, Any] = {},
    cache_results: bool = True,
    use_openlit: bool = False,
    openlit_provider: Optional[str] = None
)
def generate_content(input_data: InputType) -> str:
    # Implementation
    return output
```

**Parameters:**
- `evals`: List of evaluators to use
- `threshold`: Threshold score for evaluation (0.0 to 1.0)
- `metadata`: Additional metadata for evaluation
- `openlit_config`: OpenLIT configuration
- `cache_results`: Whether to cache evaluation results
- `use_openlit`: Whether to use OpenLIT evaluators
- `openlit_provider`: Provider for OpenLIT evaluations

**Returns:**
- The decorated function with evaluation capabilities

### Classes

#### `Operation`

Represents a type-safe operation in a graph.

```python
from hapax import Operation

# Usually created via @ops decorator
operation = Operation(
    func: Callable[[T], U],
    config: OpConfig
)
```

**Methods:**
- `__call__(input_data: T) -> U`: Execute the operation
- `compose(other: Operation[U, V]) -> Operation[T, V]`: Compose with another operation
- `__rshift__(other: Operation[U, V]) -> Operation[T, V]`: Operator version of compose

#### `Graph`

Represents a computational graph.

```python
from hapax import Graph

# Create a graph
graph = Graph(
    name: str,
    description: Optional[str] = None
)

# Fluent interface
graph.then(operation)
graph.branch(op1, op2, op3)
graph.merge(merger_func)
graph.condition(predicate, true_branch, false_branch)
graph.loop(operation, condition, max_iterations)
```

**Methods:**
- `execute(input_data: Any) -> Any`: Execute the graph
- `validate() -> None`: Validate the graph
- `visualize(filename: Optional[str] = None) -> None`: Generate a visualization
- `then(operation: Operation) -> Graph`: Add a sequential operation
- `branch(*operations: Operation) -> Graph`: Add parallel operations
- `merge(merger: Callable[[List[Any]], Any]) -> Graph`: Merge branch results
- `condition(predicate: Callable[[Any], bool], true_branch: Operation, false_branch: Operation) -> Graph`: Add conditional logic
- `loop(operation: Operation, condition: Callable[[Any], bool], max_iterations: Optional[int] = None) -> Graph`: Add a loop
- `with_gpu_monitoring(enabled: bool = True, sample_rate_seconds: int = 1, custom_config: Optional[Dict[str, Any]] = None) -> Graph`: Enable GPU monitoring
- `with_evaluation(eval_type: str = "all", threshold: float = 0.5, provider: str = "openai", fail_on_evaluation: bool = False, model: Optional[str] = None, api_key: Optional[str] = None, custom_config: Optional[Dict[str, Any]] = None) -> Graph`: Add evaluation

#### `OpConfig`

Configuration for an operation.

```python
from hapax import OpConfig

config = OpConfig(
    name: str,
    description: Optional[str] = None,
    tags: List[str] = [],
    metadata: Dict[str, Any] = {},
    openlit_config: Optional[Dict[str, Any]] = None
)
```

### Flow Control

#### `Branch`

Creates parallel execution branches.

```python
from hapax.core.flow import Branch

branch = Branch(name: str)
branch.add(*operations: Operation)
```

#### `Merge`

Merges results from parallel branches.

```python
from hapax.core.flow import Merge

merge = Merge(
    name: str,
    merger: Callable[[List[Any]], Any]
)
```

#### `Condition`

Adds conditional logic to a graph.

```python
from hapax.core.flow import Condition

condition = Condition(
    name: str,
    predicate: Callable[[Any], bool],
    true_branch: Operation,
    false_branch: Operation
)
```

#### `Loop`

Repeats an operation until a condition is met.

```python
from hapax.core.flow import Loop

loop = Loop(
    name: str,
    operation: Operation,
    condition: Callable[[Any], bool],
    max_iterations: Optional[int] = None
)
```

## Monitoring & Evaluations API

### OpenLIT Configuration

```python
from hapax import set_openlit_config, get_openlit_config

# Set global OpenLIT configuration
set_openlit_config({
    "otlp_endpoint": "http://localhost:4318",
    "environment": "development",
    "application_name": "my_app",
    "trace_content": True,
    "disable_metrics": False
})

# Get current configuration
current_config = get_openlit_config()
```

### GPU Monitoring

```python
from hapax import enable_gpu_monitoring, get_gpu_metrics

# Enable GPU monitoring
enable_gpu_monitoring(
    otlp_endpoint: Optional[str] = None,
    sample_rate_seconds: int = 1,
    application_name: Optional[str] = None,
    environment: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
)

# Get current GPU metrics
metrics = get_gpu_metrics()
```

### Evaluators

```python
from hapax import (
    OpenLITEvaluator,
    HallucinationEvaluator,
    BiasEvaluator,
    ToxicityEvaluator,
    AllEvaluator
)

# Create an evaluator
evaluator = HallucinationEvaluator(
    provider: str = "openai",
    threshold: float = 0.5,
    collect_metrics: bool = True,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
)

# Evaluate text
result = evaluator.evaluate(
    text: str,
    contexts: List[str],
    prompt: Optional[str] = None
)
```

## OpenLIT and Observability Architecture

Hapax provides comprehensive observability through deep integration with OpenLIT (Open Lineage In Tracing), which is built on top of the OpenTelemetry standard. This allows for detailed monitoring, tracing, and metrics collection at all levels of the execution stack.

### Observability Integration

Hapax's observability architecture has three key components:

1. **Automatic Instrumentation**: All operations and graphs are automatically traced
2. **Hierarchical Spans**: Traces capture the full execution tree with proper parent-child relationships
3. **Rich Context**: Input/output data, types, and errors are captured in spans

```
┌─────────────────────────────────────────────────────┐
│                  Hapax Application                  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               OpenLIT Instrumentation               │
│                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │ Trace Data  │    │   Metrics   │    │   Logs   │ │
│  └──────┬──────┘    └──────┬──────┘    └────┬─────┘ │
└─────────┼─────────────────┼─────────────────┼───────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│               OpenTelemetry Protocol                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│            OpenTelemetry Collector/Backend          │
│  (Jaeger, Zipkin, Prometheus, Grafana, etc.)        │
└─────────────────────────────────────────────────────┘
```

### Automatic Tracing Implementation

Hapax creates OpenTelemetry spans at multiple levels:

1. **Graph-Level Spans**: Each graph execution creates a root span
2. **Node-Level Spans**: Each operation creates a child span
3. **Branch/Merge Spans**: Flow control operations create their own spans

Implementation in Hapax's core:

```python
# Inside Graph.execute method
with self._tracer.start_as_current_span(
    name=f"graph.{self.name}",
    attributes={
        "graph.name": self.name,
        "graph.num_operations": len(self.operations),
        "graph.num_edges": len(self.edges)
    }
) as graph_span:
    # Execute operations with child spans
    for node_name in nx.topological_sort(self._graph):
        with self._tracer.start_as_current_span(
            name=f"node.{node_name}",
            attributes={"node.type": "operation"}
        ) as node_span:
            # Operation execution...
```

### OpenLIT Configuration

Hapax provides a global OpenLIT configuration that applies to all operations:

```python
import openlit
from hapax import set_openlit_config

# Initialize OpenLIT
openlit.init(
    otlp_endpoint="http://localhost:4318",
    environment="development",
    application_name="my_app",
    trace_content=True
)

# Configure Hapax to use OpenLIT
set_openlit_config({
    "trace_content": True,       # Trace operation inputs/outputs
    "disable_metrics": False,    # Enable metrics collection
    "log_level": "INFO"          # Set logging level
})
```

This configuration can also be set at the operation level through the `@ops` decorator:

```python
@ops(
    name="process_text",
    openlit_config={
        "trace_content": True,
        "disable_metrics": False
    }
)
def process_text(text: str) -> str:
    return text.upper()
```

### What Gets Traced

By default, Hapax traces the following:

1. **Operation Execution**:
   - Start and end times
   - Success/failure status
   - Error details if applicable

2. **Typed Data** (when `trace_content=True`):
   - Input values and types
   - Output values and types
   - Type validation results

3. **Graph Structure**:
   - Operation dependencies
   - Branch/merge relationships
   - Execution order

4. **Performance Metrics**:
   - Execution time
   - Memory usage (when enabled)
   - GPU utilization (when GPU monitoring is enabled)

### Practical Example

Here's a complete example showing how to create a graph with OpenLIT monitoring:

```python
import os
from typing import List, Dict
import openlit
from hapax import ops, Graph, set_openlit_config

# Initialize OpenLIT
openlit.init(
    otlp_endpoint=os.getenv("OPENLIT_ENDPOINT", "http://localhost:4318"),
    environment="development",
    application_name="text_processor"
)

# Configure global OpenLIT settings for Hapax
set_openlit_config({
    "trace_content": True,
    "disable_metrics": False
})

# Define operations - automatically traced
@ops(name="tokenize", tags=["preprocessing"])
def tokenize(text: str) -> List[str]:
    return text.split()

@ops(name="count_words", tags=["analysis"])
def count_words(tokens: List[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(tokens))

# Build pipeline with automatic tracing
pipeline = (
    Graph("text_analysis")
    .then(tokenize)    # Each operation will have its own span
    .then(count_words)
)

# Execute pipeline - traced from start to finish
result = pipeline.execute("Hello world! Hello Hapax!")
```

When this code executes, it generates spans for:
- The `text_analysis` graph execution
- The `tokenize` operation 
- The `count_words` operation
- Each with proper parent-child relationships and timing data

### Integration with External Monitoring Systems

The traces and metrics can be consumed by any OpenTelemetry-compatible system:

1. **Tracing Systems**:
   - Jaeger
   - Zipkin
   - Datadog APM
   - New Relic

2. **Metrics Systems**:
   - Prometheus
   - Grafana
   - InfluxDB
   - Datadog Metrics

3. **Logging Systems**:
   - ELK Stack
   - Loki
   - Cloud Logging (AWS CloudWatch, GCP Logging)

### Advanced Configuration

For more advanced control over OpenLIT and tracing:

```python
# Per-operation custom trace content
@ops(
    name="process_sensitive_data",
    openlit_config={
        "trace_content": False,  # Don't trace sensitive data
        "trace_metadata": True,  # Still trace metadata
        "sampling_rate": 0.1     # Only trace 10% of executions
    }
)
def process_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    # Process sensitive data
    return processed

# Custom span attributes
@ops(
    name="important_operation",
    openlit_config={
        "span_attributes": {
            "service.version": "1.2.3",
            "priority": "high"
        }
    }
)
def important_operation(data: Any) -> Any:
    # Important processing
    return result
```

## Exceptions

### `EvaluationError`

Raised when evaluations fail.

```python
try:
    result = function_with_eval(input_data)
except EvaluationError as e:
    print(f"Failed evaluations: {e.failed_evals}")
    print(f"Scores: {e.scores}")
```

### `GraphValidationError`

Raised when graph validation fails.

```python
try:
    graph.validate()
except GraphValidationError as e:
    print(f"Validation error: {e}")
```

### `GraphExecutionError`

Raised when graph execution fails.

```python
try:
    result = graph.execute(input_data)
except GraphExecutionError as e:
    print(f"Failed at node: {e.node_name}")
    print(f"Node errors: {e.node_errors}")
    print(f"Partial results: {e.partial_results}")
```

### `BranchError`

Raised when errors occur in parallel branches.

```python
try:
    result = branch_operation(input_data)
except BranchError as e:
    print(f"Branch errors: {e.branch_errors}")
    print(f"Partial results: {e.partial_results}")
```

## Utility Functions

### Register Custom Evaluators

```python
from hapax.core.decorators import register_evaluator

class MyEvaluator:
    def __init__(self, **config):
        self.config = config
    
    def evaluate(self, text: str) -> float:
        # Return score between 0 and 1
        return 0.5

register_evaluator("my_evaluator", MyEvaluator)
``` 