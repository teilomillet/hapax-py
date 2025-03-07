# Hapax Documentation

Hapax is a type-safe graph execution framework built on top of OpenLit. It allows you to build composable, monitored data processing pipelines with strong type checking and built-in observability.

## Documentation Structure

This documentation is organized into the following sections:

1. **Getting Started**
   - [Installation Guide](installation.md) - Requirements and installation options
   - [Quick Start Guide](quickstart.md) - Get up and running in under 5 minutes
   - [Comprehensive Guide](guide.md) - In-depth explanation of all key concepts
   - [Examples](examples.md) - Code examples demonstrating various features

2. **Core Features**
   - [Graph API Reference](graph.md) - Complete reference for the Graph API and building pipelines
   - [API Reference](api_reference.md) - Comprehensive reference for all Hapax classes and functions

3. **Integrations**
   - [OpenLit Basics](openlit.md) - Simple monitoring setup with OpenLit
   - [Advanced OpenLit Integration](openlit_integration.md) - GPU monitoring, evaluations, and advanced features

4. **Advanced Features**
   - [Evaluation Decorators](evaluation_decorators.md) - Using evaluation decorators for content safety

5. **Support and Troubleshooting**
   - [Troubleshooting Guide](troubleshooting.md) - Solutions for common issues

## Core Concepts

### Operations

Operations are the basic building blocks in Hapax. An operation is a pure function that transforms data from one type to another:

```python
from hapax import ops
from typing import List

@ops
def tokenize(text: str) -> List[str]:
    return text.split()
```

Key features:
- Type-safe: Input and output types are checked at runtime
- Composable: Operations can be chained together using `>>`
- Auto-monitored: Built-in OpenLit integration for observability
- Pure functions: Each operation should be deterministic and side-effect free

### Graphs

Graphs represent a composition of operations that form a data processing pipeline:

```python
from hapax import graph

@graph
def process_text(text: str) -> List[str]:
    return tokenize >> normalize >> remove_stops
```

Key features:
- Visual representation of the pipeline
- Type compatibility checking between operations
- Cycle detection
- Execution monitoring

## Getting Started

1. Install Hapax:
```bash
pip install hapax
```

2. Initialize OpenLit (optional but recommended):
```python
import openlit
openlit.init(otlp_endpoint="http://127.0.0.1:4318")
```

3. Create your first pipeline:
```python
from hapax import ops, graph
from typing import List, Dict

@ops
def tokenize(text: str) -> List[str]:
    return text.lower().split()

@ops
def count_words(tokens: List[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(tokens))

@graph
def analyze_text(text: str) -> Dict[str, int]:
    return tokenize >> count_words

# Use the pipeline
result = analyze_text("Hello world! Hello Hapax!")
```

## Advanced Features

### 1. Operation Configuration

Operations can be configured with additional metadata:

```python
@ops(
    description="Split text into tokens",
    tags=["nlp", "preprocessing"],
    metadata={"version": "1.0.0"},
    openlit_config={"trace_content": True}
)
def tokenize(text: str) -> List[str]:
    return text.split()
```

### 2. Graph Visualization

Graphs can be visualized to understand the pipeline:

```python
@graph
def my_pipeline(text: str) -> Dict[str, int]:
    return tokenize >> normalize >> count_words

# Generate a visualization
my_pipeline.visualize()
```

### 3. Type Safety

Hapax enforces type safety between operations:

```python
@ops
def tokenize(text: str) -> List[str]:
    return text.split()

@ops
def count_words(tokens: List[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(tokens))

# This works - types match
pipeline = tokenize >> count_words

# This would raise a TypeError - types don't match
# pipeline = count_words >> tokenize
```

### 4. Monitoring and Observability

Hapax integrates with OpenLit for comprehensive monitoring:

- Execution time tracking
- Success/failure rates
- Input/output validation
- Error tracking
- Custom metrics

See the [OpenLit Integration Guide](openlit.md) for details.

## Flow Control

Hapax supports complex graph structures through flow control operators:

### 1. Branching

Execute multiple operations in parallel:

```python
from hapax import ops, graph
from hapax.core.flow import Branch, Merge

@ops
def tokenize(text: str) -> List[str]:
    return text.split()

@ops
def sentiment_analysis(text: str) -> float:
    # Return sentiment score
    return 0.8

@ops
def language_detection(text: str) -> str:
    return "en"

@graph
def analyze_text(text: str) -> Dict[str, Any]:
    # Create parallel branches
    branch = Branch("text_analysis")
    branch.add(
        tokenize,
        sentiment_analysis,
        language_detection
    )
    
    # Merge results
    merge = Merge("combine_results", lambda results: {
        "tokens": results[0],
        "sentiment": results[1],
        "language": results[2]
    })
    
    return branch >> merge

# Use the graph
result = analyze_text("Great product, highly recommend!")
# {
#   "tokens": ["Great", "product", "highly", "recommend"],
#   "sentiment": 0.8,
#   "language": "en"
# }
```

### 2. Conditional Processing

Branch based on conditions:

```python
from hapax.core.flow import Condition

@graph
def process_text(text: str) -> List[str]:
    # Branch based on language
    language_branch = Condition(
        "language_check",
        lambda x: detect_language(x) == "en",
        english_pipeline,
        other_languages_pipeline
    )
    return language_branch

# Process text based on language
result = process_text("Hello world")  # Uses english_pipeline
result = process_text("Bonjour monde")  # Uses other_languages_pipeline
```

### 3. Loops and Retries

Repeat operations until a condition is met:

```python
from hapax.core.flow import Loop

@ops
def api_call(data: Dict) -> Response:
    # Make API request
    return response

@graph
def reliable_api(data: Dict) -> Response:
    # Retry API call up to 3 times
    retry_loop = Loop(
        "retry_api",
        api_call,
        condition=lambda r: r.status_code == 200,
        max_iterations=3
    )
    return retry_loop
```

### 4. Complex Graph Structures

Combine multiple flow operators:

```python
@graph
def complex_pipeline(data: Any) -> Any:
    # Split processing into branches
    process_branch = Branch("processing")
    process_branch.add(
        preprocess_pipeline,
        validation_pipeline
    )
    
    # Merge and validate results
    merge = Merge("validation_merge", 
        lambda results: all(results)  # Check all validations pass
    )
    
    # Conditional processing based on validation
    process_condition = Condition(
        "validation_check",
        lambda x: x,  # Check merged validation result
        main_pipeline,
        error_pipeline
    )
    
    return process_branch >> merge >> process_condition
```

## Error Handling

Hapax provides rich error information through specialized exceptions:

1. **BranchError**: Contains information about which branches failed and partial results
2. **MergeError**: Details about merge operation failures
3. **ConditionError**: Information about predicate evaluation failures
4. **LoopError**: Details about loop termination and iteration count
5. **GraphExecutionError**: Comprehensive error information including:
   - Failed node name
   - All node errors
   - Partial results up to the failure point

Example error handling:

```python
try:
    result = complex_pipeline(data)
except GraphExecutionError as e:
    print(f"Failed at node: {e.node_name}")
    print(f"Errors: {e.node_errors}")
    print(f"Partial results: {e.partial_results}")
```

## Visualization

The `visualize()` method now shows the complete graph structure including:
- Flow control operators
- Branch points
- Merge points
- Conditional paths
- Loop structures

```python
complex_pipeline.visualize()
```

## Best Practices

1. **Keep Operations Pure**
   - Operations should be deterministic
   - Avoid side effects
   - Use immutable data structures when possible

2. **Type Safety**
   - Always specify input and output types
   - Use type hints consistently
   - Let Hapax handle type validation

3. **Naming and Documentation**
   - Operations and graphs use function names by default
   - Add descriptions for complex operations
   - Use tags for categorization
   - Include version info in metadata

4. **Monitoring**
   - Initialize OpenLit early in your application
   - Use tags for better metric organization
   - Monitor both operations and graphs
   - Set appropriate logging levels

## API Reference

### @ops Decorator

Creates an operation from a function.

Parameters:
- `name` (Optional[str]): Operation name (defaults to function name)
- `description` (Optional[str]): Operation description
- `tags` (Optional[List[str]]): Tags for categorization
- `metadata` (Optional[Dict[str, Any]]): Additional metadata
- `openlit_config` (Optional[Dict[str, Any]]): OpenLit configuration

### @graph Decorator

Creates a graph from a function that composes operations.

Parameters:
- `name` (Optional[str]): Graph name (defaults to function name)
- `description` (Optional[str]): Graph description
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

### Operation Class

Methods:
- `__call__(input_data: T) -> U`: Execute the operation
- `compose(other: Operation[U, V]) -> Operation[T, V]`: Compose with another operation
- `visualize()`: Generate a visual representation

### Graph Class

Methods:
- `execute(input_data: Any) -> Any`: Execute the graph
- `validate()`: Validate the graph structure
- `visualize()`: Generate a visualization

## Examples

See the [examples directory](examples/) for more examples, including:
- Text processing pipelines
- Data transformation workflows
- Machine learning preprocessing
- ETL pipelines

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up the development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## License

Hapax is licensed under the MIT License. See [LICENSE](LICENSE) for details.
