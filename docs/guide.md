# Comprehensive Guide to Hapax

Hapax is a powerful type-safe graph execution framework that enables you to build composable, monitored data processing pipelines with strong type checking and built-in observability. This guide will explain all key concepts and their relationships.

## About This Guide

This guide provides an in-depth explanation of Hapax's core concepts and features. It's designed to help you understand the framework's architecture and design principles.

For more specific documentation:
- [Quick Start Guide](quickstart.md) - For getting started quickly
- [Graph API Reference](graph.md) - For detailed information on building graphs
- [OpenLit Integration](openlit.md) - For monitoring and observability features
- [Advanced OpenLIT Integration](openlit_integration.md) - For GPU monitoring and evaluations
- [Evaluation Decorators](evaluation_decorators.md) - For content safety features

## Core Concepts and Their Differences

### Graph vs. graph vs. Graph API

1. `Graph` (Class) - The fundamental class in `hapax.core.models` that represents a computation graph:
   - Handles the internal graph representation using NetworkX
   - Manages operations, edges, and metadata
   - Provides validation and execution functionality
   - Used internally by the framework

2. `graph` (Decorator) - A decorator in `hapax.core.decorators` for creating graphs from functions:
   ```python
   @graph(name="text_processing")
   def process_text(text: str) -> List[str]:
       return tokenize >> normalize >> filter_stops
   ```
   - Simplifies graph creation through function composition
   - Automatically handles type checking
   - Provides a more declarative way to build pipelines

3. Graph API (Fluent Interface) - The high-level builder API in `hapax.core.graph`:
   ```python
   pipeline = (
       Graph("text_analysis")
       .then(clean_text)
       .branch(
           summarize,
           analyze_sentiment
       )
       .merge(combine_results)
   )
   ```
   - Provides a fluent interface for building complex pipelines
   - Offers rich control flow operations (branch, merge, condition, loop)
   - More explicit and programmatic way to build pipelines

### Graph vs. Pipeline

While these terms are sometimes used interchangeably, they have distinct meanings in Hapax:

1. Graph:
   - The underlying data structure that represents operations and their connections
   - Focuses on the structure and relationships between operations
   - Handles validation, type checking, and execution

2. Pipeline:
   - A specific instance of a graph configured for a particular data processing task
   - Represents the actual workflow from input to output
   - Usually created using either the `@graph` decorator or Graph API

Example of the same logic expressed both ways:
```python
# As a pipeline using @graph decorator
@graph(name="text_analysis")
def text_pipeline(text: str) -> Dict[str, Any]:
    return clean_text >> tokenize >> analyze

# As a graph using Graph API
graph = (
    Graph("text_analysis")
    .then(clean_text)
    .then(tokenize)
    .then(analyze)
)
```

## Building Blocks

### 1. Operations

Operations are the fundamental building blocks in Hapax. They are pure functions that transform data from one type to another:

```python
@ops(name="tokenize", tags=["nlp"])
def tokenize(text: str) -> List[str]:
    return text.split()
```

Key features:
- Type-safe: Input and output types are checked at runtime
- Pure functions: Should be deterministic and side-effect free
- Composable: Can be chained using the `>>` operator
- Auto-monitored: Built-in OpenLit integration for observability

### 2. Control Flow Operations

Hapax provides several specialized operations for complex control flow:

1. Branch:
   - Executes multiple operations in parallel
   - Input is passed to all branches
   - Results are collected in a list
   ```python
   graph.branch(
       summarize,          # Branch 1
       analyze_sentiment,  # Branch 2
       extract_entities    # Branch 3
   )
   ```

2. Merge:
   - Combines results from multiple branches
   - Takes a function to specify how to combine results
   ```python
   graph.merge(lambda results: {
       "summary": results[0],
       "sentiment": results[1],
       "entities": results[2]
   })
   ```

3. Condition:
   - Adds conditional branching logic
   - Takes a predicate and true/false operations
   ```python
   graph.condition(
       lambda x: len(x) > 100,
       summarize,        # If true
       lambda x: x      # If false (pass through)
   )
   ```

4. Loop:
   - Repeats an operation until a condition is met
   - Optional maximum iterations
   ```python
   graph.loop(
       process_chunk,
       condition=lambda x: x.has_more_data,
       max_iterations=10
   )
   ```

## Best Practices

## Type Safety and Validation

One of Hapax's most powerful features is its comprehensive graph-wide validation system. Like Flyte, Hapax helps you fail fast by automatically validating your entire pipeline:

1. Automatic Graph-Wide Validation:
   - The entire graph structure is validated automatically
   - Happens during graph construction and before any execution
   - Type compatibility is checked between all connected operations
   - Cycles are detected (except in intentional Loop operations)
   - All required metadata and configurations are verified

2. When Validation Happens:
   - During graph construction (when using `>>` or `.then()`)
   - Automatically before any execution starts
   - During individual operation execution for runtime type safety
   - No need to call validate() manually - Hapax handles it for you

3. Early Error Detection:
   ```python
   # Type mismatch in the pipeline - caught immediately
   @ops
   def tokenize(text: str) -> List[str]:
       return text.split()
   
   @ops
   def count_words(text: str) -> Dict[str, int]:  # Wrong input type!
       from collections import Counter
       return dict(Counter(text.split()))
   
   # Validation happens automatically during construction
   pipeline = tokenize >> count_words
   # TypeError: Cannot compose operations: output type List[str] does not match input type str
   
   # Same automatic validation in the fluent API
   graph = (
       Graph("text_analysis")
       .then(tokenize)
       .then(count_words)  # Fails here with clear type mismatch error
   )
   ```

4. Clear Error Messages:
   - Type mismatches show both expected and received types
   - Cycle detection shows the exact path of the cycle
   - Validation errors include the specific nodes involved
   - Execution errors show partial results and error chain

5. Best Practices:
   - Always specify type hints in your operations
   - Use `mypy` for additional static type checking
   - Write unit tests for your operations
   - Let Hapax handle graph validation automatically

## Operation Design:
   - Keep operations pure and deterministic
   - Make operations focused and single-purpose
   - Use meaningful names and add descriptions

## Pipeline Structure:
   - Break complex pipelines into smaller, reusable components
   - Use branching for parallel processing when possible
   - Add proper error handling and validation

## Monitoring:
   - Configure OpenLit for observability
   - Add meaningful tags and metadata
   - Use trace_content for debugging

## Examples

Here's a complete example that showcases various Hapax features:

```python
from hapax import Graph, ops
from typing import Dict, List, Any

# Define operations
@ops(name="clean_text", tags=["preprocessing"])
def clean_text(text: str) -> str:
    return text.lower().strip()

@ops(name="tokenize", tags=["nlp"])
def tokenize(text: str) -> List[str]:
    return text.split()

@ops(name="analyze", tags=["nlp"])
def analyze(tokens: List[str]) -> Dict[str, Any]:
    return {
        "count": len(tokens),
        "unique": len(set(tokens))
    }

# Build pipeline using Graph API
pipeline = (
    Graph("text_analysis", "Analyzes text properties")
    .then(clean_text)
    .branch(
        tokenize >> analyze,  # Branch 1: Token analysis
        lambda x: len(x)      # Branch 2: Character count
    )
    .merge(lambda results: {
        "analysis": results[0],
        "char_count": results[1]
    })
)

# Use the pipeline
result = pipeline("Hello World!")
```

## Conclusion

Hapax provides a powerful and flexible framework for building data processing pipelines. By understanding the differences between its core concepts and following best practices, you can create robust, maintainable, and observable data processing workflows.

For more information:
- Check the examples directory for real-world use cases
- Read the API documentation for detailed reference
- Visit the OpenLit documentation for monitoring capabilities
