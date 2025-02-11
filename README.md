# Hapax: Type-Safe Graph Execution Framework

Hapax is a powerful Python framework for building type-safe, observable data processing pipelines. Built on top of OpenLit, it provides multi-stage type checking, rich error messages, and comprehensive monitoring out of the box.

## Features

✨ **Multi-Stage Type Safety**
- Import-time type validation through `@ops` decorator
- Definition-time type checking when building graphs
- Runtime type validation during execution
- Rich error messages that pinpoint issues

🔍 **Static Analysis**
- Graph structure validation
- Cycle detection
- Type compatibility verification
- Configuration and metadata checks

📊 **OpenLit Integration**
- Automatic monitoring and observability
- Execution time tracking
- Success/failure rates
- Graph visualization

🎮 **Intuitive API**
- Fluent interface for building pipelines
- Type-safe operation composition using `>>`
- Rich control flow (branch, merge, condition, loop)

## Quick Start

1. Install Hapax:
```bash
pip install hapax
```

2. Create your first pipeline:
```python
from hapax import ops, graph
import openlit
from typing import List, Dict

# Initialize OpenLit (optional but recommended)
openlit.init(otlp_endpoint="http://127.0.0.1:4318")

# Define operations - type checked at import time
@ops(name="clean_text")
def clean_text(text: str) -> str:
    return text.lower().strip()

@ops(name="tokenize")
def tokenize(text: str) -> List[str]:
    return text.split()

@ops(name="analyze")
def analyze(tokens: List[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(tokens))

# Build pipeline - type compatibility checked at definition time
pipeline = (
    Graph("text_processing")
    .then(clean_text)  # str -> str
    .then(tokenize)    # str -> List[str]
    .then(analyze)     # List[str] -> Dict[str, int]
)

# Execute pipeline - types checked at runtime
result = pipeline.execute("Hello World! Hello Hapax!")
```

## Core Concepts

### 1. Operations

Operations are pure functions with multi-stage type checking:

```python
@ops(name="summarize", tags=["nlp"])
def summarize(text: str) -> str:
    """Generate a concise summary."""
    return summary

# Type checking happens at:
# 1. Import time - through @ops decorator
# 2. Definition time - when used in a graph
# 3. Runtime - during execution
result = summarize(42)  # Runtime TypeError: Expected str, got int
```

### 2. Graph Building

Build complex pipelines with immediate type validation:

```python
# Using the fluent API - type compatibility checked at definition time
pipeline = (
    Graph("text_analysis")
    .then(clean_text)      # str -> str
    .branch(
        summarize,         # str -> str
        sentiment_analysis # str -> float
    )
    .merge(combine_results)
)

# Or using the >> operator for composition
pipeline = clean_text >> tokenize >> analyze  # Type compatibility checked immediately
```

### 3. Control Flow

Rich control flow operations with type safety:

```python
# Parallel Processing
pipeline = (
    Graph("parallel_nlp")
    .branch(
        summarize,          # Branch 1: str -> str
        extract_entities,   # Branch 2: str -> List[str]
        analyze_sentiment   # Branch 3: str -> float
    )
    .merge(lambda results: {
        "summary": results[0],
        "entities": results[1],
        "sentiment": results[2]
    })
)

# Conditional Logic
pipeline = (
    Graph("smart_translate")
    .then(detect_language)
    .condition(
        lambda lang: lang != "en",
        translate_to_english,  # If true
        lambda x: x           # If false (pass through)
    )
)
```

## OpenLit Integration

Hapax is built on OpenLit for automatic monitoring:

```python
# 1. Basic Setup
import openlit
openlit.init(otlp_endpoint="http://localhost:4318")

# 2. Operation-Level Monitoring
@ops(
    name="tokenize",
    tags=["nlp"],
    openlit_config={
        "trace_content": True,
        "disable_metrics": False
    }
)
def tokenize(text: str) -> List[str]:
    return text.split()

# 3. Graph-Level Monitoring
@graph(
    name="nlp_pipeline",
    description="Process text using NLP"
)
def process_text(text: str) -> Dict[str, Any]:
    return clean >> analyze
```

## Error Handling

Hapax provides clear error messages:

```python
# Type Mismatch
TypeError: Cannot compose operations: output type List[str] does not match input type Dict[str, Any]

# Structural Issues
GraphValidationError: Graph contains cycles: [['op1', 'op2', 'op1']]

# Runtime Errors
BranchError: Errors in branches: [('sentiment', ValueError('Invalid input'))]
```

## Best Practices

1. **Type Safety**
   - Always specify input and output types
   - Let Hapax handle type validation
   - Use mypy for additional static checking

2. **Operation Design**
   - Keep operations pure and focused
   - Use meaningful names
   - Add proper documentation

3. **Monitoring**
   - Initialize OpenLit early
   - Add meaningful tags
   - Use trace_content for debugging

4. **Error Handling**
   - Handle branch errors appropriately
   - Check partial results in case of failures
   - Use the rich error information

## Documentation

For more detailed information, check out:
- [Comprehensive Guide](docs/guide.md)
- [Graph API Reference](docs/graph.md)
- [OpenLit Integration](docs/openlit.md)

## License

MIT License - see [LICENSE](LICENSE) for details.