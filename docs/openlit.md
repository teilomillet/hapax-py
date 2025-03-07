# Basic OpenLit Integration in Hapax

Hapax is built on top of OpenLit to provide comprehensive monitoring and observability for your graph operations. This document covers the basic integration with OpenLit. For advanced features such as GPU monitoring and evaluations, see the [Advanced OpenLit Integration](openlit_integration.md).

## Quick Setup

The simplest way to use OpenLit with Hapax is to initialize it once at the start of your application:

```python
import openlit
import hapax

# Initialize OpenLit with default settings
openlit.init(otlp_endpoint="http://127.0.0.1:4318")

# Now all your Hapax operations will be automatically monitored
# The operation name will be the function name ("tokenize")
@ops
def tokenize(text: str) -> List[str]:
    return text.split()

# Create a graph - the graph name will be the function name ("process_text")
@graph
def process_text(text: str) -> List[str]:
    return tokenize
```

That's it! This will give you basic monitoring and observability for all your Hapax operations.

## Basic Configuration

### 1. Global Configuration
Use environment variables or the init function for application-wide settings:
```python
openlit.init(
    otlp_endpoint="http://localhost:4318",
    environment="development",
    application_name="my_nlp_app",
    trace_content=True
)
```

### 2. Operation-Specific Configuration
Add custom monitoring settings for specific operations:
```python
# You can optionally override the operation name and add more settings
@ops(
    name="custom_tokenizer",  # Optional: override the function name
    tags=["nlp"],
    openlit_config={
        "trace_content": True,
        "disable_metrics": False
    }
)
def tokenize(text: str) -> List[str]:
    return text.split()

# Graphs can also have custom names and metadata
@graph(
    name="text_pipeline",  # Optional: override the function name
    description="Process text using NLP operations"
)
def process_text(text: str) -> List[str]:
    return tokenize
```

## Basic Monitoring Features

### Automatic Monitoring
- Execution time tracking
- Success/failure rates
- Input/output type validation
- Operation dependencies

### Graph Monitoring
- Graph execution flow visualization
- Operation dependencies tracking
- Intermediate results monitoring
- Graph validation status

## Best Practices

1. **Start Simple**: 
   - Begin with just `@ops` and `@graph` without any parameters
   - The function name will be used automatically for monitoring
   - Add custom configurations only when needed

2. **Operation Naming**: 
   - By default, use descriptive function names as they'll be used in traces
   - Override names only if you need a different name in monitoring

3. **Tagging**: 
   - Add tags only when you need to filter or organize operations
   - Tags are optional and can be added later

4. **Error Handling**: 
   - Let OpenLit handle error tracking by using proper exception handling
   - All exceptions are automatically captured and traced

## Example

Here's a complete example showing both simple and advanced usage:

```python
import openlit
from hapax import ops, graph
from typing import List, Dict

# Global initialization - this is all you need to start
openlit.init(otlp_endpoint="http://localhost:4318")

# Simple operation - uses function name
@ops
def tokenize(text: str) -> List[str]:
    return text.split()

# Advanced operation with custom name and settings
@ops(
    name="word_counter",  # Optional custom name
    tags=["nlp", "analysis"],
    openlit_config={"trace_content": True}
)
def count_words(tokens: List[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(tokens))

# Simple graph - uses function name
@graph
def analyze_text(text: str) -> Dict[str, int]:
    return tokenize >> count_words

# All operations are automatically monitored
result = analyze_text("Hello world! Hello Hapax!")
```

## Advanced Features

For more advanced OpenLIT features, including:
- GPU Monitoring
- LLM Evaluations (hallucination, bias, toxicity)
- Custom Evaluation Providers

See the [Advanced OpenLit Integration](openlit_integration.md) documentation.

For more information about OpenLit, visit the [OpenLIT documentation](https://docs.openlit.dev).
