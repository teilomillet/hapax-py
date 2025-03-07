# Evaluation Decorators

The evaluation decorator system provides a pytest-like interface for running evaluations on function outputs. This allows you to easily integrate content safety checks into your existing code.

## Relationship with OpenLIT Evaluators

Hapax provides two complementary ways to evaluate content:

1. **Using Evaluation Decorators** (covered in this document) - A lightweight, decorator-based approach for adding evaluations to any Python function
2. **OpenLIT Integration for Graphs** (covered in [Advanced OpenLIT Integration](openlit_integration.md)) - Direct integration with OpenLIT evaluators in graph execution

The `@eval` decorator can work with both:
- Built-in local evaluators
- OpenLIT-based evaluators (when `use_openlit=True`)

## Setup

First, register the evaluators you want to use:

```python
from hapax.core.decorators import register_evaluator
from your_evaluators import CustomEvaluator

# Register built-in evaluators (done automatically)
# - "hallucination"
# - "bias"
# - "toxicity"

# Register custom evaluators
register_evaluator("custom", CustomEvaluator)
```

## Basic Usage

```python
from hapax.core.decorators import eval

@eval(evals=["hallucination"], threshold=0.7)
def generate_response(prompt: str) -> str:
    return f"Response to {prompt}"
```

## Features

### Registry System

The decorator uses a registry system to manage available evaluators. Built-in evaluators are registered automatically:
- `hallucination`: Check for factual accuracy
- `bias`: Detect potential biases
- `toxicity`: Identify harmful content
- `all`: Run all registered evaluators

You can add your own evaluators:
```python
class MyEvaluator:
    def __init__(self, **config):
        self.config = config
    
    def evaluate(self, text: str) -> float:
        # Return score between 0 and 1
        return 0.5

register_evaluator("my_eval", MyEvaluator)
```

### Using with OpenLIT Evaluators

To use the OpenLIT-based evaluators (which use LLMs for evaluation):

```python
@eval(
    evals=["hallucination"],
    threshold=0.7,
    use_openlit=True,  # This enables OpenLIT evaluators
    openlit_provider="openai",  # "openai" or "anthropic"
    metadata={
        "contexts": ["Einstein won the Nobel Prize in Physics in 1921."],
        "prompt": "When did Einstein win the Nobel Prize?"
    }
)
def generate_response(prompt: str) -> str:
    # Your generation code
    return "Einstein won the Nobel Prize in 1922."  # Will fail evaluation
```

### Result Caching

Results are automatically cached based on function inputs:
```python
@eval(evals=["bias"], threshold=0.3, cache_results=True)  # cache_results is True by default
def generate_content(prompt: str) -> str:
    return f"Content for {prompt}"

# First call: evaluates and caches result
result1 = generate_content("test")

# Second call: uses cached result
result2 = generate_content("test")  # faster!
```

### Custom Configuration

Provide additional metadata and evaluation-specific configuration:
```python
@eval(
    evals=["toxicity"],
    threshold=0.5,
    metadata={"domain": "social"},
    openlit_config={"model": "toxicity-v2"},
    cache_results=False  # disable caching if needed
)
def generate_social_post(topic: str) -> str:
    return f"Post about {topic}"
```

### Integration with Operations

Seamlessly combine with the `@ops` decorator for full integration with the Hapax framework:
```python
@ops(name="safe_generate", tags=["nlp", "safe"])
@eval(evals=["all"], threshold=0.6)
def safe_generation(context: str) -> str:
    return f"Safely generated response for: {context}"
```

## Example Output

Here's what happens when running different types of evaluations:

```python
# Basic usage - passes evaluation
result = simple_generation("what is the meaning of life?")
# Output: "The answer to what is the meaning of life? is always 42."

# Multiple evaluations - passes all checks
result = comprehensive_check("Tell me about history")
# Output: "Processing Tell me about history with comprehensive checks."

# Combined with @ops - passes evaluation
result = safe_generation("Generate a story")
# Output: "Safely generated response for: Generate a story"

# Strict threshold - fails bias check
try:
    result = financial_advice("Should I invest in stocks?")
except EvaluationError as e:
    # Output: "Evaluation failed: Evaluations failed: ['bias']. Scores: {'bias': 0.5}"
```

## Error Handling

When an evaluation fails, an `EvaluationError` is raised with detailed information about:
- Which evaluations failed
- The scores for each evaluation
- The threshold that was exceeded

Example error:
```
EvaluationError: Evaluations failed: ['bias']. Scores: {'bias': 0.5}
```

## Best Practices

1. **Register Custom Evaluators Early**
   - Register evaluators at module import time
   - Use descriptive names for evaluators

2. **Choose Appropriate Thresholds**
   - Lower thresholds (0.3-0.5) for strict checking
   - Higher thresholds (0.7-0.9) for more permissive checking

3. **Caching Considerations**
   - Enable caching for expensive evaluations
   - Disable caching for evaluations that should always run
   - Cache is per-process, not persistent

4. **Type Safety**
   - Evaluators only work with string outputs
   - Functions must have type hints
   - Return type must be `str`

## Complete Example

See the [examples directory](../examples/eval_decorator_examples.py) for a complete working example that demonstrates all these features.
