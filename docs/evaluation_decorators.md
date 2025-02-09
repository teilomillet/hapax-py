# Evaluation Decorators

The evaluation decorator system provides a pytest-like interface for running evaluations on function outputs. This allows you to easily integrate content safety checks into your existing code.

## Basic Usage

```python
from hapax.core.decorators import eval

@eval(evals=["hallucination"], threshold=0.7)
def generate_response(prompt: str) -> str:
    return f"Response to {prompt}"
```

## Features

### Multiple Evaluation Types

The `@eval` decorator supports several types of evaluations:
- `hallucination`: Check for factual accuracy
- `bias`: Detect potential biases
- `toxicity`: Identify harmful content
- `all`: Run all available evaluations

### Configurable Thresholds

Set acceptable thresholds for evaluations (0.0 to 1.0). Lower thresholds are more strict:
```python
@eval(evals=["bias"], threshold=0.3)  # Strict bias checking
def financial_advice(query: str) -> str:
    return f"Financial advice for {query}"
```

### Custom Configuration

Provide additional metadata and evaluation-specific configuration:
```python
@eval(
    evals=["toxicity"],
    threshold=0.5,
    metadata={"domain": "social"},
    openlit_config={"model": "toxicity-v2"}
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

1. **Choose Appropriate Thresholds**
   - Lower thresholds (0.3-0.5) for strict checking
   - Higher thresholds (0.7-0.9) for more permissive checking

2. **Evaluation Selection**
   - Use specific evaluations when possible
   - Use `all` for maximum safety

3. **Error Handling**
   - Always handle potential evaluation errors
   - Log evaluation failures for monitoring

4. **Configuration**
   - Use domain-specific configurations when available
   - Add relevant metadata for tracking

## Complete Example

See the [examples directory](../examples/eval_decorator_examples.py) for a complete working example that demonstrates all these features.
