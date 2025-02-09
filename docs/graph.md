# Hapax Graph API

The Hapax Graph API provides a flexible and intuitive way to build complex data processing pipelines. Inspired by frameworks like JAX and Flyte, it offers a fluent interface for composing operations and defining workflows, with built-in type checking at import time.

## Core Concepts

### Type-Safe Operations

Operations are the basic building blocks of a Hapax graph. Each operation is a pure function that takes an input and produces an output. Operations are type-checked at both import time and runtime:

```python
@ops(name="summarize", tags=["llm"])
def summarize(text: str) -> str:
    """Generate a concise summary using LLM."""
    # Implementation
```

The `@ops` decorator performs initial type validation at import time, ensuring the function has proper type hints. Further type checking occurs at runtime when the operation is executed.

### Type Validation Stages

Hapax performs comprehensive type checking at multiple stages:

1. Import Time (Static):
   - Validates presence of type hints through the `@ops` decorator
   - Checks input parameter types exist
   - Verifies return type annotations exist
   - Stores validated type information for later use

2. Graph Definition Time:
   - Type compatibility between connected operations
   - Structural validation (cycles, missing connections)
   - Configuration and metadata validation
   - Immediate type checking when using operation composition (`>>`)

3. Runtime (Dynamic):
   - Input type validation before operation execution
   - Output type validation after operation execution
   - Complete graph validation during execution
   - Type checking of operation results
   - Resource availability checks
   - Configuration validation

This multi-stage type checking ensures type safety throughout the entire lifecycle of your data processing pipeline:
- Early detection of type-related issues during development (import time)
- Immediate feedback when building graphs (definition time)
- Runtime safety guarantees during execution

For example, the following code would fail at different stages:

```python
# Fails at import time - missing type hints
@ops(name="bad_op")
def no_type_hints(x):
    return x + 1

# Fails at graph definition time - type mismatch
graph = (
    Graph("type_mismatch")
    .then(str_op)      # str -> str
    .then(int_op)      # int -> int  # Type error!
)

# Fails at runtime - actual input type doesn't match declaration
@ops(name="runtime_check")
def expect_string(text: str) -> str:
    return text.upper()

result = expect_string(123)  # Runtime type error
```

### Compile-Time Type Validation

A Graph is a collection of operations connected in a specific way. The Graph class provides a fluent API for building these connections, with comprehensive type checking at definition time:

```python
# Type compatibility is checked when the graph is defined
graph = (
    Graph("name", "description")
    .then(op1)  # Type compatibility checked immediately
    .then(op2)  # Type compatibility checked immediately
)
```

## Building Blocks

### 1. Sequential Operations (`.then()`)

Chain operations one after another with automatic type checking:

```python
graph = (
    Graph("text_processing")
    .then(clean_text)      # Returns str
    .then(tokenize)        # Expects str, returns List[str]
    .then(analyze)         # Expects List[str]
)
```

### 2. Parallel Processing (`.branch()`)

Execute multiple operations in parallel with type-safe result collection:

```python
graph = (
    Graph("parallel_processing")
    .branch(
        sentiment_analysis,  # Branch 1: str -> float
        entity_extraction,   # Branch 2: str -> List[str]
        topic_modeling      # Branch 3: str -> Dict[str, float]
    )
    .merge(combine_results)  # List[Union[float, List[str], Dict[str, float]]] -> Result
)
```

### 3. Conditional Logic (`.condition()`)

Add type-safe branching logic:

```python
graph = (
    Graph("language_processing")
    .then(detect_language)  # str -> str
    .condition(
        lambda lang: lang != "en",
        translate,          # str -> str
        lambda x: x        # str -> str (identity)
    )
)
```

### 4. Loops (`.loop()`)

Repeat an operation with type-safe condition checking:

```python
graph = (
    Graph("retry_logic")
    .loop(
        api_call,           # Request -> Response
        condition=lambda response: response.status == "success",
        max_iterations=3
    )
)
```

## Error Handling

Hapax provides clear error messages when validation fails:

1. Type Mismatch:
```python
TypeError: Cannot compose operations: output type List[str] does not match input type Dict[str, Any]
```

2. Structural Issues:
```python
GraphValidationError: Graph contains cycles: [['op1', 'op2', 'op1']]
```

3. Configuration Issues:
```python
ValueError: Missing required configuration: operation 'api_call' requires API endpoint
```

## Example: Advanced NLP Pipeline

Here's a real-world example that showcases the power and flexibility of the Graph API:

```python
def create_nlp_pipeline() -> Graph[str, Dict[str, Any]]:
    return (
        Graph("nlp_pipeline", "Advanced NLP processing pipeline")
        # First detect language and translate if needed
        .then(detect_language)
        .condition(
            lambda lang: lang != "en",
            translate,
            lambda x: x
        )
        # Then process in parallel
        .branch(
            summarize,           # Branch 1: Summarization
            sentiment_analysis,  # Branch 2: Sentiment
            extract_entities,    # Branch 3: Entity extraction
            extract_keywords     # Branch 4: Keyword extraction
        )
        # Merge results
        .merge(combine_results)
    )
```

This pipeline:
1. Detects the language of input text
2. Translates to English if needed
3. Processes the text in parallel:
   - Generates a summary
   - Analyzes sentiment and emotions
   - Extracts named entities
   - Identifies key topics
4. Combines all results into a structured output

## Type Safety

The Graph API includes built-in type checking to ensure type safety across operations:

```python
def tokenize(text: str) -> List[str]: ...
def analyze(tokens: List[str]) -> Dict[str, float]: ...

# Types are checked at runtime
graph = Graph("example").then(tokenize).then(analyze)
```

## Monitoring and Observability

Hapax integrates with OpenLit for monitoring and observability:

```python
# Configure globally
set_openlit_config({
    "trace_content": True,
    "disable_metrics": False
})

# Operations are automatically monitored
@ops(name="process", tags=["processing"])
def process(data: str) -> str:
    # OpenLit will trace this operation
    return data.upper()
```

## Best Practices

1. **Modularity**: Keep operations small and focused on a single task
2. **Type Hints**: Always use type hints to catch type errors early
3. **Documentation**: Add clear docstrings to operations
4. **Error Handling**: Use appropriate error handling in operations
5. **Monitoring**: Configure OpenLit for production monitoring

## Advanced Features

### Operation Composition

Operations can be composed using the `>>` operator:

```python
pipeline = tokenize >> normalize >> analyze
```

### Automatic Type Inference

The Graph API automatically infers input and output types from function signatures:

```python
@ops(name="process")
def process(text: str) -> List[str]:
    # Input and output types are automatically extracted
    return text.split()
```

### Rich Error Information

Flow operators provide detailed error information:

```python
try:
    result = pipeline(text)
except BranchError as e:
    print(f"Branch errors: {e.branch_errors}")
    print(f"Partial results: {e.partial_results}")
