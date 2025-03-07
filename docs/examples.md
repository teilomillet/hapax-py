# Hapax Examples

This document provides an overview of the example code available in the Hapax repository. These examples demonstrate different features and use cases for the Hapax framework.

## Examples Directory

All examples can be found in the `examples/` directory in the repository.

```
examples/
├── basic/
│   ├── simple_pipeline.py - Basic operations and composition
│   ├── type_checking.py - Demonstrates type safety features
│   └── monitoring.py - Basic OpenLIT monitoring
├── nlp/
│   ├── text_processing.py - Text analysis pipeline
│   ├── language_detection.py - Conditional processing based on language
│   └── summarization.py - Text summarization with evaluations
├── data/
│   ├── etl_pipeline.py - Data extraction and transformation
│   ├── data_validation.py - Validating data with operations
│   └── batch_processing.py - Processing data in batches with loops
├── advanced/
│   ├── complex_graph.py - Advanced graph with multiple flow controls
│   ├── gpu_monitoring.py - Monitoring GPU usage
│   └── evaluation_example.py - Using evaluators for content safety
└── integrations/
    ├── custom_evaluator.py - Creating custom evaluators
    ├── openai_integration.py - Using OpenAI with Hapax
    └── monitoring_dashboard.py - Visualizing metrics with OpenLIT
```

## Running the Examples

To run an example:

```bash
cd hapax
python examples/basic/simple_pipeline.py
```

Most examples require additional dependencies. Install the complete set of dependencies:

```bash
pip install "hapax[all]"
```

## Example Highlights

### Basic Operations and Composition

```python
# From examples/basic/simple_pipeline.py
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

result = analyze_text("Hello world! Hello Hapax!")
print(result)  # {'hello': 2, 'world!': 1, 'hapax!': 1}
```

### Type Safety Features

```python
# From examples/basic/type_checking.py
from hapax import ops, graph
from typing import List, Dict

@ops
def tokenize(text: str) -> List[str]:
    return text.split()

@ops
def normalize(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]

# This would fail at graph construction time with a type error
# @ops
# def incorrect_input(numbers: List[int]) -> List[int]:
#     return [n * 2 for n in numbers]
# 
# pipeline = tokenize >> incorrect_input  # TypeError: output type List[str] does not match input type List[int]
```

### Advanced Flow Control

```python
# From examples/advanced/complex_graph.py
from hapax import Graph, ops
from typing import List, Dict, Any

@ops
def detect_language(text: str) -> str:
    # Simple placeholder detection
    if "bonjour" in text.lower():
        return "fr"
    return "en"

@ops
def translate_to_english(text: str) -> str:
    # Simple placeholder translation
    return f"[Translated from French]: {text}"

@ops
def analyze_tokens(tokens: List[str]) -> Dict[str, Any]:
    return {
        "count": len(tokens),
        "unique": len(set(tokens))
    }

# Create a complex pipeline with branching and conditional logic
pipeline = (
    Graph("language_processing")
    .then(detect_language)
    .condition(
        lambda lang: lang != "en",
        translate_to_english,
        lambda x: x  # Identity function for English text
    )
    .then(lambda text: text.split())
    .then(analyze_tokens)
)

# Test with English text
result1 = pipeline.execute("Hello world")
print(result1)  # {'count': 2, 'unique': 2}

# Test with French text
result2 = pipeline.execute("Bonjour monde")
print(result2)  # {'count': 4, 'unique': 4} (after translation)
```

### Content Evaluation

```python
# From examples/advanced/evaluation_example.py
from hapax import ops, eval, graph
from typing import Dict, Any

@ops
@eval(evals=["toxicity"], threshold=0.7)
def generate_text(prompt: str) -> str:
    # In a real example, this would call an LLM
    return f"Generated response to: {prompt}"

@ops
def analyze_response(text: str) -> Dict[str, Any]:
    return {
        "length": len(text),
        "words": len(text.split()),
        "response": text
    }

@graph
def safe_response_pipeline(prompt: str) -> Dict[str, Any]:
    return generate_text >> analyze_response

# This works because the generated text passes the toxicity check
result = safe_response_pipeline("Tell me about science")
print(result)
```

## Creating Your Own Examples

We encourage you to build on these examples and create your own. When sharing examples with the community, please follow these guidelines:

1. Include clear imports and dependencies
2. Add comments explaining key concepts
3. Use meaningful operation and graph names
4. Include sample inputs and expected outputs
5. Handle errors appropriately

## Contributing Examples

To contribute examples to the Hapax repository:

1. Fork the repository
2. Create a new branch for your example
3. Add your example to the appropriate directory
4. Add documentation in the example file
5. Submit a pull request

See the [Contributing Guide](../CONTRIBUTING.md) for more details. 