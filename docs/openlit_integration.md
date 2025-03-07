# Advanced OpenLIT Integration

This document explains how to use the advanced OpenLIT integration features in Hapax, including GPU monitoring and evaluations for LLM outputs. For basic OpenLIT integration, see the [Basic OpenLIT Integration](openlit.md) guide.

## Table of Contents

- [Prerequisites](#prerequisites)
- [GPU Monitoring](#gpu-monitoring)
  - [Global GPU Monitoring](#global-gpu-monitoring)
  - [Per-Graph GPU Monitoring](#per-graph-gpu-monitoring)
  - [Accessing GPU Metrics](#accessing-gpu-metrics)
- [Evaluations](#evaluations)
  - [Available Evaluators](#available-evaluators)
  - [Using the @eval Decorator](#using-the-eval-decorator)
  - [Adding Evaluations to Graphs](#adding-evaluations-to-graphs)
  - [Working with Evaluation Results](#working-with-evaluation-results)
  - [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Prerequisites

Before using the OpenLIT integration features, you need to install the required dependencies:

```bash
pip install openlit>=1.33.8
```

For evaluation capabilities, you'll also need either:

```bash
pip install openai>=1.0.0  # For OpenAI evaluations
# OR
pip install anthropic>=0.5.0  # For Anthropic evaluations
```

## GPU Monitoring

Hapax provides GPU monitoring capabilities powered by OpenLIT, allowing you to track GPU utilization, memory usage, and other metrics during pipeline execution.

### Global GPU Monitoring

You can enable GPU monitoring globally for all operations:

```python
from hapax import enable_gpu_monitoring

# Basic setup with default parameters
enable_gpu_monitoring()

# Custom configuration
enable_gpu_monitoring(
    otlp_endpoint="http://localhost:4318",  # OpenTelemetry endpoint
    sample_rate_seconds=2,                  # How often to sample metrics
    application_name="my_app",              # Application name for metrics
    environment="production",               # Environment identifier
    custom_config={                         # Additional OpenLIT options
        "trace_memory": True,
        "track_processes": True
    }
)
```

### Per-Graph GPU Monitoring

You can also enable GPU monitoring for specific graph executions:

```python
from hapax import Graph, ops

@ops(name="process_data")
def process_data(data):
    # Processing logic
    return processed_data

@ops(name="run_model")
def run_model(data):
    # Model execution that might use GPU
    return result

# Create a graph with GPU monitoring
pipeline = (
    Graph("gpu_pipeline")
    .then(process_data)
    .then(run_model)
    .with_gpu_monitoring(
        enabled=True,
        sample_rate_seconds=1,
        custom_config={"trace_memory": True}
    )
)

# Execute the graph with GPU monitoring
result = pipeline.execute(input_data)
```

### Accessing GPU Metrics

You can access GPU metrics programmatically:

```python
from hapax import get_gpu_metrics

# Get the latest GPU metrics
metrics = get_gpu_metrics()

# Example metrics structure
# {
#   "gpu_utilization": 45.2,            # Percentage
#   "gpu_memory_used": 3.2,             # GB
#   "gpu_memory_total": 8.0,            # GB
#   "gpu_temperature": 68,              # Celsius
#   "gpu_power_usage": 120.5,           # Watts
#   "gpu_processes": ["python:12345"]   # Running processes
# }
```

## Evaluations

Hapax integrates with OpenLIT's evaluation capabilities to check LLM-generated content for hallucinations, bias, and toxicity.

### Available Evaluators

Hapax provides several OpenLIT-powered evaluators:

| Evaluator | Description |
|-----------|-------------|
| `HallucinationEvaluator` | Checks if the text contains factual inaccuracies compared to provided context |
| `BiasEvaluator` | Detects biased or prejudiced language |
| `ToxicityEvaluator` | Identifies harmful or offensive content |
| `AllEvaluator` | Combines all the above evaluations |

### Using the @eval Decorator

You can use the `@eval` decorator with OpenLIT to evaluate LLM outputs:

```python
from hapax import eval

@eval(
    evals=["hallucination"],                # Evaluation type(s)
    threshold=0.7,                          # Failure threshold (0-1)
    use_openlit=True,                       # Enable OpenLIT evaluations
    openlit_provider="openai",              # LLM provider for evaluation
    metadata={
        "contexts": ["Einstein won the Nobel Prize in Physics in 1921."],
        "prompt": "When did Einstein win the Nobel Prize?"
    }
)
def generate_response(prompt: str) -> str:
    # LLM generation code here
    return "Einstein won the Nobel Prize in 1922."  # Will fail evaluation
```

Available options for the `evals` parameter when `use_openlit=True`:
- `"hallucination"`: Check for factual inaccuracies
- `"bias"`: Check for biased language
- `"toxicity"`: Check for harmful content
- `"all"`: Check for all of the above

### Adding Evaluations to Graphs

You can also add evaluations to graph executions:

```python
from hapax import Graph, ops

@ops(name="generate_answer")
def generate_answer(query: str) -> str:
    # LLM response generation
    return response

# Create a graph with evaluation
pipeline = (
    Graph("answering_pipeline")
    .then(generate_answer)
    .with_evaluation(
        eval_type="all",                # Evaluation type
        threshold=0.6,                  # Threshold score
        provider="openai",              # "openai" or "anthropic"
        fail_on_evaluation=True,        # Whether to raise an exception on failure
        model="gpt-4o",                 # Specific model to use (optional)
        api_key=None,                   # API key (uses env var if None)
        custom_config={                 # Additional configuration
            "collect_metrics": True
        }
    )
)

# Execute the graph with evaluation
try:
    result = pipeline.execute("When did Einstein win the Nobel Prize?")
    print(f"Result: {result}")
    
    # Access evaluation results
    if pipeline.last_evaluation:
        print(f"Evaluation: {pipeline.last_evaluation}")
except Exception as e:
    print(f"Evaluation failed: {e}")
```

### Working with Evaluation Results

Evaluation results are returned as a dictionary with the following keys:

```python
{
    "score": 0.8,                      # Evaluation score (0-1)
    "verdict": "yes",                  # "yes" if issue detected, "no" otherwise
    "evaluation": "hallucination",     # Evaluation type
    "classification": "factual_inaccuracy", # Specific category of issue
    "explanation": "The text incorrectly states Einstein won the Nobel Prize in 1922, but according to the context, he won it in 1921." # Explanation
}
```

### Configuration Options

#### Evaluator Options

When creating evaluators directly:

```python
from hapax import HallucinationEvaluator

evaluator = HallucinationEvaluator(
    provider="openai",            # "openai" or "anthropic"
    threshold=0.5,                # Score threshold for verdict
    collect_metrics=True,         # Whether to collect metrics
    model="gpt-4o",               # Specific model to use (optional)
    api_key="your-api-key",       # API key (uses env var if None)
    base_url="https://custom-endpoint.com"  # Custom API endpoint (optional)
)

result = evaluator.evaluate(
    text="Einstein won the Nobel Prize in 1922.",
    contexts=["Einstein won the Nobel Prize in Physics in 1921."],
    prompt="When did Einstein win the Nobel Prize?"
)
```

Supported models based on provider:

- **OpenAI**: `"gpt-4o"`, `"gpt-4o-mini"` (default is provider-dependent)
- **Anthropic**: `"claude-3-5-sonnet"`, `"claude-3-5-haiku"`, `"claude-3-opus"` (default is provider-dependent)

## Troubleshooting

### Common Issues

1. **OpenLIT Not Installed**

   ```
   ImportError: OpenLIT not installed. Install with 'pip install openlit' to use evaluations.
   ```

   Solution: Install OpenLIT with `pip install openlit`.

2. **API Key Not Found**

   ```
   OpenLIT Error: Provider API key not found.
   ```

   Solution: Set the appropriate environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`), or pass the key explicitly via the `api_key` parameter.

3. **Model Selection Issues**

   If you get unexpected evaluation results, make sure you're specifying the appropriate model for the provider. If no model is specified, OpenLIT uses the provider's default model.

## Advanced Usage

### Custom Providers and Endpoints

You can use custom endpoints for API providers:

```python
evaluator = AllEvaluator(
    provider="openai",
    base_url="https://my-custom-endpoint/v1",
    model="custom-model"
)
```

### Combining with Other Features

You can combine GPU monitoring and evaluations:

```python
pipeline = (
    Graph("full_pipeline")
    .then(preprocess)
    .then(generate_response)
    .with_gpu_monitoring(enabled=True)
    .with_evaluation(eval_type="all")
)
```

### Using OpenTelemetry Metrics

Both GPU monitoring and evaluations can export metrics via OpenTelemetry. Configure the metrics collection:

```python
import openlit

# Initialize OpenLIT with a metrics endpoint
openlit.init(
    otlp_endpoint="http://localhost:4318",
    environment="development",
    application_name="my_app"
)

# Then use Hapax with OpenLIT features
from hapax import enable_gpu_monitoring
enable_gpu_monitoring()
``` 