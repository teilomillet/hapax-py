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
from hapax import ops, Graph
import openlit
from typing import List, Dict

# Initialize OpenLit (optional but recommended)
openlit.init(otlp_endpoint="http://127.0.0.1:4318")

# Define operations - type checked at import time
@ops(
    name="clean_text",
    tags=["text", "preprocessing"],
    config={"trace_content": True}
)
def clean_text(text: str) -> str:
    return text.lower().strip()

@ops(name="tokenize", tags=["text", "nlp"])
def tokenize(text: str) -> List[str]:
    return text.split()

@ops(name="count_tokens")
def count_tokens(tokens: List[str]) -> Dict[str, int]:
    return {token: tokens.count(token) for token in set(tokens)}

# Create a pipeline using the Graph class
word_counter = Graph("word_counter")
word_counter.then(clean_text)
            .then(tokenize)
            .then(count_tokens)

# Execute the pipeline
result = word_counter.execute("Hello, world!")
print(result)  # {'hello,': 1, 'world!': 1}
```

3. Add text evaluation with the `@eval` decorator:
```python
from hapax import eval

@eval(
    name="safe_generator",
    evaluators=["hallucination", "toxicity"],
    threshold=0.3,
    config={"check_factuality": True}
)
def generate_response(prompt: str) -> str:
    # This function would typically call an LLM
    return f"Response to: {prompt}"

# The response will be automatically evaluated
response = generate_response("Tell me about history")
```

4. See Hapax in Action:
Check out our demo comparing standard Python implementation with Hapax:
```bash
python examples/hapax_demo.py
```
This demo showcases Hapax's key advantages:
- Multi-stage type checking
- Declarative graph-based pipelines 
- Automatic monitoring integration
- Centralized error handling
- Built-in evaluation framework

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
pipeline = Graph("text_analysis")
pipeline.then(clean_text)      # str -> str
pipeline.branch(
    summarize,         # str -> str
    sentiment_analysis # str -> float
)
pipeline.merge(combine_results)

# You can also chain methods for a more fluent API
pipeline = (Graph("nlp_pipeline")
            .then(clean_text)
            .then(tokenize)
            .then(analyze))
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

The documentation is organized into the following sections:

1. **Getting Started**
   - [Installation Guide](docs/installation.md) - Requirements and installation options
   - [Quick Start Guide](docs/quickstart.md) - Get up and running in under 5 minutes
   - [Comprehensive Guide](docs/guide.md) - In-depth explanation of all key concepts
   - [Examples](docs/examples.md) - Code examples demonstrating various features

2. **Core Features**
   - [Graph API Reference](docs/graph.md) - Complete reference for the Graph API and building pipelines
   - [API Reference](docs/api_reference.md) - Comprehensive reference for all Hapax classes and functions

3. **Integrations**
   - [OpenLit Basics](docs/openlit.md) - Simple monitoring setup with OpenLit
   - [Advanced OpenLit Integration](docs/openlit_integration.md) - GPU monitoring, evaluations, and advanced features

4. **Advanced Features**
   - [Evaluation Decorators](docs/evaluation_decorators.md) - Using evaluation decorators for content safety

5. **Support and Troubleshooting**
   - [Troubleshooting Guide](docs/troubleshooting.md) - Solutions for common issues

For a complete overview, start with the [documentation index](docs/index.md).

## Hapax Documentation Assistant (MCP)

The Hapax Documentation Assistant is an MCP (Model Context Protocol) server that provides AI-powered access to Hapax documentation and source code through tools that can be used by AI assistants like Claude in Cursor.

### Features

- 🔍 **Smart Documentation Search**: Find relevant documentation based on natural language queries
- 📚 **Source Code Navigation**: Explore and understand Hapax source code
- 🛠️ **Implementation Guidance**: Get guidance on implementing specific patterns or features
- 🔧 **Troubleshooting**: Get help with errors and issues
- 📖 **Topic Exploration**: Comprehensive exploration of Hapax concepts and topics

### Installation

The Documentation Assistant can be easily installed as a package:

```bash
# Using uv (recommended)
uv add "hapax[assistant]"

# Or using pip
pip install "hapax[assistant]"
```

### Setting Up Cursor Integration

Run the automatic setup script to configure the assistant for Cursor:

```bash
# If you installed with uv
uv run -m hapax_docs cursor-setup

# If you installed with pip
python -m hapax_docs cursor-setup
```

Then restart Cursor to activate the assistant.

### Running Manually

If you need to run the assistant manually:

```bash
# Run the documentation server
python -m hapax_docs run

# With custom documentation and source directories
python -m hapax_docs run --docs-dir /path/to/docs --source-dir /path/to/source
```

### Advanced Configuration

For advanced users who want to manually configure the assistant in Cursor:

1. Go to **Cursor Settings > Features > MCP** 
2. Click the **+ Add New MCP Server** button
3. Configure as follows:
   - **Type**: CLI (stdio transport)
   - **Name**: Hapax Documentation Assistant
   - **Command**: `python -m hapax_docs.server`

### Project-Specific Configuration

For project-specific configuration in Cursor, create a `.cursor/mcp.json` file in your project:

```json
{
  "mcpServers": {
    "hapax-docs": {
      "command": "python",
      "args": ["-m", "hapax_docs", "run"]
    }
  }
}
```

Or to use with custom directories:

```json
{
  "mcpServers": {
    "hapax-docs": {
      "command": "python",
      "args": ["-m", "hapax_docs", "run", "--docs-dir", "./docs", "--source-dir", "./src"]
    }
  }
}
```

### Available Tools

The Hapax Documentation Assistant provides the following tools:

- **search_docs**: Search the Hapax documentation and source code
- **get_section_content**: Get detailed information about a documentation section
- **get_implementation_pattern**: Get guidance on implementing specific patterns
- **get_source_element**: View source code for specific elements (functions, classes, etc.)
- **find_usage_examples_tool**: Find examples of how components are used in the codebase
- **get_implementation_guidance**: Get guidance on implementing features or components
- **troubleshoot_issue**: Get help with errors or issues
- **understand_hapax_component**: Get comprehensive information about a Hapax component
- **explore_hapax_topic**: Explore general topics in the Hapax framework

### Using the Tools in Agent

Once added, the Hapax Documentation Assistant will be available to the Agent in Cursor's Composer. You can:

1. Open Composer in Cursor
2. Ask questions about Hapax documentation, for example:
   - "How do I use the Graph component in Hapax?"
   - "Search for documentation about operations in Hapax"
   - "Show me an example of implementing error handling in Hapax"

The Agent will automatically use your MCP tools when relevant. You can also directly prompt tool usage by mentioning a specific tool:
- "Use the search_docs tool to find information about error handling in Hapax"

## License

MIT License - see [LICENSE](LICENSE) for details.