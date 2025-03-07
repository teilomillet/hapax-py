# Hapax Documentation Assistant

![Hapax](https://img.shields.io/badge/Hapax-Documentation-blue)
![MCP](https://img.shields.io/badge/MCP-Enabled-green)
![Cursor](https://img.shields.io/badge/Cursor-Compatible-orange)

An AI-powered documentation assistant for implementing Hapax in your projects. This tool integrates with [Cursor IDE](https://cursor.sh) to provide contextual help, code examples, and implementation guidance directly in your development environment.

## Why Use This Assistant?

When implementing Hapax in your project, this assistant will help you:

- **Navigate complex concepts**: Understand Graph, Operation, and type system concepts
- **Find implementation patterns**: Get examples tailored to your use case
- **Troubleshoot errors**: Quickly resolve common implementation issues
- **Build correctly**: Follow best practices for type-safe graph construction

## Quick Start

1. **Install Hapax with the assistant extra**:

```bash
# Using uv (recommended)
uv add "hapax[assistant]"

# Or using pip
pip install "hapax[assistant]"
```

2. **Set up Cursor integration**:

```bash
# If you installed with uv
uv run -m hapax_docs cursor-setup

# If you installed with pip
python -m hapax_docs cursor-setup
```

3. **Restart Cursor** to activate the assistant

4. **Start asking questions** like:
   - "How do I set up a Hapax project?"
   - "Show me how to create a simple graph with input and output nodes"
   - "Explain how to implement a custom operation"

## Try the Example

We've included an example script that demonstrates how to use Hapax with the documentation assistant:

```bash
# Using uv
uv run examples/hapax_assistant_example.py

# Using python
python examples/hapax_assistant_example.py
```

This example:
1. Checks if Hapax and the documentation assistant are installed
2. Sets up Cursor integration
3. Creates a simple Hapax graph with an addition operation
4. Provides next steps for using the assistant

## Example Implementation Assistance

The assistant excels at providing practical implementation guidance:

### Building a Basic LLM Pipeline

Ask:
> "I want to build a simple LLM pipeline with Hapax. Can you guide me through creating a graph that takes a prompt, sends it to OpenAI, and processes the response?"

The assistant will provide step-by-step implementation guidance, including:
- Project structure
- Required imports
- Graph construction
- Type definitions
- Running the graph

### Adding Type Safety to Existing Code

Ask:
> "I have a Python script that calls OpenAI. How do I convert it to use Hapax's type-safe graph execution?"

The assistant will analyze your needs and provide guidance on:
- Converting imperative code to graph-based execution
- Adding appropriate type annotations
- Structuring the graph properly
- Validating inputs and outputs

### Debugging Implementation Issues

Ask:
> "I'm getting a type error when connecting my custom operation to the graph. The error is: 'Cannot connect Operation[str] to Operation[dict]'. How do I fix this?"

The assistant will help troubleshoot with:
- Explanations of type compatibility issues
- Solutions for type conversion
- Best practices to avoid similar issues
- Code examples showing the correct approach

## Features

- **Documentation Search**: Find relevant information in the Hapax documentation
- **Code Examples**: Get implementation patterns with explanations
- **API Reference**: Explore the Hapax API right in your editor
- **Usage Examples**: See how components are used in real code
- **Implementation Guidance**: Get step-by-step instructions
- **Troubleshooting**: Resolve errors in your implementation

## Advanced Usage

For more information on advanced usage, see the [detailed README](hapax_docs/README.md) in the hapax_docs directory.

## Contributing

We welcome contributions to improve the assistant. Please see our [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. 