# Hapax Documentation Assistant

A documentation assistant for the Hapax framework using the Model Context Protocol (MCP). This tool helps you implement and use Hapax in your projects by providing code assistance, documentation search, and implementation guidance directly within Cursor.

## What is Hapax?

[Hapax](https://github.com/teilomillet/hapax-py) is a type-safe graph execution framework built on top of OpenLit for LLM pipelines. It provides a structured way to build, validate, and execute complex AI workflows with strong typing.

## What This Assistant Does

This assistant helps you:

- Search Hapax documentation and source code
- Understand Hapax components and concepts
- Find implementation patterns and examples
- Get guidance for specific implementation challenges
- Troubleshoot errors and issues
- Navigate the Hapax API

## Installation

### Prerequisites

- Python 3.11 or higher
- [Cursor IDE](https://cursor.sh) installed
- Basic understanding of Python

### Setup

```bash
# Using uv (recommended)
uv add "hapax[assistant]"

# Or using pip
pip install "hapax[assistant]"
```

### Quick Setup for Cursor

Run the setup script to automatically configure the assistant for Cursor:

```bash
# If you installed with uv
uv run -m hapax_docs cursor-setup

# If you installed with pip
python -m hapax_docs cursor-setup
```

Then restart Cursor to activate the assistant.

## Getting Started with Hapax Implementation

### 1. Setting Up Your Project

After installing the assistant, you can ask Cursor's AI about Hapax setup:

Example prompt:
> "How do I set up a new project with Hapax? Show me the initial project structure and dependencies."

The assistant will use its tools to provide guidance on project setup, including:
- Required dependencies
- Initial file structure
- Basic configuration

### 2. Building Your First Graph

When you're ready to build your first Hapax graph, you can ask:

Example prompt:
> "Show me how to create a simple Hapax graph with input and output nodes"

The assistant will provide examples and explain key concepts:
- Graph creation and structure
- Node types and connections
- Type safety principles
- Execution methods

### 3. Working with Operations

To understand how to implement operations:

Example prompt:
> "How do I implement custom operations in Hapax? I need to process text data."

The assistant will provide:
- Code examples of operations
- Explanation of operation patterns
- Type handling best practices
- Integration methods

### 4. Implementing Complex Workflows

For more advanced implementations:

Example prompt:
> "Show me how to implement a multi-step LLM pipeline with Hapax that includes preprocessing and validation"

## Common Implementation Tasks

### Integrating with LLMs

```python
# Example of using Hapax with OpenAI
from hapax import Graph, Operation, types
from openai import OpenAI

# Ask the assistant for details on how to implement this pattern
```

Example prompt:
> "How do I connect an OpenAI client to a Hapax graph? Show me the type definitions I need."

### Type-Safe Graph Construction

```python
# Example of creating a type-safe graph
from hapax import Graph, Operation, types

# Ask the assistant for details on how to implement this pattern
```

Example prompt:
> "Explain how type checking works in Hapax and how I can ensure my graph is type-safe"

### Error Handling in Graphs

```python
# Example of error handling patterns
from hapax import Graph, Operation, types

# Ask the assistant for details on how to implement this pattern
```

Example prompt:
> "What's the best way to handle errors in a Hapax graph? Show me implementation examples."

## Available Tools

The Hapax Documentation Assistant provides these specialized tools:

- `search_docs` - Search through Hapax documentation and source code for specific concepts or patterns
- `get_section_content` - Get detailed explanations of specific documentation sections
- `get_implementation_pattern` - Get code examples and explanations for common implementation patterns
- `get_source_element` - Examine source code for specific functions, classes, or methods
- `find_usage_examples_tool` - Find real-world examples of how components are used in the codebase
- `get_implementation_guidance` - Get step-by-step guidance for implementing specific features
- `troubleshoot_issue` - Get help with errors or issues in your Hapax implementation
- `understand_hapax_component` - Get comprehensive information about key components
- `explore_hapax_topic` - Deep dive into specific Hapax topics

## Example Use Cases

### Creating a Chat Application with Hapax

Ask the assistant:
> "I want to build a chat application with Hapax. How should I structure the graph to handle user input, context tracking, and LLM responses?"

### Building a Document Processing Pipeline

Ask the assistant:
> "I need to build a document processing pipeline with Hapax that extracts information from PDFs. How do I implement this?"

### Implementing Real-Time Type Validation

Ask the assistant:
> "Show me how to implement real-time validation of user inputs in a Hapax graph that connects to a web service"

## Advanced Configuration

### Environment Variables

Set the documentation and source code directories:

```bash
hapax-docs run --docs-dir /path/to/docs --source-dir /path/to/source
```

Or using environment variables:

```bash
hapax-docs run -v HAPAX_DOCS_DIR /path/to/docs -v HAPAX_SOURCE_DIR /path/to/source
```

### Troubleshooting

If you encounter issues with the assistant:

```bash
# Run diagnostic tests
python -m hapax_docs debug

# Reinstall Cursor integration
python -m hapax_docs cursor-setup
```

## Integration with Cursor

To manually integrate with Cursor (if the automatic setup doesn't work):

1. Open Cursor and go to Settings > Features > MCP
2. Click '+ Add New MCP Server'
3. Set 'Type' to 'stdio'
4. Set 'Name' to 'Hapax Documentation Assistant'
5. Set 'Command' to: `python -m hapax_docs.server`
6. Click 'Save'

## License

This project is licensed under the MIT License. 