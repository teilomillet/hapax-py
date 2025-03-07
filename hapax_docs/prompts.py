"""
MCP prompts for the Hapax Documentation Assistant.

This module defines all the prompts exposed by the MCP server.
"""

def register_prompts(mcp):
    """Register all prompts with the MCP server"""
    
    @mcp.prompt()
    def how_to_use_hapax_docs() -> str:
        """Guide on how to use the Hapax documentation assistant"""
        return """
# How to Use the Hapax Documentation Assistant

I can help you understand and implement Hapax in your Python projects. This documentation assistant provides access to both the curated documentation and the actual source code.

## Available Tools

### Searching and Exploring
- `search_docs`: Search both documentation and source code with a specific query
- `explore_hapax_topic`: Get a comprehensive view of a specific topic (like "evaluation" or "type checking")
- `understand_hapax_component`: Get complete information about a component (like "Graph" or "Operation")

### Getting Specific Information
- `get_section_content`: Retrieve a specific section from the documentation
- `get_implementation_pattern`: Find code examples for specific patterns
- `get_source_element`: View the source code of functions, classes, or methods
- `find_usage_examples_tool`: See how a component is used in practice

### Implementation Help
- `get_implementation_guidance`: Get guidance for implementing a feature
- `troubleshoot_issue`: Get help with errors or issues

## How to Get Started

1. **To understand Hapax concepts**:
   - Use `explore_hapax_topic` with topics like "operations", "graphs", or "type safety"
   - Example: `explore_hapax_topic("graph building")`

2. **To get implementation details**:
   - Use `understand_hapax_component` for core components
   - Example: `understand_hapax_component("Graph")`

3. **To search for specific information**:
   - Use `search_docs` with your query
   - Example: `search_docs("branch and merge")`

4. **If you encounter errors**:
   - Use `troubleshoot_issue` with your error message
   - Example: `troubleshoot_issue("TypeError: Input type mismatch")`

## Tip: Be Specific
The more specific your queries, the more helpful the results will be. Include error messages, component names, or specific functionality you're interested in.

What would you like to know about Hapax?
"""

    @mcp.prompt()
    def hapax_implementation_guide() -> str:
        """Guide for implementing Hapax in a project"""
        return """
# Implementing Hapax in Your Project

I'll guide you through the process of integrating Hapax into your Python project. Let's start with understanding your requirements:

## Questions to Consider

1. What's the main purpose of your data pipeline?
   - Text processing
   - LLM interaction
   - Data transformation
   - Other (please specify)

2. What kind of data processing do you need?
   - Sequential (one step after another)
   - Parallel (multiple operations on same data)
   - Conditional (different paths based on data)
   - Combinations of the above

3. Do you need additional features like:
   - GPU monitoring
   - Evaluation of LLM outputs
   - Visualization of pipelines

## Implementation Steps

Based on your needs, here's a general approach:

1. **Installation**:
   ```python
   pip install hapax
   ```

2. **Define Operations**:
   - Create focused operations with the `@ops` decorator
   - Use proper type annotations
   - Handle errors gracefully

3. **Build Pipeline**:
   - Create a `Graph` object
   - Add operations using `.then()`, `.branch()`, etc.
   - Use `.merge()` to combine parallel branches

4. **Execute and Monitor**:
   - Use `.execute()` to run the pipeline
   - Add evaluation or monitoring as needed
   - Visualize with `.visualize()`

## Example Implementation Patterns

I can provide specific code examples for:
- Sequential pipelines
- Parallel processing
- Conditional branches
- LLM integration
- Error handling

Let's start building your Hapax implementation!
"""

    @mcp.prompt()
    def troubleshooting_guide() -> str:
        """Guide for troubleshooting common Hapax issues"""
        return """
# Troubleshooting Hapax Issues

Having trouble with your Hapax implementation? I can help you diagnose and fix common issues.

## Common Error Types

1. **Type Mismatch Errors**
   - `TypeError: Input type mismatch in operation_name`
   - `TypeError: Output type mismatch in operation_name`
   - `TypeError: Type mismatch in composition`

2. **Graph Construction Errors**
   - `ValueError: Cannot merge before branching`
   - `TypeError: Function must have type hints`

3. **LLM Integration Issues**
   - JSON parsing errors
   - Malformed responses
   - Missing fields

4. **Memory/Performance Issues**
   - High memory usage
   - Slow execution
   - GPU memory leaks

## To Get the Best Help

1. Provide your **error message** exactly as shown
2. Share a **code snippet** showing the issue
3. Describe what you're **trying to accomplish**

I'll analyze your issue using:
- Documentation sections on common problems
- Source code examples
- Relevant implementation patterns
- Specific troubleshooting guidance

Let me know what issue you're experiencing!
"""

    @mcp.prompt()
    def learning_hapax_concepts() -> str:
        """Guide for learning Hapax core concepts"""
        return """
# Learning Hapax Core Concepts

Let's explore the fundamental concepts of Hapax to help you understand how it works.

## Core Components

1. **Operations**
   - Atomic units of computation
   - Created with the `@ops` decorator
   - Type-safe with input/output type checking
   - Composable with the `>>` operator
   
2. **Graphs**
   - Represent the flow of data through operations
   - Created with the `Graph` class
   - Support sequential, parallel, and conditional flows
   - Can be visualized for better understanding

3. **Type Safety**
   - Runtime type checking for data integrity
   - Type annotations required for all operations
   - Type compatibility checked during composition
   - Error messages identify type mismatches

4. **LLM Integration**
   - Built-in support for LLM operations
   - Error handling for LLM responses
   - Evaluation capabilities for generated content
   - Monitoring for resource usage

## Learning Path

1. **Start with simple sequential operations**
   - Create basic operations with `@ops`
   - Compose them with `>>`
   - Execute them with simple inputs

2. **Move to parallel processing**
   - Use `.branch()` to create parallel paths
   - Use `.merge()` to combine results
   - Design effective merge functions

3. **Add conditional flows**
   - Use `.condition()` for branching logic
   - Create predicate functions

4. **Add advanced features**
   - Implement error handling
   - Add monitoring
   - Add evaluation for LLM outputs

What aspect of Hapax would you like to explore first?
"""

    @mcp.prompt()
    def building_llm_pipelines() -> str:
        """Guide specifically for building LLM pipelines with Hapax"""
        return """
# Building LLM Pipelines with Hapax

Hapax provides excellent support for creating robust LLM pipelines with type safety and error handling.

## LLM Pipeline Components

1. **Input Processing Operations**
   - Text cleaning and normalization
   - Prompt construction
   - Context management

2. **LLM Interaction Operations**
   - API calls to LLM providers
   - Error handling and retries
   - Response parsing

3. **Output Processing Operations**
   - Validation of LLM outputs
   - Extraction of structured data
   - Post-processing of content

4. **Evaluation and Monitoring**
   - Quality assessment with `@eval`
   - GPU resource monitoring
   - OpenLIT integration

## Best Practices

1. **Robust Error Handling**
   - Always wrap LLM calls in try/except blocks
   - Provide fallback values for missing fields
   - Validate response structures

2. **Type-Safe Response Parsing**
   - Define clear type annotations for LLM responses
   - Convert string fields to appropriate types
   - Handle unexpected response formats

3. **Pipeline Structure**
   - Use parallel branches for different analyses
   - Merge results into structured outputs
   - Include evaluation operations

4. **Performance Optimization**
   - Batch requests when possible
   - Implement caching for expensive operations
   - Monitor and manage resource usage

## Example Implementation

I can provide example code for:
- LLM-based sentiment analysis
- Entity extraction pipelines
- Text summarization with evaluation
- Multi-LLM orchestration

How would you like to use LLMs in your Hapax pipeline?
""" 