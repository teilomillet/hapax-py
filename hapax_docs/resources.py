"""
MCP resources for the Hapax Documentation Assistant.

This module defines all the resources exposed by the MCP server.
"""
import traceback
from pathlib import Path

def register_resources(mcp):
    """Register all resources with the MCP server"""
    try:
        print("Registering resource: hapax://docs/index")
        @mcp.resource("hapax://docs/index")
        def get_docs_index() -> str:
            """Get the documentation index."""
            try:
                # Get the index from the context - using request_context from the mcp instance
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return "Error: Documentation index is not initialized."
                return doc_index.get_index_content()
            except Exception as e:
                return f"Error retrieving docs index: {str(e)}"

        print("Registering resource: hapax://docs/file/{filename}")
        @mcp.resource("hapax://docs/file/{filename}")
        def get_doc_file(filename: str) -> str:
            """Get a documentation file by name.
            
            Args:
                filename: The name of the documentation file to retrieve
                
            Returns:
                The contents of the documentation file
            """
            try:
                # Get the index from the context
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return f"Error: Documentation index is not initialized."
                return doc_index.get_doc_file(filename)
            except Exception as e:
                return f"Error retrieving file '{filename}': {str(e)}"

        print("Registering resource: hapax://source/{filepath}")
        @mcp.resource("hapax://source/{filepath}")
        def get_source_file(filepath: str) -> str:
            """Get a source code file by path.
            
            Args:
                filepath: The path to the source file to retrieve
                
            Returns:
                The contents of the source file
            """
            try:
                # Get the index from the context
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return f"Error: Documentation index is not initialized."
                return doc_index.get_source_file(filepath)
            except Exception as e:
                return f"Error retrieving source file '{filepath}': {str(e)}"

        print("Registering resource: hapax://section/{section_name}")
        @mcp.resource("hapax://section/{section_name}")
        def get_section_resource(section_name: str) -> str:
            """Get documentation for a specific section.
            
            Args:
                section_name: The name of the section to retrieve
                
            Returns:
                The section content as a formatted string
            """
            try:
                # Get the index from the context
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return f"Error: Documentation index is not initialized."
                return doc_index.get_section_content(section_name)
            except Exception as e:
                return f"Error retrieving section '{section_name}': {str(e)}"

        print("Registering resource: hapax://element/{element_name}")
        @mcp.resource("hapax://element/{element_name}")
        def get_element_resource(element_name: str) -> str:
            """Get documentation for a specific API element.
            
            Args:
                element_name: The name of the element to retrieve
                
            Returns:
                The element documentation as a formatted string
            """
            try:
                # Get the index from the context
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return f"Error: Documentation index is not initialized."
                
                # Find the element in the documentation
                element_info = doc_index.get_element_info(element_name)
                
                if not element_info:
                    return f"Element '{element_name}' not found in documentation."
                
                return element_info
            except Exception as e:
                return f"Error retrieving element '{element_name}': {str(e)}"

        print("Registering resource: hapax://example/{pattern_name}")
        @mcp.resource("hapax://example/{pattern_name}")
        def get_example_resource(pattern_name: str) -> str:
            """Get an example for a specific implementation pattern.
            
            Args:
                pattern_name: The name of the pattern to get examples for
                
            Returns:
                Example code and explanation as a formatted string
            """
            try:
                # Get the index from the context
                doc_index = mcp.current_request_context.lifespan_context
                if doc_index is None:
                    return f"Error: Documentation index is not initialized."
                
                # Get examples for the pattern
                examples = doc_index.get_pattern_examples(pattern_name)
                
                if not examples:
                    return f"No examples found for pattern '{pattern_name}'."
                
                return examples
            except Exception as e:
                return f"Error retrieving examples for pattern '{pattern_name}': {str(e)}"
                
        # Add a simple test resource that doesn't depend on the DocIndex
        print("Registering resource: hapax://test")
        @mcp.resource("hapax://test")
        def get_test_resource() -> str:
            """A simple test resource to verify the server is working."""
            return "This is a test resource. The server is working correctly!"
            
        print("All resources registered successfully")
    except Exception as e:
        print(f"ERROR registering resources: {str(e)}")
        print(traceback.format_exc()) 