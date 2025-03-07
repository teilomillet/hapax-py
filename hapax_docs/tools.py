"""
MCP tools for the Hapax Documentation Assistant.

This module defines all the tools exposed by the MCP server.
"""

import json
import re
from typing import List, Dict, Any, Optional
import mcp.server.fastmcp
from .models import SearchResult
from .indexer import search_content, find_usage_examples

# Import Context directly from the module
from mcp.server.fastmcp import Context

def register_tools(mcp):
    """Register all tools with the MCP server"""
    
    @mcp.tool()
    def search_docs(query: str, include_source: bool = True) -> str:
        """
        Search the Hapax documentation and optionally source code.
        
        Args:
            query: The search term to look for
            include_source: Whether to include source code in search results
            
        Returns:
            Search results with matches and context
        """
        doc_index = mcp.current_request_context.lifespan_context
        results = search_content(doc_index, query, include_source=include_source)
        
        # Format results for JSON serialization
        formatted_results = []
        for result in results[:15]:  # Limit to top 15 results
            formatted_result = {
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type,
                "score": result.score
            }
            
            # Add optional fields if present
            if result.name:
                formatted_result["name"] = result.name
            if result.docstring:
                formatted_result["docstring"] = result.docstring
            if result.code:
                # Trim code to keep response size reasonable
                formatted_result["code"] = result.code[:300] + "..." if len(result.code) > 300 else result.code
                
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "query": query,
            "include_source": include_source,
            "results_count": len(results),
            "results": formatted_results
        })

    @mcp.tool()
    def get_section_content(section_name: str) -> str:
        """
        Get content for a specific documentation section.
        
        Args:
            section_name: The name of the section to retrieve
            
        Returns:
            Section content from all matching sections
        """
        doc_index = mcp.current_request_context.lifespan_context
        sections = doc_index.get_section(section_name)
        
        # Format sections for JSON serialization
        formatted_sections = []
        for section in sections:
            formatted_sections.append({
                "title": section.title,
                "content": section.content,
                "file_path": section.file_path,
                "line_start": section.line_start,
                "line_end": section.line_end
            })
        
        return json.dumps({
            "section": section_name,
            "found": len(sections) > 0,
            "count": len(sections),
            "sections": formatted_sections
        })

    @mcp.tool()
    def get_implementation_pattern(pattern_name: str) -> str:
        """
        Get code examples for a specific implementation pattern.
        
        Args:
            pattern_name: The name or type of pattern to retrieve
            
        Returns:
            Code examples and explanations for the pattern
        """
        doc_index = mcp.current_request_context.lifespan_context
        examples = doc_index.get_code_example(pattern_name)
        
        # Format examples for JSON serialization
        formatted_examples = []
        for example in examples:
            formatted_examples.append({
                "pattern_name": example.pattern_name,
                "description": example.description,
                "code": example.code,
                "file_path": example.file_path,
                "line_start": example.line_start,
                "line_end": example.line_end
            })
        
        return json.dumps({
            "pattern": pattern_name,
            "found": len(examples) > 0,
            "count": len(examples),
            "examples": formatted_examples
        })

    @mcp.tool()
    def get_source_element(
        element_name: str, 
        element_type: Optional[str] = None
    ) -> str:
        """
        Get source code for a specific function, class, or method.
        
        Args:
            element_name: The name of the function, class, or method
            element_type: Optional type filter ('function', 'class', 'method')
            
        Returns:
            Source code and docstring for the element
        """
        doc_index = mcp.current_request_context.lifespan_context
        elements = doc_index.get_source_element(element_name, element_type)
        
        # Format elements for JSON serialization
        formatted_elements = []
        for element in elements:
            formatted_elements.append({
                "name": element.name,
                "type": element.type,
                "file_path": element.file_path,
                "line_start": element.line_start,
                "line_end": element.line_end,
                "docstring": element.docstring,
                "code": element.code,
                "parent": element.parent
            })
        
        return json.dumps({
            "element_name": element_name,
            "element_type": element_type,
            "found": len(elements) > 0,
            "count": len(elements),
            "elements": formatted_elements
        })

    @mcp.tool()
    def find_usage_examples_tool(element_name: str) -> str:
        """
        Find examples of how a function, class, or method is used in the code and documentation.
        
        Args:
            element_name: The name of the function, class, or method
            
        Returns:
            Usage examples from source code and documentation
        """
        doc_index = mcp.current_request_context.lifespan_context
        results = find_usage_examples(doc_index, element_name)
        
        # Format results for JSON serialization
        formatted_results = []
        for result in results:
            formatted_result = {
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type,
                "score": result.score
            }
            
            # Add optional fields if present
            if result.code:
                formatted_result["code"] = result.code
                
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "element_name": element_name,
            "found": len(results) > 0,
            "count": len(results),
            "examples": formatted_results
        })

    @mcp.tool()
    def get_implementation_guidance(
        topic: str, 
        context: Optional[str] = None,
        include_source: bool = True
    ) -> str:
        """
        Get implementation guidance for a specific Hapax feature or concept.
        
        Args:
            topic: The feature or concept to get guidance on
            context: Optional additional context about the implementation
            include_source: Whether to include source code examples
            
        Returns:
            Implementation guidance with relevant sections, code examples, and source code
        """
        doc_index = mcp.current_request_context.lifespan_context
        
        # Search for relevant content
        search_results = search_content(doc_index, topic, include_source=include_source)
        
        # Get relevant sections
        sections = doc_index.get_section(topic)
        formatted_sections = []
        for section in sections[:3]:  # Top 3 matching sections
            formatted_sections.append({
                "title": section.title,
                "content": section.content,
                "file_path": section.file_path
            })
        
        # Get relevant code examples
        examples = doc_index.get_code_example(topic)
        formatted_examples = []
        for example in examples[:3]:  # Top 3 code examples
            formatted_examples.append({
                "pattern_name": example.pattern_name,
                "description": example.description,
                "code": example.code,
                "file_path": example.file_path
            })
        
        # Get relevant source elements if requested
        formatted_elements = []
        if include_source:
            elements = doc_index.get_source_element(topic)
            for element in elements[:3]:  # Top 3 source elements
                formatted_elements.append({
                    "name": element.name,
                    "type": element.type,
                    "file_path": element.file_path,
                    "docstring": element.docstring,
                    "code_snippet": element.code[:500] + "..." if len(element.code) > 500 else element.code,
                    "parent": element.parent
                })
        
        # Format search results
        formatted_search_results = []
        for result in search_results[:5]:  # Top 5 search results
            formatted_search_results.append({
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type
            })
        
        return json.dumps({
            "topic": topic,
            "context": context,
            "matching_sections": formatted_sections,
            "code_examples": formatted_examples,
            "source_elements": formatted_elements,
            "search_results": formatted_search_results
        })

    @mcp.tool()
    def troubleshoot_issue(
        error_message: str,
        code_snippet: Optional[str] = None
    ) -> str:
        """
        Get troubleshooting guidance for a Hapax-related error or issue.
        
        Args:
            error_message: The error message or description of the issue
            code_snippet: Optional code snippet related to the issue
            
        Returns:
            Troubleshooting guidance with potential solutions
        """
        doc_index = mcp.current_request_context.lifespan_context
        
        # Extract key terms from the error message
        error_terms = re.findall(r'\b\w+\b', error_message.lower())
        error_terms = [term for term in error_terms if len(term) > 3]
        
        # Look for relevant troubleshooting sections in documentation
        troubleshooting_results = []
        for term in error_terms:
            results = search_content(doc_index, term, include_source=False)
            troubleshooting_results.extend(results)
        
        # Look for similar issues in source code
        source_results = []
        if code_snippet:
            # Extract function/class names from code snippet
            code_terms = re.findall(r'(?:def|class)\s+(\w+)', code_snippet)
            for term in code_terms + error_terms:
                if len(term) > 3:
                    results = search_content(doc_index, term, include_source=True)
                    source_results.extend([r for r in results if r.result_type.startswith("source_")])
        
        # Prioritize results from troubleshooting guide
        troubleshooting_results.sort(key=lambda x: 
            3 if "troubleshooting" in x.file_path.lower() else 
            2 if "error" in getattr(x, "context", "").lower() else
            1
        , reverse=True)
        
        # Prioritize source results based on relevance to error terms
        source_results.sort(key=lambda x:
            sum(1 for term in error_terms if term in getattr(x, "match", "").lower() or 
                                            term in getattr(x, "code", "").lower())
        , reverse=True)
        
        # Format troubleshooting results
        formatted_troubleshooting = []
        for result in troubleshooting_results[:5]:  # Top 5 relevant troubleshooting sections
            formatted_troubleshooting.append({
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type
            })
        
        # Format source results
        formatted_source = []
        for result in source_results[:5]:  # Top 5 relevant source code examples
            formatted_source.append({
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type
            })
        
        return json.dumps({
            "error_message": error_message,
            "code_snippet": code_snippet,
            "extracted_terms": error_terms,
            "troubleshooting_guidance": formatted_troubleshooting,
            "source_guidance": formatted_source
        })

    @mcp.tool()
    def understand_hapax_component(component_name: str) -> str:
        """
        Get a comprehensive understanding of a Hapax component.
        
        Args:
            component_name: The name of the component to understand (e.g., 'Graph', 'Operation', 'ops')
            
        Returns:
            Complete information about the component including documentation, source code, and examples
        """
        doc_index = mcp.current_request_context.lifespan_context
        
        # 1. Find documentation sections about this component
        sections = doc_index.get_section(component_name)
        formatted_sections = []
        for section in sections:
            formatted_sections.append({
                "title": section.title,
                "content": section.content,
                "file_path": section.file_path
            })
        
        # 2. Find source code implementation
        elements = doc_index.get_source_element(component_name)
        formatted_elements = []
        for element in elements:
            formatted_elements.append({
                "name": element.name,
                "type": element.type,
                "file_path": element.file_path,
                "docstring": element.docstring,
                "code": element.code,
                "parent": element.parent
            })
        
        # 3. Find usage examples
        usage_results = find_usage_examples(doc_index, component_name)
        formatted_usage = []
        for result in usage_results:
            formatted_usage.append({
                "file_path": result.file_path,
                "line": result.line,
                "match": result.match,
                "context": result.context,
                "type": result.result_type
            })
        
        # 4. Find code examples in documentation
        examples = doc_index.get_code_example(component_name)
        formatted_examples = []
        for example in examples:
            formatted_examples.append({
                "pattern_name": example.pattern_name,
                "description": example.description,
                "code": example.code,
                "file_path": example.file_path
            })
        
        # 5. Find related search results
        search_results = search_content(doc_index, component_name)
        formatted_search = []
        for result in search_results[:10]:
            if (result.file_path not in [s["file_path"] for s in formatted_sections] and
                result.file_path not in [e["file_path"] for e in formatted_elements]):
                formatted_search.append({
                    "file_path": result.file_path,
                    "line": result.line,
                    "match": result.match,
                    "context": result.context,
                    "type": result.result_type
                })
        
        return json.dumps({
            "component": component_name,
            "documentation": formatted_sections,
            "implementation": formatted_elements,
            "usage_examples": formatted_usage[:10],
            "documentation_examples": formatted_examples,
            "related_results": formatted_search[:5]
        })

    @mcp.tool()
    def explore_hapax_topic(topic: str) -> str:
        """
        Explore a Hapax topic by finding all related information.
        
        Args:
            topic: The topic to explore (e.g., 'evaluation', 'type checking', 'graph building')
            
        Returns:
            All documentation and code related to the topic
        """
        doc_index = mcp.current_request_context.lifespan_context
        
        # Perform a broad search
        search_results = search_content(doc_index, topic)
        
        # Group results by type
        grouped_results = {
            "documentation_sections": [],
            "code_examples": [],
            "source_code": [],
            "other": []
        }
        
        for result in search_results:
            # Check documentation sections
            if "documentation" in result.result_type:
                # Check if this is a section title
                is_section = False
                for section_title in doc_index.sections.keys():
                    if topic.lower() in section_title.lower():
                        is_section = True
                        break
                
                if is_section:
                    sections = doc_index.get_section(topic)
                    for section in sections:
                        grouped_results["documentation_sections"].append({
                            "title": section.title,
                            "content": section.content,
                            "file_path": section.file_path
                        })
                else:
                    grouped_results["documentation_sections"].append({
                        "file_path": result.file_path,
                        "line": result.line,
                        "match": result.match,
                        "context": result.context
                    })
            
            # Check code examples
            elif "example" in result.result_type:
                grouped_results["code_examples"].append({
                    "file_path": result.file_path,
                    "line": result.line,
                    "match": result.match,
                    "context": result.context,
                    "code": getattr(result, "code", None)
                })
            
            # Check source code
            elif result.result_type.startswith("source_"):
                grouped_results["source_code"].append({
                    "file_path": result.file_path,
                    "line": result.line,
                    "name": getattr(result, "name", None),
                    "type": result.result_type,
                    "match": result.match,
                    "context": result.context,
                    "code": getattr(result, "code", None)
                })
            
            # Other results
            else:
                grouped_results["other"].append({
                    "file_path": result.file_path,
                    "line": result.line,
                    "match": result.match,
                    "context": result.context
                })
        
        # Remove duplicates
        for key in grouped_results:
            unique_items = {}
            for item in grouped_results[key]:
                file_line = f"{item.get('file_path')}:{item.get('line', 0)}"
                if file_line not in unique_items:
                    unique_items[file_line] = item
            
            grouped_results[key] = list(unique_items.values())
        
        return json.dumps({
            "topic": topic,
            "documentation_sections": grouped_results["documentation_sections"][:10],
            "code_examples": grouped_results["code_examples"][:10],
            "source_code": grouped_results["source_code"][:10],
            "other_results": grouped_results["other"][:5]
        }) 