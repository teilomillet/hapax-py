"""
Documentation and source code indexer for the Hapax Documentation MCP server.

This module contains functions to parse and index documentation files 
and source code for fast searching and retrieval.
"""

import ast
import re
from pathlib import Path
from typing import List

from .models import DocIndex, DocSection, CodeExample, CodeElement, SearchResult

def extract_sections(file_path: str, content: str) -> List[DocSection]:
    """Extract titled sections from documentation file"""
    sections = []
    lines = content.split('\n')
    
    current_section = None
    section_start = 0
    
    for i, line in enumerate(lines):
        if line.strip() and all(c == '-' for c in line.strip()):
            # This is a title underline, title is on the previous line
            if i > 0 and lines[i-1].strip():
                # If we were in a section, close it
                if current_section:
                    sections.append(DocSection(
                        title=current_section,
                        content='\n'.join(lines[section_start:i-1]),
                        file_path=file_path,
                        line_start=section_start,
                        line_end=i-1
                    ))
                
                # Start a new section
                current_section = lines[i-1].strip()
                section_start = i+1
    
    # Add the last section if there is one
    if current_section:
        sections.append(DocSection(
            title=current_section,
            content='\n'.join(lines[section_start:]),
            file_path=file_path,
            line_start=section_start,
            line_end=len(lines)
        ))
        
    return sections

def extract_code_examples(file_path: str, content: str) -> List[CodeExample]:
    """Extract code examples from documentation file"""
    examples = []
    lines = content.split('\n')
    
    in_code_block = False
    code_start = 0
    current_code = []
    pattern_name = "general"
    description = ""
    
    for i, line in enumerate(lines):
        # Check for pattern markers
        if "PATTERN" in line and ":" in line:
            pattern_parts = line.split(":", 1)
            if len(pattern_parts) > 1:
                pattern_name = pattern_parts[0].strip()
                description = pattern_parts[1].strip()
        
        # Check for code block start/end
        if line.strip().startswith("```python"):
            in_code_block = True
            code_start = i+1
            current_code = []
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            examples.append(CodeExample(
                pattern_name=pattern_name,
                code='\n'.join(current_code),
                file_path=file_path,
                line_start=code_start,
                line_end=i,
                description=description
            ))
        elif in_code_block:
            current_code.append(line)
            
    return examples

def extract_code_elements(file_path: str, tree: ast.AST, content: str) -> List[CodeElement]:
    """Extract functions, classes, and methods from a Python AST"""
    elements = []
    lines = content.split('\n')
    
    # Helper functions for AST traversal
    def is_node_within_parent(node: ast.AST, potential_parent: ast.AST) -> bool:
        """Check if a node is within the scope of a potential parent node"""
        return (node.lineno > potential_parent.lineno and 
                hasattr(potential_parent, 'end_lineno') and 
                (potential_parent.end_lineno is None or node.lineno < potential_parent.end_lineno))
    
    def find_node_end_line(node: ast.AST) -> int:
        """Find the end line number of a node"""
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            return node.end_lineno
        
        # Fall back to the next line after the node
        return node.lineno + 1
    
    # First pass: collect all classes
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node] = {
                'name': node.name,
                'lineno': node.lineno,
                'end_lineno': find_node_end_line(node)
            }
    
    # Second pass: collect functions and methods
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Determine if this is a method (inside a class)
            element_type = 'function'
            parent = None
            
            for class_node, class_info in classes.items():
                if (node.lineno > class_info['lineno'] and 
                    node.lineno < class_info['end_lineno']):
                    element_type = 'method'
                    parent = class_info['name']
                    break
            
            # Extract docstring
            docstring = ast.get_docstring(node)
            
            # Extract full code
            line_start = node.lineno
            line_end = find_node_end_line(node)
            code = '\n'.join(lines[line_start-1:line_end])
            
            elements.append(CodeElement(
                name=node.name,
                type=element_type,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                docstring=docstring,
                code=code,
                parent=parent
            ))
        
        elif isinstance(node, ast.ClassDef):
            # Extract class
            docstring = ast.get_docstring(node)
            line_start = node.lineno
            line_end = find_node_end_line(node)
            code = '\n'.join(lines[line_start-1:line_end])
            
            elements.append(CodeElement(
                name=node.name,
                type='class',
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                docstring=docstring,
                code=code
            ))
    
    return elements

def search_content(doc_index: DocIndex, query: str, include_source: bool = True) -> List[SearchResult]:
    """Search documentation and source code for a query"""
    results = []
    
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Search in documentation file contents
    for file_path, content in doc_index.file_contents.items():
        if query_lower in content.lower():
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    context_start = max(0, i-2)
                    context_end = min(len(lines), i+3)
                    results.append(SearchResult(
                        file_path=file_path,
                        line=i+1,
                        match=line.strip(),
                        context='\n'.join(lines[context_start:context_end]),
                        result_type="documentation",
                        score=1 if query_lower in line.lower() else 0.5
                    ))
    
    # Search in source code if requested
    if include_source:
        # Search in code elements (functions, classes, methods)
        for element in doc_index.code_elements:
            element_text = f"{element.code} {element.docstring or ''}"
            if query_lower in element_text.lower():
                results.append(SearchResult(
                    file_path=element.file_path,
                    line=element.line_start,
                    match=f"{element.type} {element.name}",
                    context=element.code[:200] + "..." if len(element.code) > 200 else element.code,
                    result_type=f"source_{element.type}",
                    score=2 if query_lower in element.name.lower() else 1,
                    name=element.name,
                    docstring=element.docstring,
                    code=element.code
                ))
        
        # Also search for strings in source files
        for file_path, content in doc_index.source_files.items():
            if query_lower in content.lower():
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if query_lower in line.lower():
                        context_start = max(0, i-2)
                        context_end = min(len(lines), i+3)
                        results.append(SearchResult(
                            file_path=file_path,
                            line=i+1,
                            match=line.strip(),
                            context='\n'.join(lines[context_start:context_end]),
                            result_type="source_code",
                            score=0.75 if query_lower in line.lower() else 0.25
                        ))
    
    # Sort results by score
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results

def find_usage_examples(doc_index: DocIndex, element_name: str) -> List[SearchResult]:
    """Find examples of how a function, class, or method is used in the code"""
    results = []
    
    # Look for usage in source code
    for file_path, content in doc_index.source_files.items():
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Look for the element name followed by ( or space
            pattern = rf'\b{re.escape(element_name)}\s*[\(\s]'
            if re.search(pattern, line):
                # Check if this isn't the definition itself
                if not re.search(rf'def\s+{re.escape(element_name)}\s*\(', line) and \
                   not re.search(rf'class\s+{re.escape(element_name)}\s*', line):
                    context_start = max(0, i-2)
                    context_end = min(len(lines), i+3)
                    results.append(SearchResult(
                        file_path=file_path,
                        line=i+1,
                        match=line.strip(),
                        context='\n'.join(lines[context_start:context_end]),
                        result_type="usage",
                        score=1.0
                    ))
    
    # Look for usage in documentation examples
    for pattern_name, examples in doc_index.code_examples.items():
        for example in examples:
            if element_name in example.code:
                results.append(SearchResult(
                    file_path=example.file_path,
                    line=example.line_start,
                    match=f"Example in pattern: {pattern_name}",
                    context=example.code,
                    result_type="example",
                    score=1.5,
                    code=example.code
                ))
    
    return results

def index_documentation(doc_index: DocIndex, docs_dir: Path) -> None:
    """Index all documentation files in a directory"""
    if not docs_dir.exists():
        print(f"Warning: Documentation directory {docs_dir} not found")
        return
    
    # Index all documentation files
    for file_path in docs_dir.glob("*.txt"):
        with open(file_path, "r") as f:
            content = f.read()
            doc_index.add_file(str(file_path), content)
            
            # Extract and index sections
            sections = extract_sections(str(file_path), content)
            for section in sections:
                doc_index.add_section(section)
            
            # Extract and index code examples
            examples = extract_code_examples(str(file_path), content)
            for example in examples:
                doc_index.add_code_example(example)

def index_source_code(doc_index: DocIndex, source_dir: Path) -> None:
    """Index all Python source files in a directory"""
    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} not found")
        return
    
    # Index all Python source files
    for file_path in source_dir.glob("**/*.py"):
        try:
            with open(file_path, "r") as f:
                content = f.read()
                doc_index.add_source_file(str(file_path), content)
                
                # Parse Python code to extract elements
                try:
                    tree = ast.parse(content)
                    elements = extract_code_elements(str(file_path), tree, content)
                    for element in elements:
                        doc_index.add_code_element(element)
                except SyntaxError as e:
                    print(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def create_index() -> DocIndex:
    """Create and populate a documentation and source code index"""
    doc_index = DocIndex()
    
    # Index documentation files
    docs_dir = Path("agent_docs")
    index_documentation(doc_index, docs_dir)
    
    # Index source code
    source_dir = Path("hapax")
    index_source_code(doc_index, source_dir)
    
    return doc_index 