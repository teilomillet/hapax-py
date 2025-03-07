"""
Data models for the Hapax Documentation MCP server.

This module contains the dataclasses and models used for representing
documentation content, source code elements, and search indexes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CodeElement:
    """Represents a code element (function, class, method) with its docstring"""
    name: str
    type: str  # 'function', 'class', 'method'
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    code: str = ""
    parent: Optional[str] = None  # For methods, the parent class name

@dataclass
class DocSection:
    """Represents a section from documentation with title and content"""
    title: str
    content: str
    file_path: str
    line_start: int
    line_end: int

@dataclass
class CodeExample:
    """Represents a code example from documentation"""
    pattern_name: str
    code: str
    file_path: str
    line_start: int
    line_end: int
    description: str = ""

@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    file_path: str
    line: int
    match: str
    context: str
    result_type: str  # 'documentation', 'source_code', 'source_function', etc.
    score: float
    # Optional fields based on result_type
    name: Optional[str] = None
    docstring: Optional[str] = None
    code: Optional[str] = None

@dataclass
class DocIndex:
    """Index of documentation content and source code for fast searching"""
    # Documentation files
    file_contents: Dict[str, str] = field(default_factory=dict)
    sections: Dict[str, List[DocSection]] = field(default_factory=dict)
    code_examples: Dict[str, List[CodeExample]] = field(default_factory=dict)
    
    # Source code
    source_files: Dict[str, str] = field(default_factory=dict)
    code_elements: List[CodeElement] = field(default_factory=list)
    
    def add_file(self, file_path: str, content: str) -> None:
        """Add a documentation file to the index"""
        self.file_contents[file_path] = content
        
    def add_section(self, section: DocSection) -> None:
        """Add a documentation section to the index"""
        if section.title not in self.sections:
            self.sections[section.title] = []
        self.sections[section.title].append(section)
    
    def add_code_example(self, example: CodeExample) -> None:
        """Add a code example to the index"""
        if example.pattern_name not in self.code_examples:
            self.code_examples[example.pattern_name] = []
        self.code_examples[example.pattern_name].append(example)
    
    def add_source_file(self, file_path: str, content: str) -> None:
        """Add a source code file to the index"""
        self.source_files[file_path] = content
    
    def add_code_element(self, element: CodeElement) -> None:
        """Add a code element to the index"""
        self.code_elements.append(element)
    
    def get_section(self, section_name: str) -> List[DocSection]:
        """Get all occurrences of a named section"""
        # Try exact match first
        if section_name in self.sections:
            return self.sections[section_name]
        
        # Try case-insensitive match
        section_name_lower = section_name.lower()
        for title, sections in self.sections.items():
            if section_name_lower in title.lower():
                return sections
        
        return []
    
    def get_code_example(self, pattern_name: str) -> List[CodeExample]:
        """Get code examples for a specific pattern"""
        # Try exact match first
        if pattern_name in self.code_examples:
            return self.code_examples[pattern_name]
        
        # Try case-insensitive match
        pattern_name_lower = pattern_name.lower()
        for name, examples in self.code_examples.items():
            if pattern_name_lower in name.lower():
                return examples
        
        return []
    
    def get_source_element(self, element_name: str, element_type: Optional[str] = None) -> List[CodeElement]:
        """Get source code elements by name and optional type"""
        results = []
        
        # Try exact match first
        for element in self.code_elements:
            if element.name == element_name:
                if element_type is None or element.type == element_type:
                    results.append(element)
        
        # If no exact match, try case-insensitive
        if not results:
            element_name_lower = element_name.lower()
            for element in self.code_elements:
                if element_name_lower in element.name.lower():
                    if element_type is None or element.type == element_type:
                        results.append(element)
        
        return results 