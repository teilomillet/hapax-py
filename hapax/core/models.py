"""Core models for Hapax."""
from typing import TypeVar, List, Dict, Any, Callable, Optional, Union, get_origin, get_args, get_type_hints
import inspect
from inspect import signature
import networkx as nx
from pydantic import BaseModel, Field, ConfigDict
import matplotlib.pyplot as plt
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from .base import BaseOperation

T = TypeVar('T')
U = TypeVar('U')

# Global OpenLit configuration
_OPENLIT_CONFIG: Optional[Dict[str, Any]] = None

def set_openlit_config(config: Optional[Dict[str, Any]]) -> None:
    """Set the global OpenLit configuration for all operations."""
    global _OPENLIT_CONFIG
    _OPENLIT_CONFIG = config

def get_openlit_config() -> Optional[Dict[str, Any]]:
    """Get the current global OpenLit configuration."""
    return _OPENLIT_CONFIG

class OpConfig(BaseModel):
    """Configuration for an operation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "tokenize",
                "description": "Split text into tokens",
                "tags": ["nlp", "preprocessing"],
                "metadata": {"version": "1.0.0"},
                "config": {
                    "otlp_endpoint": "http://localhost:4318",
                    "environment": "development"
                }
            }
        }
    )

    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generic configuration options for the operation",
    )

    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get the effective configuration, combining operation-specific config
        with the global OpenLit config if available.
        """
        global_config = get_openlit_config() or {}
        op_config = self.config or {}
        
        # Operation config takes precedence over global config
        return {**global_config, **op_config}

def _extract_type_hints(func: Callable) -> tuple[Any, Any]:
    """Extract input and output type hints from a function."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    # Get input type from first parameter
    first_param = next(iter(sig.parameters.values()))
    input_type = hints.get(first_param.name, Any)
    
    # Get return type
    output_type = hints.get('return', Any)
    
    return input_type, output_type

def _check_type(value: Any, expected_type: Any) -> bool:
    """Check if a value matches an expected type."""
    # Skip type checking for Any
    if expected_type is Any:
        return True
    
    # Skip type checking for functions
    if isinstance(value, Callable):
        return True
    
    origin = get_origin(expected_type)
    if origin is None:
        # Simple type like str, int, etc.
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # If isinstance fails (e.g. with Any), assume it's valid
            return True
    
    if origin is Union:
        # Handle Optional[T] and Union[T1, T2, ...]
        return any(_check_type(value, t) for t in get_args(expected_type))
    
    if origin is list:
        # Handle List[T]
        if not isinstance(value, list):
            return False
        item_type = get_args(expected_type)[0]
        # Skip type checking for List[Any]
        if item_type is Any:
            return True
        try:
            return all(_check_type(item, item_type) for item in value)
        except TypeError:
            # If type checking fails, assume it's valid
            return True
    
    if origin is dict:
        # Handle Dict[K, V]
        if not isinstance(value, dict):
            return False
        key_type, value_type = get_args(expected_type)
        # Skip type checking for Dict[Any, Any]
        if key_type is Any and value_type is Any:
            return True
        try:
            return all(
                _check_type(k, key_type) and _check_type(v, value_type)
                for k, v in value.items()
            )
        except TypeError:
            # If type checking fails, assume it's valid
            return True
    
    # For other generic types, just check the origin
    try:
        return isinstance(value, origin)
    except TypeError:
        # If isinstance fails, assume it's valid
        return True

class Operation(BaseOperation[T, U]):
    """An operation that transforms input data of type T to output data of type U."""
    
    def __init__(
        self,
        func: Callable[[T], U],
        config: Optional[OpConfig] = None,
        auto_map: bool = True
    ):
        """
        Initialize an Operation.
        
        Args:
            func: The function to execute
            config: Operation configuration
            auto_map: Whether to automatically map over list inputs
        """
        self.func = func
        self.config = config or OpConfig(name=func.__name__)
        self.auto_map = auto_map
        
        # Extract type hints from function signature
        self._input_type, self._output_type = _extract_type_hints(func)
        
        # Create tracer for telemetry
        self._tracer = trace.get_tracer(f"hapax.operation.{self.config.name}")
        
        # For debugging and visualization
        self.__name__ = self.config.name
        
        # Validate type compatibility for composition
        if hasattr(func, '_input_type') and hasattr(func, '_output_type'):
            if not self._is_type_compatible(self._output_type, func._input_type):
                raise TypeError(
                    f"Type mismatch in {func.__name__}: "
                    f"output type {self._output_type} is not compatible with input type {func._input_type}"
                )
    
    def _is_type_compatible(self, source_type: Any, target_type: Any) -> bool:
        """Check if source_type is compatible with target_type."""
        if target_type is Any or source_type is Any:
            return True
            
        # Handle Union types
        if get_origin(target_type) is Union:
            return any(self._is_type_compatible(source_type, t) for t in get_args(target_type))
            
        # Handle List types
        if get_origin(source_type) is list and get_origin(target_type) is list:
            source_elem_type = get_args(source_type)[0]
            target_elem_type = get_args(target_type)[0]
            return self._is_type_compatible(source_elem_type, target_elem_type)
            
        # Handle Dict types
        if get_origin(source_type) is dict and get_origin(target_type) is dict:
            source_key_type, source_val_type = get_args(source_type)
            target_key_type, target_val_type = get_args(target_type)
            return (self._is_type_compatible(source_key_type, target_key_type) and
                   self._is_type_compatible(source_val_type, target_val_type))
                   
        return source_type == target_type

    def __call__(self, input_data: T) -> U:
        """Execute the operation on input data."""
        # Get effective configuration
        effective_config = self.config.get_effective_config()
        
        # Start a new span for tracing
        with self._tracer.start_as_current_span(
            f"operation.{self.config.name}",
            attributes={
                "hapax.operation.name": self.config.name,
                "hapax.operation.tags": ",".join(self.config.tags),
                "hapax.operation.trace_content": effective_config.get("trace_content", False),
            }
        ) as span:
            try:
                # Handle automatic mapping over lists if enabled
                if self.auto_map and isinstance(input_data, list):
                    span.set_attribute("hapax.operation.auto_map", True)
                    span.set_attribute("hapax.operation.input_length", len(input_data))
                    
                    # Map the function over each element
                    result = [self.func(item) for item in input_data]
                    
                    span.set_attribute("hapax.operation.output_length", len(result))
                    span.set_status(Status(StatusCode.OK))
                    return result
                
                # Trace content if configured
                if effective_config.get("trace_content", False):
                    span.set_attribute("hapax.operation.input", str(input_data)[:1000])  # Truncate for safety
                
                # Execute the function
                result = self.func(input_data)
                
                # Trace output if configured
                if effective_config.get("trace_content", False):
                    span.set_attribute("hapax.operation.output", str(result)[:1000])  # Truncate for safety
                
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def compose(self, other: BaseOperation[U, Any]) -> BaseOperation[T, Any]:
        """Compose this operation with another operation."""
        def composed_func(x: T) -> Any:
            return other(self(x))
        
        return Operation(
            composed_func,
            config=other.config,
            auto_map=self.auto_map
        )

    def __rshift__(self, other: BaseOperation[U, Any]) -> BaseOperation[T, Any]:
        """Operator overload for >> to compose operations."""
        return self.compose(other)

    def visualize(self, filename: Optional[str] = None) -> None:
        """Generate a visual representation of this operation and its compositions."""
        G = nx.DiGraph()
        
        # Add nodes for this operation and its compositions
        G.add_node(self.config.name)
        pos = {self.config.name: (0, 0)}
        
        # Add edges for compositions
        current = self
        x = 1
        while hasattr(current, '_composed_ops') and current._composed_ops:
            next_op = current._composed_ops[0]
            G.add_node(next_op.config.name)
            G.add_edge(current.config.name, next_op.config.name)
            pos[next_op.config.name] = (x, 0)
            current = next_op
            x += 1
        
        # Draw the graph
        plt.figure(figsize=(10, 6))
        
        # Calculate node positions using hierarchical layout
        def hierarchical_layout(G):
            # Group nodes by level
            nodes_by_level = {}
            for node in G.nodes():
                level = G.nodes[node]['level']
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            # Calculate positions
            pos = {}
            levels = sorted(nodes_by_level.keys())
            
            for level in levels:
                nodes = nodes_by_level[level]
                for i, node in enumerate(nodes):
                    # X coordinate based on level (left to right)
                    x = level
                    # Y coordinate spaced evenly within level
                    if len(nodes) > 1:
                        y = (i - (len(nodes) - 1) / 2) / len(nodes)
                    else:
                        y = 0
                    pos[node] = (x, y)
            
            return pos
        
        pos = hierarchical_layout(G)
        
        # Scale and center the layout
        pos = {node: (x, y * 2) for node, (x, y) in pos.items()}  # Increase vertical spacing
        
        # Draw nodes with larger size and custom style
        node_colors = ['lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors='gray',
            linewidths=2
        )
        
        # Add labels with larger font
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=12,
            font_weight='bold',
            font_family='sans-serif'
        )
        
        # Draw edges with curved arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=30,
            connectionstyle='arc3,rad=0.2',  # Curved edges
            width=2
        )
        
        plt.title(f"Operation: {self.config.name}", pad=20, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

def ops(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> Callable[[Callable[[T], U]], 'Operation[T, U]']:
    """Decorator to create an Operation from a function."""
    def decorator(func: Callable[[T], U]) -> 'Operation[T, U]':
        # Extract description from docstring if not provided
        func_description = description
        if func_description is None:
            func_description = inspect.getdoc(func)
            if func_description:
                # Take first line of docstring
                func_description = func_description.split('\n')[0].strip()
        
        config = OpConfig(
            name=name,
            description=func_description,
            tags=tags or [],
            metadata=metadata or {},
            config=get_openlit_config()  # Use global OpenLit config
        )
        return Operation(
            func=func,
            config=config
        )
    return decorator

def graph(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to create a Graph from a function."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Extract description from docstring if not provided
        if description is None:
            func_description = inspect.getdoc(func)
            if func_description:
                # Take first line of docstring
                func_description = func_description.split('\n')[0].strip()
        
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Call the function to get the pipeline
            pipeline = func(*args, **kwargs)
            return pipeline
        return wrapper
    return decorator

class GraphData(BaseModel):
    """Data model for graph configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: Optional[str] = None
    operations: List[Union[Operation, BaseOperation]] = Field(default_factory=list)
    edges: List[tuple[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Graph:
    """
    Represents a computation graph as a composition of operations.
    The graph itself is an operation that can be composed with other operations.
    Supports complex flow control including branching, merging, and conditionals.
    All type checking is performed at graph definition time.
    """
    
    def __init__(self, name: str, description: Optional[str] = None, **data):
        """Initialize a new graph.
        
        Args:
            name: Name of the graph
            description: Optional description
            **data: Additional data for the graph
        """
        print(f"DEBUG: Graph.__init__ called with name={name}")
        self.data = GraphData(
            name=name,
            description=description,
            operations=[],
            edges=[],
            metadata=data.get('metadata', {}),
            **data
        )
        self._graph = nx.DiGraph()
        self._tracer = trace.get_tracer(__name__)
        self._merge_funcs: Dict[str, Callable] = {}
        print(f"DEBUG: Graph.__init__ completed. Methods: {[m for m in dir(self) if not m.startswith('_')]}")
    
    def __getattr__(self, name: str) -> Any:
        print(f"DEBUG: __getattr__ called for {name}")
        if name == 'execute':
            print("DEBUG: execute method not found in normal lookup")
            print(f"DEBUG: Available methods: {[m for m in dir(self) if not m.startswith('_')]}")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    @property
    def name(self) -> str:
        return self.data.name
    
    @property
    def operations(self) -> List[Union[Operation, BaseOperation]]:
        return self.data.operations
    
    @property
    def edges(self) -> List[tuple[str, str]]:
        return self.data.edges
    
    def branch(self, *operations: Union[Operation, BaseOperation]) -> 'Graph':
        """Create parallel branches in the graph."""
        # Add operations to the graph
        self.data.operations.extend(operations)
        
        # Create a branch node
        branch_name = f"{self.name}_branch_{len(self._merge_funcs)}"
        
        # Add edges from branch to each operation
        for op in operations:
            self.data.edges.append((branch_name, op.name))
        
        # Rebuild and validate the graph
        self._build_and_validate_graph()
        return self
    
    def merge(self, merge_func: Callable[[List[Any]], Any]) -> 'Graph':
        """Merge multiple branch outputs using the provided function."""
        # Create merge node
        merge_name = f"{self.name}_merge_{len(self._merge_funcs)}"
        self._merge_funcs[merge_name] = merge_func
        
        # Connect all leaf operations to merge
        for op in self.operations:
            if not any(src == op.name for src, _ in self.edges):  # If op has no outgoing edges
                self.data.edges.append((op.name, merge_name))
        
        # Rebuild and validate the graph
        self._build_and_validate_graph()
        return self
    
    def _build_and_validate_graph(self):
        """Build and validate the graph at definition time."""
        # Clear existing graph
        self._graph.clear()
        
        # Add all operations as nodes
        for op in self.operations:
            self._graph.add_node(op.name, operation=op)
        
        # Add all edges
        for src, dst in self.edges:
            self._graph.add_edge(src, dst)
        
        # Validate the graph
        self._validate_graph_structure()
        self._validate_type_compatibility()
    
    def _validate_graph_structure(self):
        """Validate graph structure at definition time."""
        # Remove loop edges for cycle detection
        non_loop_graph = self._graph.copy()
        for node in self._graph.nodes:
            if isinstance(self._graph.nodes[node].get("operator"), BaseOperation):
                loop_op = self._graph.nodes[node]["operator"]
                non_loop_graph.remove_edge(loop_op.name, node)
        
        # Check for cycles
        cycles = list(nx.simple_cycles(non_loop_graph))
        if cycles:
            raise ValueError(f"Graph contains cycles: {cycles}")
    
    def _validate_type_compatibility(self):
        """Validate type compatibility between operations at definition time."""
        for src, dst in self.edges:
            src_node = self._graph.nodes[src]
            dst_node = self._graph.nodes[dst]
            
            if "operation" in src_node and "operation" in dst_node:
                src_op = src_node["operation"]
                dst_op = dst_node["operation"]
                
                if not hasattr(src_op, "_output_type") or not hasattr(dst_op, "_input_type"):
                    raise TypeError(
                        f"Operations must have type information. Check that {src} and {dst} "
                        "are decorated with @ops"
                    )
                
                if src_op._output_type != dst_op._input_type:
                    raise TypeError(
                        f"Type mismatch between {src} ({src_op._output_type}) "
                        f"and {dst} ({dst_op._input_type})"
                    )
    
    def execute(self, input_data: Any) -> Any:
        """Execute the graph with instrumentation."""
        print(f"DEBUG: execute called with input: {type(input_data)}")
        with self._tracer.start_as_current_span(
            name=f"graph.{self.name}",
            attributes={
                "graph.name": self.name,
                "graph.num_operations": len(self.operations),
                "graph.num_edges": len(self.edges)
            }
        ) as graph_span:
            try:
                # Track execution state
                node_results = {}
                execution_errors = []
                
                # Execute nodes in topological order
                for node_name in nx.topological_sort(self._graph):
                    with self._tracer.start_as_current_span(
                        name=f"node.{node_name}",
                        attributes={"node.type": "operation"}
                    ) as node_span:
                        try:
                            # Get input data from predecessors
                            predecessors = list(self._graph.predecessors(node_name))
                            if not predecessors:
                                node_input = input_data
                            else:
                                # Handle merge nodes
                                if node_name in self._merge_funcs:
                                    # Collect results from all predecessors in order
                                    merge_inputs = [node_results[pred] for pred in predecessors]
                                    result = self._merge_funcs[node_name](merge_inputs)
                                else:
                                    # Regular operation
                                    node_input = node_results[predecessors[0]]
                                    result = self._graph.nodes[node_name]["operation"](node_input)
                            
                            node_results[node_name] = result
                            node_span.set_status(Status(StatusCode.OK))
                            
                        except Exception as e:
                            node_span.set_status(Status(StatusCode.ERROR, str(e)))
                            node_span.record_exception(e)
                            execution_errors.append((node_name, e))
                            raise GraphExecutionError(
                                f"Error in node {node_name}: {str(e)}",
                                node_name=node_name,
                                node_errors=execution_errors,
                                partial_results=node_results
                            ) from e
                
                # Get result from terminal nodes (nodes with no successors)
                terminal_nodes = [n for n in self._graph.nodes if not list(self._graph.successors(n))]
                if len(terminal_nodes) == 1:
                    final_result = node_results[terminal_nodes[0]]
                else:
                    final_result = [node_results[n] for n in terminal_nodes]
                
                graph_span.set_status(Status(StatusCode.OK))
                return final_result
                
            except Exception as e:
                graph_span.set_status(Status(StatusCode.ERROR, str(e)))
                graph_span.record_exception(e)
                raise
    
    def validate(self) -> None:
        """
        Validate the graph:
        - Check for cycles (except in Loop operators)
        - Validate type compatibility between connected operations
        - Ensure all required metadata is present
        - Validate flow operator configurations
        """
        # Remove loop edges for cycle detection
        non_loop_graph = self._graph.copy()
        for node in self._graph.nodes:
            if isinstance(self._graph.nodes[node].get("operator"), BaseOperation):
                # Remove the loop edge but keep the operation edge
                loop_op = self._graph.nodes[node]["operator"]
                non_loop_graph.remove_edge(loop_op.name, node)
        
        # Check for cycles in non-loop graph
        try:
            cycles = list(nx.simple_cycles(non_loop_graph))
            if cycles:
                raise ValueError(f"Graph contains cycles: {cycles}")
        except Exception as e:
            raise GraphValidationError("Cycle validation failed", error=e)
        
        # Validate type compatibility
        for src, dst in self.edges:
            src_node = self._graph.nodes[src]
            dst_node = self._graph.nodes[dst]
            
            try:
                if "operation" in src_node and "operation" in dst_node:
                    if src_node["operation"].output_type != dst_node["operation"].input_type:
                        raise TypeError(
                            f"Type mismatch between {src} ({src_node['operation'].output_type}) "
                            f"and {dst} ({dst_node['operation'].input_type})"
                        )
            except Exception as e:
                raise GraphValidationError(
                    f"Type validation failed between {src} and {dst}",
                    error=e
                )

    def visualize(self, layout: str = 'sugiyama', interactive: bool = True) -> None:
        """
        Generate an enhanced visualization of the graph using networkx and matplotlib.
        
        Args:
            layout: Layout algorithm to use ('sugiyama', 'spring', 'shell')
            interactive: Whether to enable interactive features
        """
        G = nx.DiGraph()
        
        # Add nodes with enhanced metadata
        for op in self.operations:
            node_label = f"{op.config.name}"
            details = {
                'input_type': op.input_type.__name__,
                'output_type': op.output_type.__name__,
                'description': op.config.description or '',
                'tags': op.config.tags or [],
                'type': op.__class__.__name__
            }
            G.add_node(node_label, **details)
            
        # Add edges
        for src, dst in self.edges:
            G.add_edge(src, dst)
            
        # Choose layout algorithm
        if layout == 'sugiyama':
            pos = nx.multipartite_layout(G, subset_key='layer')
        elif layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:  # shell layout
            pos = nx.shell_layout(G)
            
        # Setup plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes with colors based on operation type
        node_colors = [self._get_node_color(G.nodes[node]['type']) for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.7)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Add labels with input/output types
        labels = {
            node: f"{node}\n{G.nodes[node]['input_type']}->{G.nodes[node]['output_type']}"
            for node in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        if interactive:
            # Add hover annotations
            def hover(event):
                if event.inaxes != plt.gca():
                    return
                for node, (x, y) in pos.items():
                    if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                        details = G.nodes[node]
                        text = (
                            f"Operation: {node}\n"
                            f"Type: {details['type']}\n"
                            f"Input: {details['input_type']}\n"
                            f"Output: {details['output_type']}\n"
                            f"Tags: {', '.join(details['tags'])}\n"
                            f"{details['description']}"
                        )
                        # Remove old annotations
                        for child in plt.gca().get_children():
                            if isinstance(child, plt.Annotation):
                                child.remove()
                        # Add new annotation
                        plt.annotate(
                            text,
                            xy=(x, y), xytext=(20, 20),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
                        plt.draw()
                        return
            
            plt.connect('motion_notify_event', hover)
            
        plt.title(f"Graph: {self.name}")
        plt.axis('off')
        plt.tight_layout()
        
    def _get_node_color(self, op_type: str) -> str:
        """Get color based on operation type."""
        color_map = {
            'Operation': '#2ecc71',  # Green for basic operations
            'Branch': '#e74c3c',     # Red for flow control
            'Merge': '#3498db',      # Blue for aggregation
            'Loop': '#f1c40f',       # Yellow for loops
            'Condition': '#9b59b6'   # Purple for conditionals
        }
        return color_map.get(op_type, '#95a5a6')  # Gray for unknown types

class GraphExecutionError(Exception):
    """Detailed error information for graph execution failures."""
    def __init__(
        self,
        message: str,
        node_name: str,
        node_errors: List[tuple[str, Exception]],
        partial_results: Dict[str, Any]
    ):
        super().__init__(message)
        self.node_name = node_name
        self.node_errors = node_errors
        self.partial_results = partial_results

class GraphValidationError(Exception):
    """Detailed error information for graph validation failures."""
    def __init__(self, message: str, error: Exception):
        super().__init__(message)
        self.error = error
