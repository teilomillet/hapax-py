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
                "openlit_config": {
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
    openlit_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenLit configuration options",
        example={
            "otlp_endpoint": "http://localhost:4318",
            "environment": "development",
            "application_name": "my_app",
            "trace_content": True,
            "disable_metrics": False
        }
    )

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
    """An operation that transforms input type T to output type U."""
    
    def __init__(
        self,
        func: Callable[[T], U],
        config: Optional[OpConfig] = None,
        auto_map: bool = True
    ):
        """Initialize the operation.
        
        Args:
            func: The function to wrap
            config: Operation configuration
            auto_map: If True, automatically map over lists
        """
        super().__init__(config.name if config else "unnamed")
        self.func = func
        self.config = config
        self._auto_map = auto_map
        
        # Extract types from function signature
        sig = signature(func)
        # Get the first parameter's annotation (input type)
        param_types = list(sig.parameters.values())
        if not param_types:
            self._input_type = Any
        else:
            self._input_type = param_types[0].annotation
            if self._input_type == inspect.Parameter.empty:
                self._input_type = Any
        
        # Get return annotation (output type)
        self._output_type = sig.return_annotation
        if self._output_type == inspect.Parameter.empty:
            self._output_type = Any

    def __call__(self, input_data: T) -> U:
        """Execute the operation on input data."""
        # Handle list inputs when auto_map is True
        if self._auto_map and isinstance(input_data, list):
            if get_origin(self._input_type) is not list:
                # If input is a list but we expect a single item, map over the list
                return [self.func(item) for item in input_data]  # type: ignore
        
        if not _check_type(input_data, self._input_type):
            raise TypeError(
                f"Input type mismatch in {self.config.name if self.config else 'unnamed'}. "
                f"Expected {self._input_type}, got {type(input_data)}"
            )

        result = self.func(input_data)

        if not _check_type(result, self._output_type):
            raise TypeError(
                f"Output type mismatch in {self.config.name if self.config else 'unnamed'}. "
                f"Expected {self._output_type}, got {type(result)}"
            )

        return result

    def compose(self, other: BaseOperation[U, Any]) -> BaseOperation[T, Any]:
        """Compose this operation with another operation."""
        def composed_func(x: T) -> Any:
            return other(self(x))
        
        return Operation(
            composed_func,
            config=other.config,
            auto_map=self._auto_map
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
            openlit_config=get_openlit_config()  # Use global OpenLit config
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

class Graph(BaseModel):
    """
    Represents a computation graph as a composition of operations.
    The graph itself is an operation that can be composed with other operations.
    Supports complex flow control including branching, merging, and conditionals.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    operations: List[Union[Operation, BaseOperation]]
    edges: List[tuple[str, str]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        self._graph = nx.DiGraph()
        self._tracer = trace.get_tracer(__name__)
        
        # Build graph from operations and edges
        for op in self.operations:
            if isinstance(op, BaseOperation):
                # Flow operators might add multiple nodes and edges
                self._add_flow_operator(op)
            else:
                self._graph.add_node(op.config.name, operation=op)
        
        for src, dst in self.edges:
            self._graph.add_edge(src, dst)
    
    def _add_flow_operator(self, operator: BaseOperation):
        """Add a flow operator to the graph."""
        if isinstance(operator, BaseOperation):
            # Add branch node and connect to all branch operations
            self._graph.add_node(operator.name, operator=operator)
        
    def execute(self, input_data: Any) -> Any:
        """Execute the graph with instrumentation."""
        with self._tracer.start_as_current_span(
            name=f"graph.{self.name}",
            attributes={
                "graph.name": self.name,
                "graph.num_operations": len(self.operations),
                "graph.num_edges": len(self.edges)
            }
        ) as graph_span:
            try:
                # Validate graph before execution
                self.validate()
                
                # Track execution state
                node_results = {}
                execution_errors = []
                
                # Execute nodes in topological order where possible
                for node_name in nx.topological_sort(self._graph):
                    node = self._graph.nodes[node_name]
                    
                    with self._tracer.start_as_current_span(
                        name=f"node.{node_name}",
                        attributes={"node.type": type(node.get("operator", node.get("operation"))).__name__}
                    ) as node_span:
                        try:
                            # Get input data from predecessors
                            predecessors = list(self._graph.predecessors(node_name))
                            if not predecessors:
                                node_input = input_data
                            else:
                                # Handle multiple inputs for merge operations
                                if isinstance(node.get("operator"), BaseOperation):
                                    node_input = [node_results[pred] for pred in predecessors]
                                else:
                                    node_input = node_results[predecessors[0]]
                            
                            # Execute the node
                            if "operator" in node:
                                result = node["operator"](node_input)
                            else:
                                result = node["operation"](node_input)
                            
                            node_results[node_name] = result
                            node_span.set_status(Status(StatusCode.OK))
                            
                        except Exception as e:
                            node_span.set_status(Status(StatusCode.ERROR, str(e)))
                            node_span.record_exception(e)
                            execution_errors.append((node_name, e))
                            
                            # Propagate rich error information
                            if isinstance(e, (Exception)):
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

    def visualize(self) -> None:
        """
        Generate a visualization of the graph using networkx and matplotlib.
        This will show the flow of operations and their types.
        """
        G = nx.DiGraph()
        
        # Add nodes
        # First, calculate node levels using topological sort
        try:
            levels = {node: i for i, node_list in enumerate(nx.topological_generations(G)) for node in node_list}
        except nx.NetworkXUnfeasible:
            levels = {node: 0 for node in G.nodes()}  # Fallback if graph has cycles
        
        for op in self.operations:
            node_label = f"{op.config.name}"  # Keep the main label simple
            
            # Store detailed info in node attributes for the hover text
            details = [
                f"Input: {op.input_type.__name__}",
                f"Output: {op.output_type.__name__}"
            ]
            if op.config.description:
                details.append(op.config.description)
            if op.config.tags:
                details.append(f"Tags: {', '.join(op.config.tags)}")
            
            # Color based on tags
            if 'preprocessing' in op.config.tags:
                color = '#C8E6C9'  # Light green
            elif 'analysis' in op.config.tags:
                color = '#BBDEFB'  # Light blue
            else:
                color = '#F5F5F5'  # Light gray
            
            G.add_node(
                op.config.name,
                label=node_label,
                details='\n'.join(details),
                color=color,
                level=levels.get(op.config.name, 0)  # Use topological level
            )
        
        # Add edges
        for source, target in self.edges:
            G.add_edge(source, target)
        
        # Create the visualization with larger figure size
        plt.figure(figsize=(15, 10))
        
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
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors='gray',
            linewidths=2
        )
        
        # Add hover annotations
        def hover(event):
            if event.inaxes != plt.gca():
                return
            for node, (x, y) in pos.items():
                if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                    details = G.nodes[node]['details']
                    plt.annotate(
                        details,
                        xy=(x, y), xytext=(20, 20),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )
                    plt.draw()
                    return
        
        # Draw edges with curved arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=30,
            connectionstyle='arc3,rad=0.2',  # Curved edges
            width=2
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
        
        plt.title(f"Graph: {self.name}", pad=20, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add a legend for node types
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='#C8E6C9', ec='gray', label='Preprocessing'),
            plt.Rectangle((0, 0), 1, 1, fc='#BBDEFB', ec='gray', label='Analysis'),
            plt.Rectangle((0, 0), 1, 1, fc='#F5F5F5', ec='gray', label='Other')
        ]
        plt.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            fontsize=10
        )
        
        # Add hover functionality
        plt.gcf().canvas.mpl_connect('motion_notify_event', hover)
        
        # Add grid lines to show levels
        ax = plt.gca()
        levels = sorted(set(G.nodes[node]['level'] for node in G.nodes()))
        for level in levels:
            ax.axvline(x=level, color='lightgray', linestyle='--', alpha=0.5)
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        plt.show()
        plt.close()

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
