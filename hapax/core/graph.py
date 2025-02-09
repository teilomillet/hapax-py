"""Graph builder for Hapax."""
from typing import TypeVar, Generic, List, Any, Callable, Optional, Union
from .base import BaseOperation
from .models import Operation, OpConfig
from .flow import Branch, Merge, Condition, Loop

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

class Graph(Generic[T, U]):
    """A fluent API for building computation graphs.
    
    Example:
        graph = (
            Graph("text_processing")
            .then(clean_text)
            .branch(
                tokenize >> normalize,
                sentiment_analysis,
                language_detection
            )
            .merge(lambda results: {
                "tokens": results[0],
                "sentiment": results[1],
                "language": results[2]
            })
            .then(store_results)
        )
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """Initialize a new graph builder.
        
        Args:
            name: Name of the graph
            description: Optional description of the graph
        """
        self.name = name
        self.description = description
        self._operations: List[BaseOperation] = []
        self._current_branch: Optional[Branch] = None
        self._current_merge: Optional[Merge] = None
        self._current_condition: Optional[Condition] = None
        self._current_loop: Optional[Loop] = None
    
    def then(self, operation: Union[BaseOperation[T, U], Callable[[T], U]]) -> 'Graph[T, U]':
        """Add an operation to the graph.
        
        Args:
            operation: Operation to add, can be a BaseOperation or a function
        """
        if not isinstance(operation, BaseOperation):
            # If it's a function, wrap it in an Operation
            operation = Operation(
                func=operation,
                config=OpConfig(name=operation.__name__)
            )
        
        if self._operations:
            # Don't compose if either operation is a Branch or Merge
            if isinstance(operation, (Branch, Merge)) or isinstance(self._operations[-1], (Branch, Merge)):
                self._operations.append(operation)
            else:
                # Compose with the last operation
                self._operations[-1] = self._operations[-1].compose(operation)
        else:
            self._operations.append(operation)
        
        return self
    
    def branch(self, *operations: Union[BaseOperation[T, U], Callable[[T], U]]) -> 'Graph[T, List[U]]':
        """Create parallel branches in the graph.
        
        Args:
            *operations: Operations to execute in parallel
        """
        branch = Branch(f"{self.name}_branch_{len(self._operations)}")
        
        # Convert functions to Operations if needed
        ops = []
        for op in operations:
            if not isinstance(op, BaseOperation):
                op = Operation(
                    func=op,
                    config=OpConfig(name=op.__name__)
                )
            ops.append(op)
        
        branch.add(*ops)
        return self.then(branch)
    
    def merge(self, merge_func: Callable[[List[Any]], U]) -> 'Graph[List[T], U]':
        """Merge multiple branch outputs into a single output.
        
        Args:
            merge_func: Function to merge branch results
        """
        merge = Merge(f"{self.name}_merge_{len(self._operations)}", merge_func)
        return self.then(merge)
    
    def condition(
        self,
        predicate: Callable[[T], bool],
        if_true: Union[BaseOperation[T, U], Callable[[T], U]],
        if_false: Optional[Union[BaseOperation[T, U], Callable[[T], U]]] = None
    ) -> 'Graph[T, U]':
        """Add conditional branching to the graph.
        
        Args:
            predicate: Function that returns True/False
            if_true: Operation to execute if predicate is True
            if_false: Optional operation to execute if predicate is False
        """
        # Convert functions to Operations if needed
        if not isinstance(if_true, BaseOperation):
            if_true = Operation(
                func=if_true,
                config=OpConfig(name=if_true.__name__)
            )
        
        if if_false and not isinstance(if_false, BaseOperation):
            if_false = Operation(
                func=if_false,
                config=OpConfig(name=if_false.__name__)
            )
        
        condition = Condition(
            f"{self.name}_condition_{len(self._operations)}",
            predicate,
            if_true,
            if_false
        )
        return self.then(condition)
    
    def loop(
        self,
        operation: Union[BaseOperation[T, U], Callable[[T], U]],
        condition: Callable[[U], bool],
        max_iterations: Optional[int] = None
    ) -> 'Graph[T, U]':
        """Add a loop to the graph.
        
        Args:
            operation: Operation to repeat
            condition: Function that returns True to continue looping
            max_iterations: Optional maximum number of iterations
        """
        if not isinstance(operation, BaseOperation):
            operation = Operation(
                func=operation,
                config=OpConfig(name=operation.__name__)
            )
        
        loop = Loop(
            f"{self.name}_loop_{len(self._operations)}",
            operation,
            condition,
            max_iterations
        )
        return self.then(loop)
    
    def execute(self, input_data: T) -> U:
        """Execute the graph with the given input data."""
        if not self._operations:
            raise ValueError("Graph has no operations")
        
        result = input_data
        
        # Handle branching and merging
        if self._current_branch:
            branch_results = [op(input_data) for op in self._current_branch.branches]
            if self._current_merge:
                result = self._current_merge(branch_results)
            else:
                result = branch_results
        
        # Handle conditional
        elif self._current_condition:
            condition, if_true, if_false = self._current_condition
            if condition(input_data):
                result = if_true(input_data)
            else:
                result = if_false(input_data)
        
        # Handle loop
        elif self._current_loop:
            operation, condition, max_iterations = self._current_loop
            iterations = 0
            while condition(result) and iterations < max_iterations:
                result = operation(result)
                iterations += 1
        
        # Handle sequential operations
        else:
            for op in self._operations:
                result = op(result)
        
        return result
    
    def __rshift__(self, other: Union[BaseOperation[U, V], 'Graph[U, V]']) -> 'Graph[T, V]':
        """Support the >> operator for composing graphs."""
        if isinstance(other, Graph):
            other = other._operations[0]
        return self.then(other)
    
    def visualize(self) -> None:
        """Generate a simple DAG visualization with fixed positions."""
        print("Starting visualization...")
        print(f"Operations: {[type(op).__name__ for op in self._operations]}")
        
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError as e:
            print(f"Failed to import visualization libraries: {e}")
            return
        
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Get branch and merge operations
        branch_ops = []
        merge_point = None
        input_type = None
        output_type = None
        
        print("\nScanning operations...")
        for op in self._operations:
            print(f"Processing operation: {type(op).__name__}")
            if isinstance(op, Branch):
                branch_ops = [(b.name if hasattr(b, 'name') else str(b)) for b in op.branches]
                print(f"Found branches: {branch_ops}")
                # Get input type from branch operation
                input_type = op.input_type if hasattr(op, 'input_type') else None
            elif isinstance(op, Merge):
                merge_point = op.name if hasattr(op, 'name') else str(op)
                print(f"Found merge point: {merge_point}")
                # Get output type from merge operation
                output_type = op.output_type if hasattr(op, 'output_type') else None
        
        if not branch_ops or not merge_point:
            print("\nMissing required components:")
            print(f"- Branch operations found: {bool(branch_ops)}")
            print(f"- Merge point found: {bool(merge_point)}")
            return
            
        # Use type information in node labels
        input_label = f"Input\n({input_type.__name__ if input_type else 'Any'})"
        output_label = f"Output\n({output_type.__name__ if output_type else 'Any'})"
        
        # Create nodes with type information
        G.add_node(input_label)
        for op in branch_ops:
            G.add_node(op)
        G.add_node(merge_point)
        G.add_node(output_label)
        
        # Create edges
        for op in branch_ops:
            G.add_edge(input_label, op)
            G.add_edge(op, merge_point)
        G.add_edge(merge_point, output_label)
        
        # Fixed positions for perfect DAG layout
        pos = {
            input_label: (0, 0),
            merge_point: (2, 0),
            output_label: (3, 0)
        }
        
        # Position branch operations in between input and merge
        num_branches = len(branch_ops)
        for i, op in enumerate(branch_ops):
            y_pos = (i - (num_branches - 1) / 2) * 0.5  # Spread vertically
            pos[op] = (1, y_pos)  # All branch ops at x=1
        
        # Draw the graph
        plt.figure(figsize=(10, 6))
        
        # Draw nodes with different colors for input/output
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in (input_label, output_label):
                node_colors.append('lightgreen')
                node_sizes.append(2500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(2000)
        
        nx.draw_networkx_nodes(G, pos, 
                             node_color=node_colors,
                             node_size=node_sizes, 
                             alpha=0.7)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, 
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             connectionstyle='arc3,rad=0.2')  # Curved edges
        
        # Add labels with word wrap
        labels = {node: '\n'.join(str(node).split()) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
        
        plt.title(f"Graph: {self.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
