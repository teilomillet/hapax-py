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
    
    def __call__(self, input_data: T) -> U:
        """Execute the graph on input data."""
        if not self._operations:
            raise ValueError("Graph has no operations")
        
        return self._operations[0](input_data)
    
    def __rshift__(self, other: Union[BaseOperation[U, V], 'Graph[U, V]']) -> 'Graph[T, V]':
        """Support the >> operator for composing graphs."""
        if isinstance(other, Graph):
            other = other._operations[0]
        return self.then(other)
