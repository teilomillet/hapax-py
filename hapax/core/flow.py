"""Flow control operators for Hapax graphs."""
from typing import List, Any, Callable, TypeVar, Optional
from .base import BaseOperation
from .models import Operation, OpConfig

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

class FlowOperator(BaseOperation[T, U]):
    """Base class for flow control operators."""
    def __init__(self, name: str, description: Optional[str] = None):
        super().__init__(name)
        self.config = OpConfig(
            name=name,
            description=description or self.__class__.__doc__.split('\n')[0]
        )
    
    def __call__(self, input_data: T) -> U:
        raise NotImplementedError

    def compose(self, other: BaseOperation[U, Any]) -> BaseOperation[T, Any]:
        """Compose this operator with another operation."""
        def composed_func(x: T) -> Any:
            return other(self(x))
        
        return Operation(
            composed_func,
            config=other.config,
            auto_map=True
        )

class Branch(FlowOperator[T, List[U]]):
    """Creates parallel branches in the graph."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.branches: List[BaseOperation] = []
        self.input_type = None
        self.output_type = None
    
    def add(self, *operations: BaseOperation) -> 'Branch[T, List[U]]':
        """Add parallel branches to execute."""
        self.branches.extend(operations)
        # Try to get type information from first operation
        if operations and hasattr(operations[0], '__annotations__'):
            self.input_type = operations[0].__annotations__.get('input_type', None)
            self.output_type = operations[0].__annotations__.get('return_type', None)
        return self
    
    def __call__(self, input_data: T) -> List[U]:
        """Execute all branches in parallel."""
        results = []
        errors = []
        
        for branch in self.branches:
            try:
                result = branch(input_data)
                results.append(result)
            except Exception as e:
                errors.append((branch.config.name, e))
        
        if errors:
            raise BranchError(
                f"Errors in branches: {errors}",
                branch_errors=errors,
                partial_results=results
            )
        
        return results

class Merge(FlowOperator[List[T], U]):
    """Merges multiple branch outputs into a single output."""
    
    def __init__(self, name: str, merge_func: Callable[[List[Any]], U]):
        super().__init__(name)
        self.merge_func = merge_func
        self.input_type = None
        self.output_type = None
        # Try to get type information from merge function
        if hasattr(merge_func, '__annotations__'):
            self.input_type = merge_func.__annotations__.get('args', [None])[0]
            self.output_type = merge_func.__annotations__.get('return', None)
    
    def __call__(self, inputs: List[T]) -> U:
        """Merge multiple inputs using the merge function."""
        try:
            return self.merge_func(inputs)
        except Exception as e:
            raise MergeError(
                f"Error merging inputs in {self.name}: {str(e)}",
                inputs=inputs
            )

class Condition(FlowOperator[T, U]):
    """Conditional branching based on a predicate."""
    
    def __init__(
        self,
        name: str,
        predicate: Callable[[T], bool],
        if_true: BaseOperation[T, U],
        if_false: Optional[BaseOperation[T, U]] = None
    ):
        super().__init__(name)
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false
    
    def __call__(self, input_data: T) -> U:
        """Execute the appropriate branch based on the predicate."""
        try:
            result = self.predicate(input_data)
            if result:
                return self.if_true(input_data)
            elif self.if_false:
                return self.if_false(input_data)
            else:
                return input_data  # type: ignore
        except Exception as e:
            raise ConditionError(
                f"Error in condition {self.name}: {str(e)}",
                input_data=input_data,
                predicate_result=result
            )

class Loop(FlowOperator[T, U]):
    """Loops an operation until a condition is met."""
    
    def __init__(
        self,
        name: str,
        operation: BaseOperation[T, U],
        condition: Callable[[U], bool],
        max_iterations: Optional[int] = None
    ):
        super().__init__(name)
        self.operation = operation
        self.condition = condition
        self.max_iterations = max_iterations
    
    def __call__(self, input_data: T) -> U:
        """Execute the operation until the condition is met."""
        result = input_data  # type: ignore
        iterations = 0
        last_error = None
        
        while True:
            if self.max_iterations and iterations >= self.max_iterations:
                break
            
            try:
                result = self.operation(result)
                if not self.condition(result):
                    break
            except Exception as e:
                last_error = e
                break
            
            iterations += 1
        
        if last_error:
            raise LoopError(
                f"Error in loop {self.name}: {str(last_error)}",
                iterations=iterations,
                last_error=last_error
            )
        
        return result

# Custom exceptions with rich error information
class BranchError(Exception):
    """Error in parallel branch execution."""
    def __init__(self, message: str, branch_errors: List[tuple], partial_results: List[Any]):
        super().__init__(message)
        self.branch_errors = branch_errors
        self.partial_results = partial_results

class MergeError(Exception):
    """Error merging branch results."""
    def __init__(self, message: str, inputs: List[Any]):
        super().__init__(message)
        self.inputs = inputs

class ConditionError(Exception):
    """Error in conditional execution."""
    def __init__(self, message: str, input_data: Any, predicate_result: bool):
        super().__init__(message)
        self.input_data = input_data
        self.predicate_result = predicate_result

class LoopError(Exception):
    """Error in loop execution."""
    def __init__(self, message: str, iterations: int, last_error: Exception):
        super().__init__(message)
        self.iterations = iterations
        self.last_error = last_error
