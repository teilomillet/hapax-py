"""Base classes for Hapax operations and flow control."""
from typing import TypeVar, Generic, Any

T = TypeVar('T')
U = TypeVar('U')

class BaseOperation(Generic[T, U]):
    """Base class for all operations in Hapax."""
    def __init__(self, name: str):
        self.name = name
        self._input_type = None
        self._output_type = None
    
    def __call__(self, input_data: T) -> U:
        raise NotImplementedError
    
    def __rshift__(self, other: 'BaseOperation[U, Any]') -> 'BaseOperation[T, Any]':
        """Operator overload for >> to compose operations."""
        return self.compose(other)
    
    def compose(self, other: 'BaseOperation[U, Any]') -> 'BaseOperation[T, Any]':
        """Compose this operation with another operation."""
        raise NotImplementedError
