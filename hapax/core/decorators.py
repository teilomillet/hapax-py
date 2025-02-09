"""Decorators for defining operations and graphs in a functional style."""
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, get_type_hints

from .models import Operation, OpConfig

T = TypeVar('T')
U = TypeVar('U')

def ops(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to create an Operation from a function.
    
    Example:
        @ops(name="tokenize", tags=["nlp"])
        def tokenize(text: str) -> List[str]:
            return text.split()
    """
    def decorator(func: Callable[[T], U]) -> Operation[T, U]:
        op_name = name or func.__name__
        hints = get_type_hints(func)
        
        # Extract input and output types
        if "return" not in hints:
            raise TypeError(f"Operation {op_name} must specify return type annotation")
        
        input_type = next(iter(hints.values()))  # First parameter type
        output_type = hints["return"]
        
        config = OpConfig(
            name=op_name,
            description=description or func.__doc__,
            tags=tags or [],
            metadata=metadata or {},
            openlit_config=openlit_config,
        )
        
        return Operation(
            func=func,
            config=config,
            input_type=input_type,
            output_type=output_type,
        )
    
    return decorator

def graph(
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to create a Graph from a function that composes operations.
    
    Example:
        @graph(name="text_processing")
        def process_text(text: str) -> List[str]:
            return tokenize >> normalize >> filter_stops
    """
    def decorator(func: Callable) -> Operation:
        graph_name = name or func.__name__
        
        # The decorated function should return a composed operation
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, Operation):
                raise TypeError(
                    f"Graph {graph_name} must return a composed Operation, "
                    f"got {type(result)}"
                )
            return result
        
        return wrapper
    
    return decorator
