"""Decorators for defining operations and graphs in a functional style."""
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, get_type_hints, Type, Union

from .models import Operation, OpConfig
from openlit.evals import Hallucination, BiasDetector, ToxicityDetector, All

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

class EvalConfig:
    """Configuration for evaluation decorators."""
    def __init__(
        self,
        evals: List[str] = None,
        threshold: float = 0.5,
        metadata: Dict[str, Any] = None,
        openlit_config: Dict[str, Any] = None
    ):
        self.evals = evals or ["all"]
        self.threshold = threshold
        self.metadata = metadata or {}
        self.openlit_config = openlit_config or {}

def eval(
    evals: Optional[List[str]] = None,
    threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to evaluate function outputs using OpenLit evaluations.
    
    Example:
        @eval(evals=["hallucination", "bias"], threshold=0.7)
        def generate_response(prompt: str) -> str:
            return "Generated response..."
            
    Args:
        evals: List of evaluations to run. Options: ["hallucination", "bias", "toxicity", "all"]
        threshold: Threshold score for evaluations (0.0 to 1.0)
        metadata: Additional metadata for evaluations
        openlit_config: Configuration for OpenLit
    """
    def decorator(func: Callable) -> Callable:
        config = EvalConfig(
            evals=evals,
            threshold=threshold,
            metadata=metadata,
            openlit_config=openlit_config
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Run evaluations
            eval_results = {}
            for eval_type in config.evals:
                if eval_type == "hallucination" or eval_type == "all":
                    hall = Hallucination(**config.openlit_config)
                    eval_results["hallucination"] = hall.evaluate(result)
                
                if eval_type == "bias" or eval_type == "all":
                    bias = BiasDetector(**config.openlit_config)
                    eval_results["bias"] = bias.evaluate(result)
                    
                if eval_type == "toxicity" or eval_type == "all":
                    tox = ToxicityDetector(**config.openlit_config)
                    eval_results["toxicity"] = tox.evaluate(result)
            
            # Check if any evaluation failed
            failed_evals = [
                name for name, score in eval_results.items()
                if score > config.threshold
            ]
            
            if failed_evals:
                raise EvaluationError(
                    f"Evaluations failed: {failed_evals}. "
                    f"Scores: {eval_results}"
                )
            
            return result
        
        return wrapper
    
    return decorator

class EvaluationError(Exception):
    """Raised when an evaluation fails."""
    pass
