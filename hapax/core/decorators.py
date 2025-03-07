"""Decorators for defining operations and graphs in a functional style."""
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, get_type_hints
import hashlib
import json
import logging
from dataclasses import dataclass, field

from .models import Operation, OpConfig

T = TypeVar('T')
U = TypeVar('U')

# Set up logging
logger = logging.getLogger(__name__)

# Registry for evaluation functions
EVALUATOR_REGISTRY: Dict[str, type] = {}

def register_evaluator(name: str, evaluator_cls: type) -> None:
    """Register an evaluator class for use with the @eval decorator."""
    EVALUATOR_REGISTRY[name] = evaluator_cls

# Cache for evaluation results
EVAL_CACHE: Dict[str, Dict[str, float]] = {}

@dataclass
class EvalConfig:
    """Configuration for evaluation decorators."""
    evals: List[str] = field(default_factory=lambda: ["all"])
    threshold: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    openlit_config: Dict[str, Any] = field(default_factory=dict)
    cache_results: bool = True
    use_openlit: bool = False
    openlit_provider: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if not self.evals:
            raise ValueError("Must specify at least one evaluation type")
        if not self.use_openlit:
            # Only validate registered evaluators if not using OpenLIT
            unknown_evals = set(self.evals) - set(EVALUATOR_REGISTRY.keys()) - {"all"}
            if unknown_evals:
                raise ValueError(f"Unknown evaluation types: {unknown_evals}")

def _get_cache_key(func: Callable, args: tuple, kwargs: dict, eval_type: str) -> str:
    """Generate a cache key for evaluation results."""
    # Convert args and kwargs to a stable string representation
    args_str = json.dumps(args, sort_keys=True)
    kwargs_str = json.dumps(kwargs, sort_keys=True)
    func_key = f"{func.__module__}.{func.__qualname__}"
    key_str = f"{func_key}:{args_str}:{kwargs_str}:{eval_type}"
    return hashlib.sha256(key_str.encode()).hexdigest()

def eval(
    evals: Optional[List[str]] = None,
    threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None,
    cache_results: bool = True,
    use_openlit: bool = False,
    openlit_provider: Optional[str] = None,
) -> Callable:
    """
    Decorator to evaluate function outputs using evaluations.
    
    Example:
        @eval(evals=["hallucination", "bias"], threshold=0.7)
        def generate_response(prompt: str) -> str:
            return "Generated response..."
            
    Args:
        evals: List of evaluations to run. Options are registered evaluator names
              or OpenLIT evaluations if use_openlit=True
        threshold: Threshold score for evaluations (0.0 to 1.0)
        metadata: Additional metadata for evaluations
        openlit_config: Configuration for OpenLIT evaluations
        cache_results: Whether to cache evaluation results
        use_openlit: Whether to use OpenLIT evaluations instead of registered evaluators
        openlit_provider: LLM provider for OpenLIT evaluations ("openai" or "anthropic")
    """
    def decorator(func: Callable) -> Callable:
        # Extract return type for validation
        hints = get_type_hints(func)
        if "return" not in hints:
            raise TypeError(f"Function {func.__name__} must specify return type annotation")
        return_type = hints["return"]
        
        # Validate that return type is compatible with evaluators
        if return_type is not str:
            raise TypeError(f"Evaluation only supports string outputs, got {return_type}")
        
        config = EvalConfig(
            evals=evals or ["all"],
            threshold=threshold,
            metadata=metadata or {},
            openlit_config=openlit_config or {},
            cache_results=cache_results,
            use_openlit=use_openlit,
            openlit_provider=openlit_provider,
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> str:
            result = func(*args, **kwargs)
            
            # Validate return type at runtime
            if not isinstance(result, str):
                raise TypeError(
                    f"Expected string output from {func.__name__}, "
                    f"got {type(result)}"
                )
            
            # Use OpenLIT evaluations if requested
            if config.use_openlit:
                try:
                    # Import necessary OpenLIT evaluators here to avoid circular imports
                    from hapax.evaluations import (
                        AllEvaluator, 
                        HallucinationEvaluator,
                        BiasEvaluator,
                        ToxicityEvaluator
                    )
                    
                    # Map evaluation types to OpenLIT evaluator classes
                    eval_map = {
                        "all": AllEvaluator,
                        "hallucination": HallucinationEvaluator,
                        "bias": BiasEvaluator,
                        "toxicity": ToxicityEvaluator
                    }
                    
                    # Determine which OpenLIT evaluations to run
                    all_evals = set(config.evals)
                    
                    # Context from metadata if available
                    contexts = config.metadata.get("contexts", []) if config.metadata else []
                    
                    # Get the prompt from metadata or first argument if it's a string
                    prompt = config.metadata.get("prompt", "")
                    if not prompt and args and isinstance(args[0], str):
                        prompt = args[0]
                    
                    # Run evaluations
                    eval_results = {}
                    for eval_type in all_evals:
                        if eval_type in eval_map:
                            evaluator_cls = eval_map[eval_type]
                            evaluator = evaluator_cls(
                                provider=config.openlit_provider or "openai",
                                threshold=config.threshold,
                                collect_metrics=True,
                                **(config.openlit_config or {})
                            )
                            
                            eval_result = evaluator.evaluate(
                                text=result,
                                contexts=contexts,
                                prompt=prompt
                            )
                            
                            # Store results
                            eval_results[eval_type] = eval_result
                    
                    # Check if any evaluation failed
                    failed_evals = [
                        name for name, data in eval_results.items()
                        if data.get("verdict") == "yes"
                    ]
                    
                    if failed_evals:
                        raise EvaluationError(
                            f"Evaluations failed: {failed_evals}. "
                            f"Results: {eval_results}"
                        )
                    
                except ImportError:
                    logger.warning("OpenLIT not installed. Skipping OpenLIT evaluations.")
                
            else:
                # Original evaluation code for registered evaluators
                eval_results = {}
                for eval_type in config.evals:
                    # Check cache first
                    if config.cache_results:
                        cache_key = _get_cache_key(func, args, kwargs, eval_type)
                        if cache_key in EVAL_CACHE:
                            eval_results.update(EVAL_CACHE[cache_key])
                            continue
                    
                    # Get evaluator from registry
                    if eval_type in EVALUATOR_REGISTRY:
                        evaluator = EVALUATOR_REGISTRY[eval_type](**config.openlit_config)
                        score = evaluator.evaluate(result)
                        eval_results[eval_type] = score
                        
                        # Cache result
                        if config.cache_results:
                            EVAL_CACHE[cache_key] = {eval_type: score}
                    
                    elif eval_type == "all":
                        # Run all registered evaluators
                        for name, evaluator_cls in EVALUATOR_REGISTRY.items():
                            evaluator = evaluator_cls(**config.openlit_config)
                            score = evaluator.evaluate(result)
                            eval_results[name] = score
                            
                            # Cache result
                            if config.cache_results:
                                cache_key = _get_cache_key(func, args, kwargs, name)
                                EVAL_CACHE[cache_key] = {name: score}
                
                # Check if any evaluation failed with original method
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

# Register built-in evaluators
if "hallucination" not in EVALUATOR_REGISTRY:
    try:
        from examples.mock_evals import Hallucination
        register_evaluator("hallucination", Hallucination)
    except ImportError:
        logger.warning("Could not import mock evaluator: Hallucination")

if "bias" not in EVALUATOR_REGISTRY:
    try:
        from examples.mock_evals import BiasDetector
        register_evaluator("bias", BiasDetector)
    except ImportError:
        logger.warning("Could not import mock evaluator: BiasDetector")

if "toxicity" not in EVALUATOR_REGISTRY:
    try:
        from examples.mock_evals import ToxicityDetector
        register_evaluator("toxicity", ToxicityDetector)
    except ImportError:
        logger.warning("Could not import mock evaluator: ToxicityDetector")

def ops(
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Decorator to create an Operation from a function.
    Performs type checking at import time.
    
    Example:
        @ops(name="tokenize", tags=["nlp"])
        def tokenize(text: str) -> List[str]:
            return text.split()
    """
    def decorator(func: Callable[[T], U]) -> Operation[T, U]:
        # Validate type hints at import time
        hints = get_type_hints(func)
        if not hints:
            raise TypeError(f"Function {func.__name__} must have type hints")
        
        # Check input parameter types
        params = list(hints.items())
        if len(params) < 2:  # Need at least one parameter and return type
            raise TypeError(f"Function {func.__name__} must have at least one parameter with type annotation")
            
        # Validate return type exists
        if "return" not in hints:
            raise TypeError(f"Function {func.__name__} must specify return type annotation")
            
        op_name = name or func.__name__
        config = OpConfig(
            name=op_name,
            description=description or func.__doc__,
            tags=tags or [],
            metadata=metadata or {},
            openlit_config=openlit_config,
        )
        
        operation = Operation(
            func=func,
            config=config,
        )
        
        # Store validated type information
        operation._input_type = params[0][1]  # First parameter type
        operation._output_type = hints["return"]
        
        return operation
    
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
        
        return decorator
    
    return decorator
