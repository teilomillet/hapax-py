"""
Examples demonstrating the usage of the @eval decorator for evaluations.
"""
from hapax.core.decorators import ops
from examples.mock_evals import Hallucination, BiasDetector, ToxicityDetector, All
from functools import wraps
from typing import Optional, List, Dict, Any, Callable

def eval(
    evals: Optional[List[str]] = None,
    threshold: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
    openlit_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Example decorator to evaluate function outputs using evaluations.
    
    Args:
        evals: List of evaluations to run. Options: ["hallucination", "bias", "toxicity", "all"]
        threshold: Threshold score for evaluations (0.0 to 1.0)
        metadata: Additional metadata for evaluations
        openlit_config: Configuration for evaluations
    """
    def decorator(func: Callable) -> Callable:
        config = {
            "evals": evals or ["all"],
            "threshold": threshold,
            "metadata": metadata or {},
            "openlit_config": openlit_config or {}
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Run evaluations
            eval_results = {}
            for eval_type in config["evals"]:
                if eval_type == "hallucination" or eval_type == "all":
                    hall = Hallucination(**config["openlit_config"])
                    eval_results["hallucination"] = hall.evaluate(result)
                
                if eval_type == "bias" or eval_type == "all":
                    bias = BiasDetector(**config["openlit_config"])
                    eval_results["bias"] = bias.evaluate(result)
                    
                if eval_type == "toxicity" or eval_type == "all":
                    tox = ToxicityDetector(**config["openlit_config"])
                    eval_results["toxicity"] = tox.evaluate(result)
            
            # Check if any evaluation failed
            failed_evals = [
                name for name, score in eval_results.items()
                if score > config["threshold"]
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

# Basic example with single evaluation
@eval(evals=["hallucination"], threshold=0.7)
def simple_generation(prompt: str) -> str:
    """Generate a simple response and check for hallucination."""
    return f"The answer to {prompt} is always 42."

# Multiple evaluations with custom threshold
@eval(evals=["hallucination", "bias", "toxicity"], threshold=0.5)
def comprehensive_check(user_input: str) -> str:
    """Check generated content for multiple criteria."""
    return f"Processing {user_input} with comprehensive checks."

# Combining with @ops decorator
@ops(name="safe_generate", tags=["nlp", "safe"])
@eval(evals=["all"], threshold=0.6)
def safe_generation(context: str) -> str:
    """Generate content with all safety checks enabled."""
    return f"Safely generated response for: {context}"

# Custom configuration example
@eval(
    evals=["bias"],
    threshold=0.3,
    metadata={"domain": "financial"},
    openlit_config={"model": "financial-bias-v1"}
)
def financial_advice(query: str) -> str:
    """Generate financial advice with strict bias checking."""
    return f"Financial advice for {query}"

def main():
    """Run examples and demonstrate error handling."""
    try:
        # Basic usage
        result1 = simple_generation("what is the meaning of life?")
        print(f"Simple generation result: {result1}")

        # Multiple evaluations
        result2 = comprehensive_check("Tell me about history")
        print(f"Comprehensive check result: {result2}")

        # Combined with @ops
        result3 = safe_generation("Generate a story")
        print(f"Safe generation result: {result3}")

        # With custom config
        result4 = financial_advice("Should I invest in stocks?")
        print(f"Financial advice result: {result4}")

    except EvaluationError as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
