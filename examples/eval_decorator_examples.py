"""
Examples of using the evaluation decorator system.
"""

from hapax.core.decorators import eval, register_evaluator

# Example custom evaluator
class CustomEvaluator:
    """Example evaluator that checks text length."""
    
    def __init__(self, **config):
        self.max_length = config.get("max_length", 100)
    
    def evaluate(self, text: str) -> float:
        """Return a score based on text length (0.0 is best, 1.0 is worst)."""
        return min(1.0, len(text) / self.max_length)

# Register our custom evaluator
register_evaluator("length", CustomEvaluator)

# Basic usage with built-in evaluators
@eval(
    name="hallucination_check",
    evaluators=["hallucination"],
    threshold=0.7
)
def simple_generation(prompt: str) -> str:
    """Generate a simple response with hallucination check."""
    return f"The answer to {prompt} is always 42."

# Using multiple evaluations with caching
@eval(
    name="safety_check",
    evaluators=["bias", "toxicity"],
    threshold=0.5,
    cache_results=True
)
def comprehensive_check(prompt: str) -> str:
    """Generate text with multiple safety checks."""
    return f"Processing {prompt} with comprehensive checks."

# Using custom evaluator with configuration
@eval(
    name="length_control",
    evaluators=["length"],
    threshold=0.3,
    config={"max_length": 50}
)
def length_controlled(prompt: str) -> str:
    """Generate text with length control."""
    return f"Short response to: {prompt}"

# Using all available evaluators
@eval(
    name="maximum_safety_check",
    evaluators=["all"],
    threshold=0.6
)
def maximum_safety(prompt: str) -> str:
    """Run all registered evaluators (built-in and custom)."""
    return f"Super safe response to: {prompt}"

def main():
    """Run example evaluations."""
    try:
        # Basic usage - should pass
        result = simple_generation("what is the meaning of life?")
        print(f"Simple generation result: {result}")
        
        # Multiple evaluations with caching
        prompt = "Tell me about history"
        
        # First call - evaluates and caches
        result1 = comprehensive_check(prompt)
        print(f"First call result: {result1}")
        
        # Second call - uses cache
        result2 = comprehensive_check(prompt)
        print(f"Second call result (from cache): {result2}")
        
        # Custom evaluator
        result = length_controlled("make this short")
        print(f"Length controlled result: {result}")
        
        # All evaluators
        result = maximum_safety("test all checks")
        print(f"Maximum safety result: {result}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
