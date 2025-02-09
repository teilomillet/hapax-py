"""Mock evaluation classes for demonstration purposes."""

class BaseEvaluator:
    def __init__(self, **kwargs):
        self.config = kwargs
        
    def evaluate(self, text: str) -> float:
        """Return a mock score between 0 and 1."""
        return 0.5

class Hallucination(BaseEvaluator):
    """Mock hallucination detector."""
    pass

class BiasDetector(BaseEvaluator):
    """Mock bias detector."""
    pass

class ToxicityDetector(BaseEvaluator):
    """Mock toxicity detector."""
    pass

class All(BaseEvaluator):
    """Mock combined evaluator."""
    pass
