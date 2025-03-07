"""
Evaluation capabilities for Hapax using OpenLIT.

This module provides OpenLIT-powered evaluations for hallucination, bias, 
toxicity, and other potential issues in LLM-generated text.
"""
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define supported models by provider for better documentation and validation
SUPPORTED_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"]
}

class OpenLITEvaluator:
    """Base class for OpenLIT-based evaluations in Hapax."""
    
    def __init__(
        self, 
        provider: str = "openai",
        threshold: float = 0.5,
        collect_metrics: bool = True,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenLIT evaluator.
        
        Args:
            provider: LLM provider to use ("openai" or "anthropic")
            threshold: Score threshold for evaluation (0.0 to 1.0)
            collect_metrics: Whether to collect OpenTelemetry metrics
            model: Specific model to use (see SUPPORTED_MODELS)
            api_key: API key for the provider (optional)
            base_url: Base URL for the provider's API (optional)
            
        Supported models:
            - OpenAI: "gpt-4o", "gpt-4o-mini"
            - Anthropic: "claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"
        """
        if provider not in ["openai", "anthropic"]:
            logger.warning(f"Unsupported provider: {provider}. Defaulting to 'openai'.")
            provider = "openai"
            
        self.provider = provider
        self.threshold = threshold
        self.collect_metrics = collect_metrics
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        
        # Validate model selection if specified
        if model and provider in SUPPORTED_MODELS:
            if model not in SUPPORTED_MODELS[provider]:
                logger.warning(
                    f"Model '{model}' may not be supported by provider '{provider}'. "
                    f"Supported models: {SUPPORTED_MODELS[provider]}"
                )
        
        # Try to import OpenLIT to provide early feedback if not installed
        try:
            import openlit
            self._openlit = openlit
        except ImportError:
            logger.warning(
                "OpenLIT not installed. Install with 'pip install openlit' to use evaluations."
            )
            self._openlit = None
    
    def evaluate(self, text: str, contexts: Optional[List[str]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text using OpenLIT.
        
        Args:
            text: The text to evaluate
            contexts: List of context strings for evaluation (optional)
            prompt: The prompt that generated the text (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        raise NotImplementedError("Implement in subclass")
    
    def _ensure_openlit(self) -> bool:
        """
        Ensure OpenLIT is available.
        
        Returns:
            True if OpenLIT is available, False otherwise
        """
        if self._openlit is None:
            try:
                import openlit
                self._openlit = openlit
                return True
            except ImportError:
                logger.error("OpenLIT not installed. Cannot perform evaluation.")
                return False
        return True


class HallucinationEvaluator(OpenLITEvaluator):
    """Evaluates text for hallucinations using OpenLIT."""
    
    def evaluate(self, text: str, contexts: Optional[List[str]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text for hallucinations.
        
        Args:
            text: The text to evaluate
            contexts: List of context strings to check against (required for hallucination detection)
            prompt: The prompt that generated the text (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self._ensure_openlit():
            return {"error": "OpenLIT not available", "verdict": "unknown"}
        
        if not contexts:
            logger.warning("No contexts provided for hallucination detection. Results may be unreliable.")
            contexts = []
            
        try:
            detector = self._openlit.evals.HallucinationDetector(
                provider=self.provider,
                threshold_score=self.threshold,
                collect_metrics=self.collect_metrics,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            result = detector.measure(
                prompt=prompt or "",
                contexts=contexts,
                text=text
            )
            
            return result
        except Exception as e:
            logger.error(f"Error during hallucination evaluation: {e}")
            return {
                "error": str(e),
                "verdict": "error",
                "score": 0.0,
                "evaluation": "hallucination",
                "classification": "error",
                "explanation": f"Evaluation failed: {e}"
            }


class BiasEvaluator(OpenLITEvaluator):
    """Evaluates text for bias using OpenLIT."""
    
    def evaluate(self, text: str, contexts: Optional[List[str]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text for bias.
        
        Args:
            text: The text to evaluate
            contexts: List of context strings (optional for bias detection)
            prompt: The prompt that generated the text (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self._ensure_openlit():
            return {"error": "OpenLIT not available", "verdict": "unknown"}
            
        contexts = contexts or []
            
        try:
            detector = self._openlit.evals.BiasDetector(
                provider=self.provider,
                threshold_score=self.threshold,
                collect_metrics=self.collect_metrics,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            result = detector.measure(
                prompt=prompt or "",
                contexts=contexts,
                text=text
            )
            
            return result
        except Exception as e:
            logger.error(f"Error during bias evaluation: {e}")
            return {
                "error": str(e),
                "verdict": "error",
                "score": 0.0,
                "evaluation": "bias",
                "classification": "error",
                "explanation": f"Evaluation failed: {e}"
            }


class ToxicityEvaluator(OpenLITEvaluator):
    """Evaluates text for toxicity using OpenLIT."""
    
    def evaluate(self, text: str, contexts: Optional[List[str]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text for toxicity.
        
        Args:
            text: The text to evaluate
            contexts: List of context strings (optional for toxicity detection)
            prompt: The prompt that generated the text (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self._ensure_openlit():
            return {"error": "OpenLIT not available", "verdict": "unknown"}
            
        contexts = contexts or []
            
        try:
            detector = self._openlit.evals.ToxicityDetector(
                provider=self.provider,
                threshold_score=self.threshold,
                collect_metrics=self.collect_metrics,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            result = detector.measure(
                prompt=prompt or "",
                contexts=contexts,
                text=text
            )
            
            return result
        except Exception as e:
            logger.error(f"Error during toxicity evaluation: {e}")
            return {
                "error": str(e),
                "verdict": "error",
                "score": 0.0,
                "evaluation": "toxicity",
                "classification": "error",
                "explanation": f"Evaluation failed: {e}"
            }


class AllEvaluator(OpenLITEvaluator):
    """Evaluates text for hallucination, bias, and toxicity using OpenLIT."""
    
    def evaluate(self, text: str, contexts: Optional[List[str]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate text for all issues (hallucination, bias, toxicity).
        
        Args:
            text: The text to evaluate
            contexts: List of context strings (helpful for hallucination detection)
            prompt: The prompt that generated the text (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self._ensure_openlit():
            return {"error": "OpenLIT not available", "verdict": "unknown"}
            
        contexts = contexts or []
            
        try:
            detector = self._openlit.evals.All(
                provider=self.provider,
                threshold_score=self.threshold,
                collect_metrics=self.collect_metrics,
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            result = detector.measure(
                prompt=prompt or "",
                contexts=contexts,
                text=text
            )
            
            return result
        except Exception as e:
            logger.error(f"Error during comprehensive evaluation: {e}")
            return {
                "error": str(e),
                "verdict": "error",
                "score": 0.0,
                "evaluation": "all",
                "classification": "error",
                "explanation": f"Evaluation failed: {e}"
            } 