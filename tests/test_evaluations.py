"""Tests for the evaluation module with OpenLIT integration."""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hapax.evaluations import (
    OpenLITEvaluator,
    HallucinationEvaluator,
    BiasEvaluator,
    ToxicityEvaluator,
    AllEvaluator
)
from hapax.core.decorators import eval, EvaluationError


class TestOpenLITEvaluator(unittest.TestCase):
    """Test the base OpenLITEvaluator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        evaluator = OpenLITEvaluator()
        self.assertEqual(evaluator.provider, "openai")
        self.assertEqual(evaluator.threshold, 0.5)
        self.assertTrue(evaluator.collect_metrics)
        self.assertIsNone(evaluator.model)
        self.assertIsNone(evaluator.api_key)
        self.assertIsNone(evaluator.base_url)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        evaluator = OpenLITEvaluator(
            provider="anthropic",
            threshold=0.7,
            collect_metrics=False,
            model="claude-3-5-sonnet",
            api_key="test-key",
            base_url="https://custom-api.com"
        )
        self.assertEqual(evaluator.provider, "anthropic")
        self.assertEqual(evaluator.threshold, 0.7)
        self.assertFalse(evaluator.collect_metrics)
        self.assertEqual(evaluator.model, "claude-3-5-sonnet")
        self.assertEqual(evaluator.api_key, "test-key")
        self.assertEqual(evaluator.base_url, "https://custom-api.com")

    def test_ensure_openlit_missing(self):
        """Test _ensure_openlit when OpenLIT is not available."""
        evaluator = OpenLITEvaluator()
        
        # Simulate import error
        with patch.object(evaluator, '_openlit', None):
            with patch('builtins.__import__', side_effect=ImportError):
                result = evaluator._ensure_openlit()
                self.assertFalse(result)

    def test_ensure_openlit_available(self):
        """Test _ensure_openlit when OpenLIT is available."""
        evaluator = OpenLITEvaluator()
        evaluator._openlit = None
        
        # Create a mock OpenLIT module
        mock_openlit = MagicMock()
        mock_openlit.evals = MagicMock()
        
        # Should import and set _openlit
        with patch('builtins.__import__', return_value=mock_openlit):
            result = evaluator._ensure_openlit()
            self.assertTrue(result)
            self.assertIsNotNone(evaluator._openlit)

    def test_evaluate_not_implemented(self):
        """Test that evaluate raises NotImplementedError in base class."""
        evaluator = OpenLITEvaluator()
        with self.assertRaises(NotImplementedError):
            evaluator.evaluate("test text")


class TestHallucinationEvaluator(unittest.TestCase):
    """Test the HallucinationEvaluator class."""

    def test_evaluate_openlit_not_available(self):
        """Test evaluate when OpenLIT is not available."""
        evaluator = HallucinationEvaluator()
        evaluator._openlit = None
        
        with patch.object(evaluator, '_ensure_openlit', return_value=False):
            result = evaluator.evaluate("test text")
            self.assertEqual(result["verdict"], "unknown")
            self.assertEqual(result["error"], "OpenLIT not available")

    def test_evaluate_without_contexts(self):
        """Test evaluate warns when no contexts are provided."""
        evaluator = HallucinationEvaluator()
        
        with patch('hapax.evaluations.logger') as mock_logger:
            with patch.object(evaluator, '_ensure_openlit', return_value=True):
                mock_detector = MagicMock()
                mock_detector.measure.return_value = {"verdict": "no", "score": 0.1}
                mock_openlit = MagicMock()
                mock_openlit.evals.HallucinationDetector.return_value = mock_detector
                evaluator._openlit = mock_openlit
                
                result = evaluator.evaluate("test text")
                
                # Should log warning about missing contexts
                mock_logger.warning.assert_called_once()
                self.assertEqual(result["verdict"], "no")

    def test_evaluate_with_contexts(self):
        """Test evaluate with contexts."""
        evaluator = HallucinationEvaluator()
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {
                "verdict": "yes",
                "score": 0.8,
                "evaluation": "hallucination",
                "classification": "factual_inaccuracy",
                "explanation": "Test explanation"
            }
            mock_openlit = MagicMock()
            mock_openlit.evals.HallucinationDetector.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            contexts = ["Einstein won the Nobel Prize in 1921."]
            result = evaluator.evaluate(
                "Einstein won the Nobel Prize in 1922.",
                contexts=contexts,
                prompt="When did Einstein win the Nobel Prize?"
            )
            
            # Should call measure with correct parameters
            mock_detector.measure.assert_called_once_with(
                text="Einstein won the Nobel Prize in 1922.",
                contexts=contexts,
                prompt="When did Einstein win the Nobel Prize?"
            )
            self.assertEqual(result["verdict"], "yes")
            self.assertEqual(result["score"], 0.8)
            self.assertEqual(result["classification"], "factual_inaccuracy")

    def test_model_parameter_passed_correctly(self):
        """Test that model parameter is passed correctly to OpenLIT."""
        evaluator = HallucinationEvaluator(
            model="gpt-4o",
            provider="openai"
        )
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {"verdict": "no"}
            mock_openlit = MagicMock()
            mock_openlit.evals.HallucinationDetector.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            evaluator.evaluate("test text")
            
            # Should create detector with correct model parameter
            mock_openlit.evals.HallucinationDetector.assert_called_once_with(
                provider="openai",
                threshold_score=0.5,
                collect_metrics=True,
                model="gpt-4o",
                api_key=None,
                base_url=None
            )

    def test_evaluate_exception_handling(self):
        """Test exception handling in evaluate."""
        evaluator = HallucinationEvaluator()
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.side_effect = Exception("Test error")
            mock_openlit = MagicMock()
            mock_openlit.evals.HallucinationDetector.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            with patch('hapax.evaluations.logger') as mock_logger:
                result = evaluator.evaluate("test text")
                
                # Should log error
                mock_logger.error.assert_called_once()
                self.assertEqual(result["verdict"], "error")
                self.assertEqual(result["error"], "Test error")
                self.assertEqual(result["evaluation"], "hallucination")


class TestAllEvaluator(unittest.TestCase):
    """Test the AllEvaluator class."""

    def test_evaluate_with_all_types(self):
        """Test evaluate with all evaluation types."""
        evaluator = AllEvaluator()
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {
                "verdict": "yes",
                "score": 0.9,
                "evaluation": "bias",
                "classification": "gender_bias",
                "explanation": "Test explanation"
            }
            mock_openlit = MagicMock()
            mock_openlit.evals.All.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            result = evaluator.evaluate(
                "Men are better at math than women.",
                contexts=["Research shows no inherent gender difference in math ability."],
                prompt="Discuss gender differences in math ability."
            )
            
            self.assertEqual(result["verdict"], "yes")
            self.assertEqual(result["evaluation"], "bias")
            self.assertEqual(result["classification"], "gender_bias")

    def test_model_parameter_passed_correctly(self):
        """Test that model parameter is passed correctly for All evaluator."""
        evaluator = AllEvaluator(
            model="claude-3-5-sonnet",
            provider="anthropic"
        )
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {"verdict": "no"}
            mock_openlit = MagicMock()
            mock_openlit.evals.All.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            evaluator.evaluate("test text")
            
            # Should create detector with correct model parameter
            mock_openlit.evals.All.assert_called_once_with(
                provider="anthropic",
                threshold_score=0.5,
                collect_metrics=True,
                model="claude-3-5-sonnet",
                api_key=None,
                base_url=None
            )


class TestOtherEvaluators(unittest.TestCase):
    """Test BiasEvaluator and ToxicityEvaluator."""

    def test_bias_evaluator(self):
        """Test BiasEvaluator functionality."""
        evaluator = BiasEvaluator(model="gpt-4o")
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {"verdict": "yes", "evaluation": "bias"}
            mock_openlit = MagicMock()
            mock_openlit.evals.BiasDetector.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            result = evaluator.evaluate("test text")
            
            # Should create detector with correct model parameter
            mock_openlit.evals.BiasDetector.assert_called_once_with(
                provider="openai",
                threshold_score=0.5,
                collect_metrics=True,
                model="gpt-4o",
                api_key=None,
                base_url=None
            )
            self.assertEqual(result["verdict"], "yes")
            self.assertEqual(result["evaluation"], "bias")

    def test_toxicity_evaluator(self):
        """Test ToxicityEvaluator functionality."""
        evaluator = ToxicityEvaluator(model="custom-model")
        
        with patch.object(evaluator, '_ensure_openlit', return_value=True):
            mock_detector = MagicMock()
            mock_detector.measure.return_value = {"verdict": "yes", "evaluation": "toxicity"}
            mock_openlit = MagicMock()
            mock_openlit.evals.ToxicityDetector.return_value = mock_detector
            evaluator._openlit = mock_openlit
            
            result = evaluator.evaluate("test text")
            
            # Should create detector with correct model parameter
            mock_openlit.evals.ToxicityDetector.assert_called_once_with(
                provider="openai",
                threshold_score=0.5,
                collect_metrics=True,
                model="custom-model",
                api_key=None,
                base_url=None
            )
            self.assertEqual(result["verdict"], "yes")
            self.assertEqual(result["evaluation"], "toxicity")


@patch('hapax.evaluations.HallucinationEvaluator')
class TestEvalDecorator(unittest.TestCase):
    """Test the @eval decorator with OpenLIT integration."""

    def test_eval_with_openlit(self, mock_evaluator_cls):
        """Test @eval decorator with OpenLIT evaluations."""
        # Mock the evaluate method
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {"verdict": "no", "score": 0.2}
        mock_evaluator_cls.return_value = mock_evaluator
        
        # Define a function with the @eval decorator
        @eval(
            evals=["hallucination"],
            threshold=0.7,
            use_openlit=True,
            openlit_provider="openai",
            metadata={
                "contexts": ["Einstein won the Nobel Prize in Physics in 1921."],
                "prompt": "When did Einstein win the Nobel Prize?"
            }
        )
        def generate_fact(prompt: str) -> str:
            return "Einstein won the Nobel Prize in 1921."
        
        # Call the function
        result = generate_fact("When did Einstein win the Nobel Prize?")
        
        # Verify the decorator behavior
        mock_evaluator_cls.assert_called_once_with(
            provider="openai",
            threshold=0.7,
            collect_metrics=True
        )
        mock_evaluator.evaluate.assert_called_once_with(
            text="Einstein won the Nobel Prize in 1921.",
            contexts=["Einstein won the Nobel Prize in Physics in 1921."],
            prompt="When did Einstein win the Nobel Prize?"
        )
        self.assertEqual(result, "Einstein won the Nobel Prize in 1921.")

    def test_eval_with_openlit_failure(self, mock_evaluator_cls):
        """Test @eval decorator with OpenLIT evaluations that fail."""
        # Mock the evaluate method to return failure
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "verdict": "yes",
            "score": 0.8,
            "evaluation": "hallucination",
            "classification": "factual_inaccuracy"
        }
        mock_evaluator_cls.return_value = mock_evaluator
        
        # Define a function with the @eval decorator
        @eval(
            evals=["hallucination"],
            threshold=0.7,
            use_openlit=True,
            openlit_provider="openai"
        )
        def generate_wrong_fact(prompt: str) -> str:
            return "Einstein won the Nobel Prize in 1922."  # Wrong date
        
        # Call the function and expect EvaluationError
        with self.assertRaises(EvaluationError):
            generate_wrong_fact("When did Einstein win the Nobel Prize?")


if __name__ == '__main__':
    unittest.main() 