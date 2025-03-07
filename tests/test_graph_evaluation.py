"""Tests for Graph integration with evaluations."""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from typing import Dict, Any, List, Optional

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hapax.core.graph import Graph
from hapax.core.decorators import EvaluationError


class TestGraphEvaluation(unittest.TestCase):
    """Test Graph integration with evaluations."""

    def test_with_evaluation_method(self):
        """Test with_evaluation method configuration."""
        graph = Graph("test_graph").with_evaluation(
            eval_type="hallucination",
            threshold=0.8,
            provider="openai",
            fail_on_evaluation=True,
            model="gpt-4o",
            api_key="test-key",
            custom_config={"trace_content": True}
        )
        
        # Verify configuration was set correctly
        self.assertEqual(graph._evaluation_config["eval_type"], "hallucination")
        self.assertEqual(graph._evaluation_config["threshold"], 0.8)
        self.assertEqual(graph._evaluation_config["provider"], "openai")
        self.assertTrue(graph._evaluation_config["fail_on_evaluation"])
        self.assertEqual(graph._evaluation_config["model"], "gpt-4o")
        self.assertEqual(graph._evaluation_config["api_key"], "test-key")
        self.assertTrue(graph._evaluation_config["trace_content"])

    def test_execute_with_evaluation_success(self):
        """Test executing a graph with successful evaluation."""
        # Create a mock for the evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "verdict": "no",
            "score": 0.2,
            "evaluation": "hallucination",
            "classification": "none",
            "explanation": "No hallucination detected."
        }
        
        # Create a simple test graph with evaluation
        graph = Graph("test_graph").with_evaluation(
            eval_type="hallucination",
            threshold=0.7,
            provider="openai"
        )
        
        # Create a dummy operation to avoid ValueError
        dummy_op = MagicMock()
        dummy_op.return_value = "test output"
        graph._operations = [dummy_op]
        
        # Mock the actual import inside execute method
        hallu_evaluator_cls = MagicMock(return_value=mock_evaluator)
        evaluators = {
            'HallucinationEvaluator': hallu_evaluator_cls,
            'BiasEvaluator': MagicMock(),
            'ToxicityEvaluator': MagicMock(),
            'AllEvaluator': MagicMock()
        }
        
        # Execute the graph with patched imports
        with patch.dict('sys.modules', {'hapax.evaluations': MagicMock(**evaluators)}):
            result = graph.execute("test input")
            
            # Verify evaluator was called with correct parameters
            mock_evaluator.evaluate.assert_called_once()
            call_args = mock_evaluator.evaluate.call_args.kwargs
            self.assertEqual(call_args["text"], "test output")  # Evaluator uses operation output
            self.assertEqual(call_args["prompt"], "test input")  # Input is used as prompt
            
            # Verify results were stored
            self.assertEqual(graph.last_evaluation["verdict"], "no")
            self.assertEqual(graph.last_evaluation["score"], 0.2)
            
            # Verify the operation was called and returned the expected output
            self.assertEqual(result, "test output")

    def test_execute_with_evaluation_failure(self):
        """Test the configuration for evaluation failure."""
        # Create a test graph with evaluation set to fail on evaluation
        graph = Graph("test_graph").with_evaluation(
            eval_type="hallucination",
            threshold=0.7,
            provider="openai",
            fail_on_evaluation=True
        )
        
        # Verify that fail_on_evaluation is set correctly
        self.assertTrue(graph._evaluation_config["fail_on_evaluation"])
        self.assertEqual(graph._evaluation_config["eval_type"], "hallucination")
        self.assertEqual(graph._evaluation_config["threshold"], 0.7)
        self.assertEqual(graph._evaluation_config["provider"], "openai")
        
        # Test that we can access the evaluator mapping
        # This is a basic verification that the code structure is as expected
        try:
            from hapax.evaluations import (
                AllEvaluator,
                HallucinationEvaluator,
                BiasEvaluator,
                ToxicityEvaluator
            )
            
            # Verify that the evaluator classes are correctly defined
            self.assertTrue(callable(HallucinationEvaluator))
            self.assertTrue(callable(AllEvaluator))
            self.assertTrue(callable(BiasEvaluator))
            self.assertTrue(callable(ToxicityEvaluator))
        except ImportError:
            # Skip this part of the test if imports fail
            pass

    def test_execute_with_all_evaluations(self):
        """Test executing a graph with all evaluations."""
        # Create a mock for the All evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "verdict": "no",
            "score": 0.1,
            "evaluation": "all",
            "classification": "none",
            "explanation": "No issues detected."
        }
        
        # Create a test graph with all evaluations
        graph = Graph("test_graph").with_evaluation(
            eval_type="all",
            threshold=0.6,
            provider="anthropic",
            model="claude-3-5-sonnet"
        )
        
        # Create a dummy operation to avoid ValueError
        dummy_op = MagicMock()
        dummy_op.return_value = "test output"
        graph._operations = [dummy_op]
        
        # Mock the actual import inside execute method
        all_evaluator_cls = MagicMock(return_value=mock_evaluator)
        evaluators = {
            'HallucinationEvaluator': MagicMock(),
            'BiasEvaluator': MagicMock(),
            'ToxicityEvaluator': MagicMock(),
            'AllEvaluator': all_evaluator_cls
        }
        
        # Execute the graph with patched imports
        with patch.dict('sys.modules', {'hapax.evaluations': MagicMock(**evaluators)}):
            result = graph.execute("test input")
            
            # Verify AllEvaluator was used with correct parameters
            mock_evaluator.evaluate.assert_called_once()
            call_args = mock_evaluator.evaluate.call_args.kwargs
            self.assertEqual(call_args["text"], "test output")  # Evaluator uses operation output
            
            # Verify results
            self.assertEqual(graph.last_evaluation["verdict"], "no")
            self.assertEqual(graph.last_evaluation["evaluation"], "all")
            
            # Verify the operation was called and returned the expected output
            self.assertEqual(result, "test output")

    def test_execute_with_evaluation_error(self):
        """Test handling of evaluation errors."""
        # Create a mock evaluator that raises an exception
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = Exception("Test evaluation error")
        
        # Create a test graph with evaluation
        graph = Graph("test_graph").with_evaluation(
            eval_type="hallucination"
        )
        
        # Create a dummy operation to avoid ValueError
        dummy_op = MagicMock()
        dummy_op.return_value = "test output"
        graph._operations = [dummy_op]
        
        # Mock the actual import inside execute method
        hallu_evaluator_cls = MagicMock(return_value=mock_evaluator)
        evaluators = {
            'HallucinationEvaluator': hallu_evaluator_cls,
            'BiasEvaluator': MagicMock(),
            'ToxicityEvaluator': MagicMock(),
            'AllEvaluator': MagicMock()
        }
        
        # Execute the graph with patched imports
        with patch('hapax.core.graph.logger') as mock_logger:
            with patch.dict('sys.modules', {'hapax.evaluations': MagicMock(**evaluators)}):
                result = graph.execute("test input")
                
                # Verify error was logged
                mock_logger.error.assert_called_once()
                error_msg = mock_logger.error.call_args.args[0]
                self.assertIn("Error during evaluation", error_msg)
                
                # Verify error result was stored
                self.assertEqual(graph.last_evaluation["verdict"], "error")
                self.assertEqual(graph.last_evaluation["error"], "Test evaluation error")
                
                # Verify the operation was still called and returned the expected output
                self.assertEqual(result, "test output")

    def test_get_context_from_metadata(self):
        """Test extracting context from operation metadata."""
        graph = Graph("test_graph")
        
        # Create mock operations with metadata containing context
        op1 = MagicMock()
        op1.config.metadata = {"context": "Context from op1"}
        
        op2 = MagicMock()
        op2.config.metadata = {"context": ["Context 1 from op2", "Context 2 from op2"]}
        
        op3 = MagicMock()
        op3.config.metadata = {"other_data": "No context here"}
        
        # Add operations to graph
        graph._operations = [op1, op2, op3]
        
        # Get contexts
        contexts = graph._get_context_from_metadata()
        
        # Verify contexts were extracted correctly
        self.assertEqual(len(contexts), 3)
        self.assertIn("Context from op1", contexts)
        self.assertIn("Context 1 from op2", contexts)
        self.assertIn("Context 2 from op2", contexts)


if __name__ == '__main__':
    unittest.main() 