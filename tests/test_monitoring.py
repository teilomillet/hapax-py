"""Tests for GPU monitoring functionality."""
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hapax.monitoring import enable_gpu_monitoring, get_gpu_metrics


class TestGPUMonitoring(unittest.TestCase):
    """Test GPU monitoring functions."""

    def test_enable_gpu_monitoring_basic(self):
        """Test basic GPU monitoring initialization."""
        # Create a mock for OpenLIT
        mock_openlit = MagicMock()
        
        # Use patch to replace the import statement
        with patch('builtins.__import__', return_value=mock_openlit) as mock_import:
            # Configure mock_import to return our mock_openlit only for 'openlit'
            mock_import.side_effect = lambda name, *args, **kwargs: mock_openlit if name == 'openlit' else __import__(name, *args, **kwargs)
            
            enable_gpu_monitoring()
            
            # Verify openlit.init was called with correct parameters
            mock_openlit.init.assert_called_once()
            call_kwargs = mock_openlit.init.call_args.kwargs
            self.assertEqual(call_kwargs.get('application_name'), 'hapax')
            self.assertEqual(call_kwargs.get('environment'), 'development')
            self.assertTrue(call_kwargs.get('gpu_monitoring'))
            self.assertEqual(call_kwargs.get('gpu_sample_rate'), 5)

    def test_enable_gpu_monitoring_custom(self):
        """Test GPU monitoring with custom parameters."""
        # Create a mock for OpenLIT
        mock_openlit = MagicMock()
        
        # Use patch to replace the import statement
        with patch('builtins.__import__', return_value=mock_openlit) as mock_import:
            # Configure mock_import to return our mock_openlit only for 'openlit'
            mock_import.side_effect = lambda name, *args, **kwargs: mock_openlit if name == 'openlit' else __import__(name, *args, **kwargs)
            
            enable_gpu_monitoring(
                otlp_endpoint="http://custom:4318",
                sample_rate_seconds=2,
                application_name="test_app",
                environment="production",
                custom_config={"trace_memory": True}
            )
            
            # Verify openlit.init was called with correct parameters
            mock_openlit.init.assert_called_once()
            call_kwargs = mock_openlit.init.call_args.kwargs
            self.assertEqual(call_kwargs.get('otlp_endpoint'), 'http://custom:4318')
            self.assertEqual(call_kwargs.get('application_name'), 'test_app')
            self.assertEqual(call_kwargs.get('environment'), 'production')
            self.assertTrue(call_kwargs.get('gpu_monitoring'))
            self.assertEqual(call_kwargs.get('gpu_sample_rate'), 2)
            self.assertTrue(call_kwargs.get('trace_memory'))

    @patch('hapax.monitoring.logger')
    def test_enable_gpu_monitoring_import_error(self, mock_logger):
        """Test handling of import error when OpenLIT is not available."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openlit'")):
            enable_gpu_monitoring()
            
            # Should log warning
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args.args[0]
            self.assertIn("OpenLIT not installed", warning_message)

    @patch('hapax.monitoring.logger')
    def test_enable_gpu_monitoring_exception(self, mock_logger):
        """Test exception handling during initialization."""
        # Create a mock for OpenLIT that raises an exception on init
        mock_openlit = MagicMock()
        mock_openlit.init.side_effect = Exception("Test error")
        
        # Use patch to replace the import statement
        with patch('builtins.__import__', return_value=mock_openlit) as mock_import:
            # Configure mock_import to return our mock_openlit only for 'openlit'
            mock_import.side_effect = lambda name, *args, **kwargs: mock_openlit if name == 'openlit' else __import__(name, *args, **kwargs)
            
            enable_gpu_monitoring()
            
            # Should log error
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args.args[0]
            self.assertIn("Failed to initialize GPU monitoring", error_message)

    def test_get_gpu_metrics_success(self):
        """Test getting GPU metrics successfully."""
        # Mock metrics data
        mock_metrics = {
            "gpu_utilization": 45.2,
            "gpu_memory_used": 3.2,
            "gpu_temperature": 68
        }
        
        # Create a mock for OpenLIT
        mock_openlit = MagicMock()
        mock_openlit.metrics.get_gpu_metrics.return_value = mock_metrics
        
        # Use patch to replace the import statement
        with patch('builtins.__import__', return_value=mock_openlit) as mock_import:
            # Configure mock_import to return our mock_openlit only for 'openlit'
            mock_import.side_effect = lambda name, *args, **kwargs: mock_openlit if name == 'openlit' else __import__(name, *args, **kwargs)
            
            # Get the metrics
            result = get_gpu_metrics()
            
            # Verify the result
            self.assertEqual(result, mock_metrics)
            mock_openlit.metrics.get_gpu_metrics.assert_called_once()

    @patch('hapax.monitoring.logger')
    def test_get_gpu_metrics_import_error(self, mock_logger):
        """Test handling of import error when getting GPU metrics."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openlit'")):
            result = get_gpu_metrics()
            
            # Should log warning and return empty dict
            mock_logger.warning.assert_called_once()
            warning_message = mock_logger.warning.call_args.args[0]
            self.assertIn("OpenLIT not installed", warning_message)
            self.assertEqual(result, {})

    @patch('hapax.monitoring.logger')
    def test_get_gpu_metrics_exception(self, mock_logger):
        """Test exception handling when getting GPU metrics."""
        # Create a mock for OpenLIT that raises an exception
        mock_openlit = MagicMock()
        mock_openlit.metrics.get_gpu_metrics.side_effect = Exception("Test error")
        
        # Use patch to replace the import statement
        with patch('builtins.__import__', return_value=mock_openlit) as mock_import:
            # Configure mock_import to return our mock_openlit only for 'openlit'
            mock_import.side_effect = lambda name, *args, **kwargs: mock_openlit if name == 'openlit' else __import__(name, *args, **kwargs)
            
            result = get_gpu_metrics()
            
            # Should log error and return empty dict
            mock_logger.error.assert_called_once()
            error_message = mock_logger.error.call_args.args[0]
            self.assertIn("Failed to get GPU metrics", error_message)
            self.assertEqual(result, {})


class TestGraphIntegration(unittest.TestCase):
    """Test integration with Graph class."""
    
    @patch('hapax.monitoring.enable_gpu_monitoring')
    def test_graph_with_gpu_monitoring(self, mock_enable_monitoring):
        """Test Graph.with_gpu_monitoring method."""
        from hapax.core.graph import Graph
        
        # Create a graph with GPU monitoring
        graph = Graph("test_graph").with_gpu_monitoring(
            enabled=True, 
            sample_rate_seconds=3,
            custom_config={"trace_memory": True}
        )
        
        # Verify the configuration
        self.assertTrue(graph._gpu_monitoring)
        self.assertEqual(graph._gpu_config["gpu_sample_rate"], 3)
        self.assertTrue(graph._gpu_config["gpu_monitoring"])
        self.assertTrue(graph._gpu_config["trace_memory"])
        
        # Execute the graph to trigger monitoring
        with patch.object(graph, '_operations', []):
            try:
                graph.execute("test")
            except ValueError:
                # The graph has no operations, so it will raise a ValueError
                pass
        
        # Verify monitoring was enabled
        mock_enable_monitoring.assert_called_once()
        call_kwargs = mock_enable_monitoring.call_args.kwargs
        self.assertTrue(call_kwargs["custom_config"]["gpu_monitoring"])
        self.assertEqual(call_kwargs["custom_config"]["gpu_sample_rate"], 3)
        self.assertTrue(call_kwargs["custom_config"]["trace_memory"])


if __name__ == '__main__':
    unittest.main() 