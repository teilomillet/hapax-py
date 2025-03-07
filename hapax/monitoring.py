"""
GPU and hardware monitoring functionality for Hapax using OpenLIT.
"""
from typing import Optional, Dict, Any, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

def enable_gpu_monitoring(
    otlp_endpoint: Optional[str] = None,
    sample_rate_seconds: int = 5,
    application_name: str = "hapax",
    environment: str = "development",
    custom_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Enable GPU monitoring through OpenLIT integration.
    
    Args:
        otlp_endpoint: OpenTelemetry endpoint for sending metrics
        sample_rate_seconds: How often to sample GPU metrics (in seconds)
        application_name: Name of the application for metrics
        environment: Environment name (dev, prod, etc.)
        custom_config: Additional configuration options for OpenLIT
    
    Returns:
        None
    """
    try:
        import openlit
    except ImportError:
        logger.warning(
            "OpenLIT not installed. Install with 'pip install openlit' to enable GPU monitoring."
        )
        return
    
    # Prepare configuration
    config = custom_config or {}
    config.update({
        "gpu_monitoring": True,
        "gpu_sample_rate": sample_rate_seconds
    })
    
    try:
        # Initialize OpenLIT with GPU monitoring
        openlit.init(
            otlp_endpoint=otlp_endpoint,
            application_name=application_name,
            environment=environment,
            **config
        )
        logger.info(f"GPU monitoring enabled with sample rate of {sample_rate_seconds}s")
    except Exception as e:
        logger.error(f"Failed to initialize GPU monitoring: {e}")

def get_gpu_metrics() -> Dict[str, Union[float, int, str]]:
    """
    Get current GPU metrics.
    
    Returns:
        Dictionary containing current GPU metrics from OpenLIT
    """
    try:
        import openlit
        return openlit.metrics.get_gpu_metrics()
    except ImportError:
        logger.warning("OpenLIT not installed. Cannot fetch GPU metrics.")
        return {}
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        return {} 