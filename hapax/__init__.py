"""Hapax - A simple framework for building and monitoring data pipelines."""
from .core import (
    ops,
    Operation,
    OpConfig,
    BaseConfig,
    set_openlit_config,
    get_openlit_config,
    eval,
)
from .core.graph import Graph
from .monitoring import enable_gpu_monitoring, get_gpu_metrics
from .evaluations import (
    OpenLITEvaluator,
    HallucinationEvaluator,
    BiasEvaluator,
    ToxicityEvaluator,
    AllEvaluator,
)

__version__ = "0.1.4"

__all__ = [
    # Core functionality
    'ops',
    'Operation',
    'Graph',
    'OpConfig',
    'BaseConfig',
    'set_openlit_config',
    'get_openlit_config',
    'eval',
    
    # GPU Monitoring
    'enable_gpu_monitoring',
    'get_gpu_metrics',
    
    # Evaluations
    'OpenLITEvaluator',
    'HallucinationEvaluator',
    'BiasEvaluator',
    'ToxicityEvaluator',
    'AllEvaluator',
]
