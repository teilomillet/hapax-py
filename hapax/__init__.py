"""Hapax - A simple framework for building and monitoring data pipelines."""
from .core import (
    ops,
    Operation,
    OpConfig,
    set_openlit_config,
    get_openlit_config,
)
from .core.graph import Graph

__version__ = "0.1.2"

__all__ = [
    'ops',
    'Operation',
    'Graph',
    'OpConfig',
    'set_openlit_config',
    'get_openlit_config'
]
