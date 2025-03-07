"""
Hapax Documentation Assistant.

This package provides a documentation server for the Hapax framework
using the Model Context Protocol (MCP).
"""

from .version import __version__

from .server import mcp_server, run_server

__all__ = ["mcp_server", "run_server", "__version__"] 