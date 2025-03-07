"""Server implementation for the Hapax Documentation Assistant.

This module handles the lifecycle of the documentation index and
provides the main entry point for the MCP server.

The server provides documentation search and exploration functionality
using the Model Context Protocol (MCP).
"""

import asyncio
import os
import signal
import sys
import traceback
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

# Import FastMCP directly from the module
from mcp.server.fastmcp import FastMCP, Context

try:
    # For cleaner Windows process termination
    import win32api
    def handle_win32_event(sig):
        """Windows-specific signal handler"""
        global shutdown_requested
        print(f"Received signal {sig}, shutting down gracefully...")
        shutdown_requested = True
        return True
    
    # Register Windows signal handlers
    win32api.SetConsoleCtrlHandler(handle_win32_event, True)
except ImportError:
    # Not on Windows or win32api not available
    pass

from .indexer import DocIndex
from .resources import register_resources
from .tools import register_tools

# Global flag to handle graceful shutdown
shutdown_requested = False

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown"""
    def handle_signal(sig, frame):
        global shutdown_requested
        print(f"Received signal {sig}, shutting down gracefully...")
        shutdown_requested = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # On Windows, SIGBREAK can also be received
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, handle_signal)

@asynccontextmanager
async def docs_lifespan(server: FastMCP) -> AsyncIterator[Optional[DocIndex]]:
    """Manage the lifecycle of the documentation index.
    
    This context manager ensures that the documentation index is properly
    initialized when the server starts and cleaned up when it stops.
    
    Args:
        server: The MCP server instance
        
    Yields:
        The initialized documentation index
    """
    # Get configuration from environment variables
    docs_dir = os.environ.get("HAPAX_DOCS_DIR", "agent_docs")
    source_dir = os.environ.get("HAPAX_SOURCE_DIR", "hapax")
    
    # Print a message to indicate server startup
    print(f"Initializing Hapax Documentation Assistant...")
    print(f"Docs directory: {docs_dir}")
    print(f"Source directory: {source_dir}")
    sys.stdout.flush()  # Ensure output is immediately visible
    
    try:
        # Check if directories exist
        docs_path = Path(docs_dir)
        source_path = Path(source_dir)
        
        if not docs_path.exists():
            print(f"WARNING: Docs directory '{docs_dir}' does not exist.")
            print(f"Current working directory: {os.getcwd()}")
            # Create it if it doesn't exist
            os.makedirs(docs_dir, exist_ok=True)
            print(f"Created directory: {docs_dir}")
            
            # Add a sample document to make indexing work
            docs_path = Path(docs_dir)
            sample_file = docs_path / "README.md"
            if not sample_file.exists():
                with open(sample_file, "w") as f:
                    f.write("# Hapax Documentation\n\nThis is a sample documentation file.")
                print(f"Created sample documentation file: {sample_file}")
        
        if not source_path.exists():
            print(f"WARNING: Source directory '{source_dir}' does not exist.")
            print(f"Current working directory: {os.getcwd()}")
            # Create it if it doesn't exist
            os.makedirs(source_dir, exist_ok=True)
            print(f"Created directory: {source_dir}")
            
            # Add a sample source file to make indexing work
            source_path = Path(source_dir)
            sample_file = source_path / "sample.py"
            if not sample_file.exists():
                with open(sample_file, "w") as f:
                    f.write('"""Sample module for Hapax.\n\nThis is a placeholder file.\n"""\n\ndef sample_function():\n    """Sample function for demonstration purposes."""\n    return "Hello, World!"')
                print(f"Created sample source file: {sample_file}")
            
        # Initialize the documentation index
        print(f"Creating DocIndex with docs_dir={docs_dir}, source_dir={source_dir}")
        sys.stdout.flush()
        doc_index = DocIndex(docs_dir, source_dir)
        print(f"DocIndex created successfully")
        sys.stdout.flush()
        
        # Yield the index to make it available as the lifespan context
        yield doc_index
        
        print("Server shutdown requested, cleaning up...")
    except Exception as e:
        print(f"ERROR during initialization: {str(e)}")
        print(traceback.format_exc())
        sys.stdout.flush()
        # We still need to yield something even in case of error
        # Create a minimal DocIndex that won't crash when methods are called
        try:
            os.makedirs(docs_dir, exist_ok=True)
            os.makedirs(source_dir, exist_ok=True)
            minimal_index = DocIndex(docs_dir, source_dir)
            yield minimal_index
        except:
            # Last resort - yield None and let resources/tools handle it
            yield None
    finally:
        # Clean up when the server is shutting down
        print("Server shutdown complete")
        sys.stdout.flush()

def create_server() -> FastMCP:
    """Create and configure the MCP server.
    
    Returns:
        The configured MCP server
    """
    try:
        # Create the server with a name and lifespan
        print("Creating FastMCP server instance...")
        server = FastMCP(
            "Hapax Documentation Assistant",
            lifespan=docs_lifespan,
            dependencies=["mcp>=1.3.0", "pydantic>=2.0.0"]
        )
        
        # Register resources
        print("Registering resources...")
        register_resources(server)
        print("Resources registered successfully")
        sys.stdout.flush()
        
        # Register tools
        print("Registering tools...")
        register_tools(server)
        print("Tools registered successfully")
        
        print("Server configuration complete")
        
        # Add a simple test tool that doesn't depend on DocIndex
        @server.tool()
        def test_tool() -> str:
            """Test tool to verify server functionality"""
            return "The server is working correctly!"
            
        # Add another simple tool that Cursor can more easily detect
        @server.tool()
        def ping() -> str:
            """Simple ping tool to test server connectivity"""
            return "pong"
            
        # Add a hello world tool that's very simple
        @server.tool()
        def hello_world() -> str:
            """Simple hello world tool"""
            return "Hello, world!"
        
        print("Test tools registered successfully")
        sys.stdout.flush()
        
        return server
    except Exception as e:
        print(f"ERROR creating server: {str(e)}")
        print(traceback.format_exc())
        sys.stdout.flush()
        # Re-raise to prevent server from starting with incorrect configuration
        raise

# Set up signal handlers
setup_signal_handlers()

# Create the server at module load time
try:
    print("Initializing mcp_server...")
    sys.stdout.flush()
    mcp_server = create_server()
    print("mcp_server initialized successfully")
    sys.stdout.flush()
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize mcp_server: {str(e)}")
    print(traceback.format_exc())
    sys.stdout.flush()
    sys.exit(1)

def run_server():
    """Run the MCP server directly.
    
    This function is the main entry point for running the server in production.
    """
    print("Starting Hapax Documentation Assistant MCP server...")
    print("Press Ctrl+C to exit")
    
    # Add handling for environment variables
    docs_dir = os.environ.get("HAPAX_DOCS_DIR", "agent_docs")
    source_dir = os.environ.get("HAPAX_SOURCE_DIR", "hapax")
    
    print(f"Using docs directory: {docs_dir}")
    print(f"Using source directory: {source_dir}")
    sys.stdout.flush()
    
    try:
        print("Starting server...")
        sys.stdout.flush()  # Ensure output is flushed to prevent buffering issues
        
        # Run the server
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error running server: {e}")
        print(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    run_server() 