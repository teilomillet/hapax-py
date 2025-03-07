#!/usr/bin/env python3
"""
Debug utility for the Hapax Documentation Assistant MCP server.

This script tests various aspects of the MCP server to identify any issues.
"""

import os
import sys
import traceback
from pathlib import Path
import subprocess
import time

def check_environment():
    """Check the Python environment"""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    
    # Check if hapax_docs package is importable
    try:
        import hapax_docs
        print(f"hapax_docs package: Found at {hapax_docs.__file__}")
    except ImportError as e:
        print(f"ERROR: hapax_docs package not importable: {e}")
    
    # Check MCP installation
    try:
        import mcp
        print(f"MCP package: Found at {mcp.__file__}, version {mcp.__version__}")
    except ImportError as e:
        print(f"ERROR: MCP package not installed: {e}")
        return False
    except AttributeError:
        print(f"MCP package found at {mcp.__file__}, but __version__ attribute not available")
    
    # Check directory structure
    docs_dir = os.environ.get("HAPAX_DOCS_DIR", "agent_docs")
    source_dir = os.environ.get("HAPAX_SOURCE_DIR", "hapax")
    
    docs_path = Path(docs_dir)
    source_path = Path(source_dir)
    
    print(f"Docs directory: {docs_dir} (exists: {docs_path.exists()})")
    print(f"Source directory: {source_dir} (exists: {source_path.exists()})")
    
    return True

def check_mcp_server():
    """Check the MCP server"""
    print("\n=== MCP Server Check ===")
    
    # Try to import server module
    try:
        from hapax_docs.server import mcp_server
        print(f"Server instance created: {mcp_server}")
        
        # Check server properties
        print(f"Server name: {mcp_server.name}")
        
        # Check if tools were registered
        tool_count = 0
        try:
            # This may not work depending on the FastMCP implementation
            # but it's worth a try
            tool_count = len(mcp_server._tools)
            print(f"Tools registered: {tool_count}")
        except (AttributeError, TypeError) as e:
            print(f"Unable to count tools: {e}")
    
        # Check if resources were registered
        resource_count = 0
        try:
            # This may not work depending on the FastMCP implementation
            resource_count = len(mcp_server._resources)
            print(f"Resources registered: {resource_count}")
        except (AttributeError, TypeError) as e:
            print(f"Unable to count resources: {e}")
            
    except ImportError as e:
        print(f"ERROR: Cannot import server module: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return False
    
    return True

def try_run_server():
    """Try to run the server in a subprocess for a short time"""
    print("\n=== Server Startup Test ===")
    
    try:
        # Run the server in a subprocess
        print("Starting server subprocess...")
        process = subprocess.Popen(
            [sys.executable, "-m", "hapax_docs.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Let it run for 5 seconds
        print("Letting server run for 5 seconds...")
        time.sleep(5)
        
        # Get output
        process.terminate()
        stdout, stderr = process.communicate(timeout=2)
        
        print("\nServer stdout:")
        print(stdout)
        
        if stderr:
            print("\nServer stderr:")
            print(stderr)
            
        print("\nServer process terminated")
    except Exception as e:
        print(f"ERROR running server subprocess: {e}")
        print(traceback.format_exc())
        return False
    
    return True

def main():
    """Main debug function"""
    print("=== Hapax Documentation Assistant MCP Server Debug ===\n")
    
    env_ok = check_environment()
    if not env_ok:
        print("\nEnvironment check failed!")
    
    server_ok = check_mcp_server()
    if not server_ok:
        print("\nServer check failed!")
    
    run_ok = try_run_server()
    if not run_ok:
        print("\nServer startup test failed!")
    
    print("\n=== Debug Summary ===")
    print(f"Environment check: {'PASSED' if env_ok else 'FAILED'}")
    print(f"Server check: {'PASSED' if server_ok else 'FAILED'}")
    print(f"Server startup test: {'PASSED' if run_ok else 'FAILED'}")
    
    if env_ok and server_ok and run_ok:
        print("\nAll checks passed! If you're still having issues, the problem may be in the MCP client or in Cursor's configuration.")
        print("Suggestions:")
        print("1. Make sure the server is properly registered in Cursor")
        print("2. Check for network issues or firewall settings")
        print("3. Try restarting Cursor")
    else:
        print("\nSome checks failed. Please fix the issues above before trying again.")

if __name__ == "__main__":
    main() 