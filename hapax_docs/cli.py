#!/usr/bin/env python3
"""
Command-line interface for the Hapax Documentation Assistant MCP server.

This script provides a convenient way to run the server in different modes,
including integration with Cursor via the Model Context Protocol (MCP).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Hapax Documentation MCP Server"
    )
    
    # Add command argument
    parser.add_argument(
        "command",
        choices=["run", "dev", "install", "install-cursor", "cursor-setup", "debug"],
        help="Command to execute (run, dev, install, install-cursor, cursor-setup, or debug)"
    )
    
    # Add optional arguments
    parser.add_argument(
        "--name",
        default="Hapax Documentation Assistant",
        help="Name for the server when installing in Claude Desktop or Cursor"
    )
    
    parser.add_argument(
        "--docs-dir",
        default="agent_docs",
        help="Directory containing documentation files"
    )
    
    parser.add_argument(
        "--source-dir",
        default="hapax",
        help="Directory containing source code files"
    )
    
    # Add environment variables for server configuration
    parser.add_argument(
        "-v", "--env-var",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set environment variables for the server (can be used multiple times)"
    )
    
    parser.add_argument(
        "-f", "--env-file",
        help="Load environment variables from a file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get the absolute path to the server script
    script_dir = Path(__file__).parent
    server_script = script_dir / "server.py"
    
    # Set environment variables
    if args.docs_dir:
        os.environ["HAPAX_DOCS_DIR"] = args.docs_dir
    
    if args.source_dir:
        os.environ["HAPAX_SOURCE_DIR"] = args.source_dir
    
    # Set additional environment variables if provided
    if args.env_var:
        for key, value in args.env_var:
            os.environ[key] = value
    
    # Load environment variables from file if provided
    if args.env_file:
        try:
            with open(args.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error loading environment file: {e}")
            sys.exit(1)
    
    # Execute the command
    if args.command == "run":
        # Run the server directly
        from .server import run_server
        run_server()
        
    elif args.command == "dev":
        # Run in development mode with MCP Inspector
        os.system(f"python -m mcp dev {server_script}")
        
    elif args.command == "install":
        # Install in Claude Desktop
        os.system(f"python -m mcp install {server_script} --name \"{args.name}\"")
        
    elif args.command == "install-cursor":
        # Install in Cursor
        cursor_config = create_cursor_config(args.name, server_script)
        print(f"To add this server to Cursor, follow these steps:")
        print(f"1. Open Cursor and go to Settings > Features > MCP")
        print(f"2. Click '+ Add New MCP Server'")
        print(f"3. Set 'Type' to 'stdio'")
        print(f"4. Set 'Name' to '{args.name}'")
        print(f"5. Set 'Command' to: python -m hapax_docs.server")
        print(f"6. Click 'Save'")
        
    elif args.command == "cursor-setup":
        # Run the Cursor setup script
        from .cursor_setup import main as setup_main
        setup_main()
        
    elif args.command == "debug":
        # Run the debug script
        from .debug import main as debug_main
        debug_main()
        
def create_cursor_config(name: str, server_script: Path) -> Dict[str, Any]:
    """Create configuration for Cursor MCP integration.
    
    Args:
        name: The name to display in Cursor
        server_script: The path to the server script
        
    Returns:
        Dictionary with Cursor configuration
    """
    return {
        "name": name,
        "command": f"python -m hapax_docs.server",
        "type": "stdio"
    }
    
if __name__ == "__main__":
    main() 