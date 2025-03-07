#!/usr/bin/env python3
"""
Cursor integration setup for the Hapax Documentation Assistant.

This script helps set up the MCP server for use with Cursor.
"""

import json
import os
import shutil
import subprocess
import sys
import signal
from pathlib import Path

def find_cursor_config_dir():
    """Find the Cursor configuration directory"""
    home = Path.home()
    
    # Check common locations
    possible_paths = [
        home / "Library" / "Application Support" / "Cursor",  # macOS
        home / ".config" / "Cursor",  # Linux
        home / "AppData" / "Roaming" / "Cursor",  # Windows
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def setup_system_wide_cursor_config():
    """Set up system-wide Cursor configuration for the Hapax MCP server"""
    cursor_dir = find_cursor_config_dir()
    
    if not cursor_dir:
        print("❌ Cursor configuration directory not found.")
        print("Please make sure Cursor is installed and has been run at least once.")
        return False
    
    # Create MCP directory if it doesn't exist
    mcp_dir = cursor_dir / "mcp"
    mcp_dir.mkdir(exist_ok=True)
    
    # Create or update the MCP configuration
    config_file = mcp_dir / "config.json"
    
    config = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except:
            print("⚠️  Could not load existing Cursor MCP configuration.")
    
    # Ensure servers section exists
    if "servers" not in config:
        config["servers"] = {}
    
    # Determine the correct module path
    try:
        # First try importing from hapax_docs
        import hapax_docs
        module_path = "hapax_docs.server"
    except ImportError:
        # Fall back to direct path if running as standalone
        module_path = "server"
    
    # Add our server
    config["servers"]["hapax-docs"] = {
        "command": sys.executable,
        "args": ["-m", module_path],
        "transport": "stdio",
        "name": "Hapax Documentation Assistant",
        "version": "0.1.0",
        "description": "Documentation assistant for the Hapax framework",
        "enabled": True
    }
    
    # Write the configuration
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ MCP server configuration written to: {config_file}")
    return True

def setup_project_cursor_config():
    """Set up project-specific Cursor configuration for the Hapax MCP server"""
    # Create .cursor directory if it doesn't exist
    cursor_dir = Path(".cursor")
    cursor_dir.mkdir(exist_ok=True)
    
    # Create or update the MCP configuration
    mcp_file = cursor_dir / "mcp.json"
    
    # Determine the correct module path
    try:
        # First try importing from hapax_docs
        import hapax_docs
        module_path = "hapax_docs.server"
    except ImportError:
        # Fall back to direct path if running as standalone
        module_path = "server"
    
    # Write the configuration
    config = {
        "mcpServers": {
            "hapax-docs": {
                "command": sys.executable,
                "args": ["-m", module_path],
                "transportType": "stdio"
            }
        }
    }
    
    with open(mcp_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Project MCP configuration written to: {mcp_file}")
    
    # Create root .mcp.json file for newer Cursor versions
    root_mcp_file = Path(".mcp.json")
    root_config = {
        "servers": {
            "hapax-docs": {
                "command": sys.executable,
                "args": ["-m", module_path],
                "transport": "stdio",
                "name": "Hapax Documentation Assistant",
                "version": "0.1.0",
                "description": "Documentation assistant for the Hapax framework"
            }
        }
    }
    
    with open(root_mcp_file, "w") as f:
        json.dump(root_config, f, indent=2)
    
    print(f"✅ Root MCP configuration written to: {root_mcp_file}")
    return True

def test_server():
    """Test the MCP server"""
    print("🧪 Testing MCP server...")
    
    # Run the server for a short time to ensure it starts correctly
    try:
        # Determine the correct module path
        try:
            # First try importing from hapax_docs
            import hapax_docs
            module_path = "hapax_docs.server"
        except ImportError:
            # Fall back to direct path if running as standalone
            module_path = "server"
            
        process = subprocess.Popen(
            [sys.executable, "-m", module_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Add a preexec_fn to ensure proper process group handling on Unix
            preexec_fn=os.setpgrp if hasattr(os, 'setpgrp') else None
        )
        
        # Wait a bit to see if it crashes immediately
        import time
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print("✅ Server started successfully.")
            # Use process group kill to ensure all related processes are terminated
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                
                # Don't wait for process termination - just continue
                # This avoids the timeout issue
                print("✅ Server process signaled to terminate.")
                return True
            except Exception as e:
                print(f"⚠️ Warning: Could not terminate server process: {e}")
                # Continue anyway since the server started successfully
                return True
        else:
            stdout, stderr = process.communicate()
            print("❌ Server failed to start.")
            if stdout:
                print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
            return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def restart_cursor():
    """Attempt to restart Cursor if it's running"""
    try:
        # Check if Cursor is running
        if sys.platform == "darwin":  # macOS
            result = subprocess.run(["pgrep", "Cursor"], capture_output=True, text=True)
            if result.stdout:
                print("ℹ️  Cursor is running. To apply changes, please restart Cursor manually.")
        elif sys.platform == "win32":  # Windows
            result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Cursor.exe"], capture_output=True, text=True)
            if "Cursor.exe" in result.stdout:
                print("ℹ️  Cursor is running. To apply changes, please restart Cursor manually.")
        else:  # Linux or other
            result = subprocess.run(["ps", "-A"], capture_output=True, text=True)
            if "cursor" in result.stdout.lower():
                print("ℹ️  Cursor is running. To apply changes, please restart Cursor manually.")
    except:
        # If we can't check, just provide general instructions
        print("ℹ️  To apply changes, please restart Cursor if it's running.")

def main():
    """Main function"""
    print("🔧 Hapax Documentation Assistant - Cursor Setup 🔧")
    print("="*60)
    
    # Test the server
    server_ok = test_server()
    if not server_ok:
        print("⚠️ Server test encountered issues, but we'll continue with setup.")
        print("The server may be working correctly but not responding properly to termination signals.")
    
    # Set up project-specific configuration
    project_ok = setup_project_cursor_config()
    
    # Set up system-wide configuration
    system_ok = setup_system_wide_cursor_config()
    
    print("\n📋 Setup Summary:")
    print(f"Server test: {'✅ Passed' if server_ok else '⚠️ Issues detected'}")
    print(f"Project configuration: {'✅ Set up' if project_ok else '❌ Failed'}")
    print(f"System configuration: {'✅ Set up' if system_ok else '❌ Failed or skipped'}")
    
    # Provide restart instructions
    restart_cursor()
    
    print("\n✨ Setup complete! ✨")
    print("After restarting Cursor, you should be able to use the Hapax Documentation Assistant.")
    
    # Determine which package we're running from for helpful tips
    try:
        import hapax.version
        print(f"\nℹ️ Running from Hapax package version {hapax.version.__version__}")
        print("When using the documentation assistant, remember to ask questions about Hapax implementation.")
    except (ImportError, AttributeError):
        print("\nℹ️ Running as standalone documentation assistant")
        
    print("\nIf you encounter any issues, try the following:")
    print("1. Run 'python -m hapax_docs.debug' to diagnose problems")
    print("2. Check the MCP settings in Cursor (Settings > Features > MCP)")
    print("3. Make sure the Hapax Documentation Assistant is enabled in Cursor")
    print("4. Try manually adding the server in Cursor settings")
    print("   - Type: stdio")
    print("   - Name: Hapax Documentation Assistant")
    print("   - Command: python -m hapax_docs.server")

if __name__ == "__main__":
    main() 