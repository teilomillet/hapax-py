#!/usr/bin/env python3
"""
Example script demonstrating how to use Hapax with the Documentation Assistant.

This script shows how to:
1. Set up a project with Hapax
2. Enable the documentation assistant
3. Create a simple graph

To run this example:
    uv run examples/hapax_assistant_example.py

Or:
    python examples/hapax_assistant_example.py
"""

import sys

def check_installation():
    """Check if Hapax and the Documentation Assistant are installed."""
    try:
        import hapax
        print(f"✅ Hapax is installed (version: {hapax.__version__})")
    except ImportError:
        print("❌ Hapax is not installed. Install with:")
        print("   uv add hapax")
        return False
    
    try:
        from hapax_docs import __version__ as docs_version
        print(f"✅ Hapax Documentation Assistant is available (version: {docs_version})")
    except ImportError:
        print("⚠️ Hapax Documentation Assistant is not available.")
        print("   To enable it, install Hapax with the 'assistant' extra:")
        print("   uv add 'hapax[assistant]'")
        return False
    
    return True

def setup_cursor_integration():
    """Set up Cursor integration for the documentation assistant."""
    try:
        from hapax_docs import cursor_setup
        print("Setting up Cursor integration...")
        cursor_setup.main()
    except ImportError:
        print("❌ Cannot set up Cursor integration: Documentation Assistant not available")
        return False
    
    return True

def create_simple_graph():
    """Create a simple Hapax graph to demonstrate basic functionality."""
    try:
        from hapax import Graph, Operation, types
        
        # Define a simple addition operation
        class Add(Operation[types.Tuple[float, float], float]):
            def __init__(self):
                super().__init__()
            
            def forward(self, inputs):
                a, b = inputs
                return a + b
        
        # Create a graph
        graph = Graph()
        
        # Add operations
        inputs = graph.add_input("inputs", type=types.Tuple[float, float])
        add_op = graph.add_operation(Add(), name="add")
        output = graph.add_output("result", type=float)
        
        # Connect operations
        graph.connect(inputs, add_op)
        graph.connect(add_op, output)
        
        # Execute the graph
        result = graph.execute((3.0, 4.0))
        
        print(f"Graph execution result: {result}")
        print("✅ Successfully created and executed a simple Hapax graph")
        
        return True
    except ImportError:
        print("❌ Cannot create graph: Hapax not available")
        return False
    except Exception as e:
        print(f"❌ Error creating graph: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("Hapax with Documentation Assistant Example")
    print("=" * 60)
    
    # Check installation
    installed = check_installation()
    if not installed:
        print("\nPlease install Hapax with the required extras and try again.")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    print("Setting up Cursor Integration")
    print("-" * 60)
    
    # Set up Cursor integration
    setup_cursor_integration()
    
    print("\n" + "-" * 60)
    print("Creating a Simple Hapax Graph")
    print("-" * 60)
    
    # Create a simple graph
    create_simple_graph()
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("""
Next steps:
1. Open your project in Cursor
2. Ask questions about Hapax implementation, like:
   - "How do I create a more complex graph with multiple operations?"
   - "Show me how to implement a custom operation that processes text"
   - "Explain how to handle errors in a Hapax graph"
""")
    print("=" * 60)

if __name__ == "__main__":
    main() 