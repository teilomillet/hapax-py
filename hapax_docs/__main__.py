"""Main entry point for running the Hapax Documentation Assistant as a module.

This allows the server to be run with `python -m hapax_docs`.
"""

from .cli import main

if __name__ == "__main__":
    main() 