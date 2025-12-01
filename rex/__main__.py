"""Command-line entry point for the Rex AI Assistant package.

This module allows running Rex as a Python module:
    python -m rex

It also enables installation as a console script via pyproject.toml.
"""

from __future__ import annotations

import sys

# Import the main function from the top-level script
# This maintains backward compatibility while providing a proper package entry point
try:
    from rex_assistant import main
except ImportError:
    # Fallback if the module structure is different
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from rex_assistant import main


if __name__ == "__main__":
    sys.exit(main())
