"""CLI entry point for StatWorkbench."""

import sys


def main() -> int:
    """Launch the StatWorkbench application."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401

        from statworkbench.app import StatWorkbenchApp
    except ImportError as e:
        print(f"Error: Required dependency not found: {e}", file=sys.stderr)
        print("Install with: pip install -e '.[dev]'", file=sys.stderr)
        return 1

    app = StatWorkbenchApp()
    return int(app.run())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
