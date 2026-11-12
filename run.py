#!/usr/bin/env python3
"""
SSIM LAB Launcher
One-command entry point to run unit tests or launch the interactive research studio.
"""

import sys
import os
import argparse
import webbrowser

# Add current directory to path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def run_tests():
    """Runs the formal pytest suite."""
    print("🔬 Running SSIM LAB mathematical verification test suite...")
    import pytest
    exit_code = pytest.main(["-v", os.path.join(BASE_DIR, "tests")])
    sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(description="SSIM LAB — Intelligent Image Quality Platform")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server (default: 8000)")
    parser.add_argument("--test", action="store_true", help="Run the test suite and exit")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")
    args = parser.parse_args()

    if args.test:
        run_tests()

    from app.server import start_server

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    start_server(args.port)

if __name__ == "__main__":
    main()
