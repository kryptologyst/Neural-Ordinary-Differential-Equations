#!/usr/bin/env python3
"""Script to run the Neural ODEs demo."""

import subprocess
import sys
import os


def main():
    """Run the Streamlit demo."""
    demo_path = os.path.join(os.path.dirname(__file__), "demo", "app.py")
    
    if not os.path.exists(demo_path):
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", demo_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
