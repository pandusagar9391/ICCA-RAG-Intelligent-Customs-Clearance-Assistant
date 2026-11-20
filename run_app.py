"""
ICCA-RAG Web Application Launcher.

Runs the Streamlit interface:
    python run_app.py [--port 8501]

Or directly:
    streamlit run icca/ui/app.py
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "icca" / "ui" / "app.py"


def main():
    print("=" * 70)
    print("  ICCA-RAG: Intelligent Customs Clearance Assistant")
    print("  Web Application Launcher")
    print("=" * 70)

    # Check if streamlit is installed
    try:
        import streamlit
        print(f"[ICCA-RAG] Streamlit detected (version {streamlit.__version__}).")
    except ImportError:
        print("\n[!] Streamlit is not installed in the current Python environment.")
        print("    To run the web interface, please install it:")
        print("        pip install streamlit")
        print("\n    Alternatively, you can run the interactive CLI immediately:")
        print("        python run_cli.py\n")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH)
    ]

    # Append any extra command-line arguments passed to run_app.py
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    print(f"[ICCA-RAG] Launching Streamlit UI from: {APP_PATH}")
    print(f"[ICCA-RAG] Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[ICCA-RAG] Web server stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Streamlit application exited with code: {e.returncode}")


if __name__ == "__main__":
    main()
