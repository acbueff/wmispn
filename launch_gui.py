#!/usr/bin/env python3
"""
WMI-SPN GUI Launcher
Choose between Streamlit web interface or Tkinter desktop application
"""

import sys
import subprocess
import os
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print(" " * 15 + "WMI-SPN GUI Launcher")
    print("=" * 60)
    print()


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")

    return missing


def launch_streamlit():
    """Launch Streamlit web interface"""
    try:
        import streamlit
        script_path = Path(__file__).parent / "wmi_spn_gui.py"
        print(f"\nLaunching Streamlit interface...")
        print(f"If browser doesn't open automatically, go to: http://localhost:8501")
        print()
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])
    except ImportError:
        print("\nError: Streamlit is not installed.")
        print("Install it with: pip install streamlit")
        print()


def launch_tkinter():
    """Launch Tkinter desktop interface"""
    try:
        import tkinter
        script_path = Path(__file__).parent / "wmi_spn_gui_tkinter.py"
        print(f"\nLaunching Tkinter desktop application...")
        subprocess.run([sys.executable, str(script_path)])
    except ImportError:
        print("\nError: Tkinter is not available.")
        print("On Linux, install with: sudo apt-get install python3-tk")
        print()


def main():
    """Main launcher function"""
    print_banner()

    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"\nWarning: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print("\nWhich interface would you like to use?")
    print()
    print("1. Streamlit Web Interface (Recommended)")
    print("   - Modern, interactive web-based UI")
    print("   - Rich visualizations")
    print("   - Runs in your web browser")
    print()
    print("2. Tkinter Desktop Application")
    print("   - Traditional desktop application")
    print("   - Standalone, no browser needed")
    print("   - Lighter weight")
    print()
    print("3. Exit")
    print()

    while True:
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            launch_streamlit()
            break
        elif choice == "2":
            launch_tkinter()
            break
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLauncher interrupted. Goodbye!")
        sys.exit(0)
