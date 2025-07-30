#!/usr/bin/env python3
"""
Simple Streamlit App Runner for EV Cost Predictor
=================================================

This script runs the basic Streamlit interface for EV cost prediction.

Usage:
    python run_streamlit.py
"""

import subprocess
import sys
import os

def main():
    print("🔌 Starting EV Cost Predictor (Streamlit)")
    print("📍 App will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit application stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

if __name__ == "__main__":
    main() 