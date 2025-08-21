"""
Streamlit App Launcher for Earthquake Map Dashboard
Run this file to start the interactive web dashboard
"""

import streamlit as st
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.earthquake_map import create_streamlit_app

if __name__ == "__main__":
    create_streamlit_app()
