#!/bin/bash
# Launch script for WMISPN GUI

echo "🚀 Starting WMISPN GUI..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Launch the GUI
streamlit run wmispn_app.py
