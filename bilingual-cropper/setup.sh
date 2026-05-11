#!/bin/bash

# Install system dependencies (for Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y poppler-utils libgl1-mesa-glx libglib2.0-0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/uploads data/crops_hindi data/crops_english data/examples

echo "Setup complete. Run 'streamlit run app.py' to start."
