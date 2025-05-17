#!/bin/bash

# Exit on error
set -e

# Python version
echo "Python version:"
python --version

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install setuptools wheel

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

echo "Build completed successfully!" 