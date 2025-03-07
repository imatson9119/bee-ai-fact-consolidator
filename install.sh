#!/bin/bash

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
    echo "Error: Python 3.9 or higher is required. Found Python $python_version"
    exit 1
fi

echo "Installing Bee AI Fact Consolidator..."

# Check if pipenv is installed
if command -v pipenv &> /dev/null; then
    echo "Using pipenv for installation..."
    pipenv install
    echo "Installation complete! Run 'pipenv run python fact_consolidator.py' to start."
else
    echo "Pipenv not found. Using pip for installation..."
    pip3 install -e .
    echo "Installation complete! Run 'bee-fact-consolidator' to start."
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit the .env file and add your Bee AI API token."
fi

echo "Done!" 